# Standard Library
from abc import ABCMeta, abstractmethod
from enum import Enum


class MbStatus(Enum):
    """
    FWD or BWD it means the microbatch is executing FWD or BWD.
    READY_FOR_FWD means it is ready to be picked for FWD pass.
    READY_FOR_BWD means it is ready to be picked for BWD pass.
    DONE means both stages are done for microbatch
    """

    READY_FOR_FWD = 0
    READY_FOR_BWD = 1
    DONE = 2
    FWD = 3
    BWD = 4


class PTPipeline:
    """
        With module server design, once a mb was passed to execute FWD or BWD, we are guaranteed
        that the task will be done. Server will keep track of the execution and ensure it completes.
        This guarantee simplifies the pipeline design and makes the sequence clean.
        This pipeline will only be used by the step leader.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.init_step()

    def init_step(self):
        from smdistributed.modelparallel.torch.state_mod import state

        self.status = [MbStatus.READY_FOR_FWD for _ in range(state.num_microbatches())]
        self.mb_pointer = None

    @abstractmethod
    def get_next_microbatch(self):
        raise NotImplementedError

    def mark_done(self, mb):
        self.status[mb] = MbStatus.DONE

    def mark_ready_for_backward(self, mb):
        self.status[mb] = MbStatus.READY_FOR_BWD

    def mark_wait_for_backward_done(self, mb):
        self.status[mb] = MbStatus.WAIT_FOR_BWD_DONE

    def get_status(self, mb):
        return self.status[mb]

    def promote_status(self, mb):
        if self.status[mb] == MbStatus.READY_FOR_FWD:
            self.status[mb] = MbStatus.FWD
        elif self.status[mb] == MbStatus.READY_FOR_BWD:
            self.status[mb] = MbStatus.BWD

    def get_next_mb_with_status(self, status, after=None):
        if not isinstance(status, list):
            status = [status]
        for mb, s in enumerate(self.status):
            if after is not None and mb <= after:
                continue
            if s in status:
                return mb
        if after is not None:
            # if after was passed and we didn't find microbatch with specified status, then retry from beginning
            return self.get_next_mb_with_status(status, after=None)
        return None

    def has_more_ticks(self):
        return not all([x is MbStatus.DONE for x in self.status])


class OnlyForwardPipeline(PTPipeline):
    """ A pipeline which does only forward passes for testing
    """

    def get_next_microbatch(self):
        new_mb_pointer = self.get_next_mb_with_status(MbStatus.READY_FOR_FWD, after=self.mb_pointer)
        if new_mb_pointer is not None:
            self.mb_pointer = new_mb_pointer
            return self.mb_pointer

        return None


class SimplePipeline(PTPipeline):
    """ A pipeline which finishes all forward passes and then picks up all backward passes.
        examples:
        F0 F1 B0 B1 F2 F3 B2 B3 F4 F5 B4 B5
        F0 F1 B0 B1 F2 F3 B3 B2 F4 F5 B4 B5
    """

    def get_next_microbatch(self):
        new_mb_pointer = self.get_next_mb_with_status(MbStatus.READY_FOR_FWD, after=self.mb_pointer)

        if new_mb_pointer is None:
            # check if any forward going on
            if self.get_next_mb_with_status(MbStatus.FWD) is None:
                # if there's no forward left execute backward
                new_mb_pointer = self.get_next_mb_with_status(MbStatus.READY_FOR_BWD)
        if new_mb_pointer is not None:
            self.mb_pointer = new_mb_pointer
            return self.mb_pointer

        return None


class InterleavedPipeline(PTPipeline):
    """
        Crux of this pipeline is that we work on forward passes and execute backward for only one microbatch at a time
        when it is possible to do so. Whoever has the final output needs to start the backward pass across all processes,
        so that will be pp_rank=0 in module server design.

        A side note: if consecutive subgraphs are on same device,
        it's equivalent to treating it as one less subgraph.

        gpu2 SG2:       F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5
        gpu1 SG1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7
        gpu0 SG0: F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3 F8 B4

        TODO: Do we want to introduce a limit on how many microbatches can have forward
        pass executed while waiting on a prev microbatch backward pass
        essentially how far ahead can forward pass go compared to backward pass,
        i.e. max(mb for status[mb] != done) - min(mb for status[mb] == backward)
    """

    def get_next_microbatch(self):
        # execute next bwd
        new_mb_pointer = self.get_next_mb_with_status(MbStatus.READY_FOR_BWD)
        if new_mb_pointer is None:
            # execute next fwd
            new_mb_pointer = self.get_next_mb_with_status(MbStatus.READY_FOR_FWD)
        if new_mb_pointer is not None:
            self.mb_pointer = new_mb_pointer
            return self.mb_pointer
        return None
