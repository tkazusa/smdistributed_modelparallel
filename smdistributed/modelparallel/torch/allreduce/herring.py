# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.allreduce.reducer import GradReducer
from smdistributed.modelparallel.torch.exceptions import InvalidHandleError, SMPUnsupportedError

logger = get_logger()


class HerringAllreducer(GradReducer):
    def __init__(
        self,
        named_parameters,
        grad_counter,
        backward_passes_per_step,
        overlapping_allreduce,
        average_grads_across_microbatches,
        num_microbatches,
        scaled_batch,
        tp_size,
    ):
        super(HerringAllreducer, self).__init__(
            named_parameters,
            grad_counter,
            backward_passes_per_step,
            overlapping_allreduce,
            average_grads_across_microbatches,
            num_microbatches,
            scaled_batch,
            tp_size,
        )
        self._handles = {}
        for n, p in self.named_parameters.items():
            # will return local parameters
            # init fusion buffers
            raise SMPUnsupportedError

    def _allreduce_grad_async(self, p):
        # handle = herring.allreduce()
        raise SMPUnsupportedError
        return handle

    def _synchronize_internal(self):
        for n, p in self.named_parameters.items():
            if p.requires_grad:
                if not self.overlapping_allreduce or p not in self._handles:
                    if not self.overlapping_allreduce:
                        if self.average_grads_across_microbatches:
                            if self.scaled_batch:
                                param.grad /= float(self.num_microbatches * self.tp_size)
                            else:
                                param.grad /= float(self.num_microbatches)
                    else:
                        pass
                        # this grad should only be 0 if grad hook wasn't called, as long as optimizer.zero_grad was called
                        # so it doesn't matter whether or not we divide by num_microbatches for params whose hook wasn't called
                        # ignoring divide by num microbatches

                    handle, ctx = self._allreduce_grad_async(n, p)
                    self._handles[p] = (handle, ctx)

        size = float(dist.get_world_size())
        for p, (handle, ctx) in self._handles.items():
            if handle == None:
                raise InvalidHandleError(
                    "Handle should not be none, it is expected that handles only has params whose allreduce was called from the hook"
                )
            handle.wait()
            p.grad.data /= size
        self._handles.clear()

    def _hook_internal(self, p, grads):
        handle = self._allreduce_grad_async(p)
        self._handles[p] = handle
