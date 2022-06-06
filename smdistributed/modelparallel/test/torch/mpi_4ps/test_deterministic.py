# Standard Library
import random
import unittest

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.messages import ModuleExecutionRequest
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.server_queue import (
    DeterministicServerQueue,
    IncomingMessageTask,
    NextMicrobatchTask,
)
from smdistributed.modelparallel.torch.state_mod import state


class MockServer:
    def __init__(self, queue, rounds):
        self.queue = queue
        self.rounds = rounds

    def run(self):
        order = []
        if smp.pp_rank() == 0:
            while self.queue.has_more_tasks():
                order.append(self.queue.get_next_task())
        else:
            for i in range(self.rounds):
                order.append(self.queue.get_next_task())
        return order


class MockPipeline:
    def __init__(self, rounds):
        self.mbs = list(range(rounds))
        self.current = 0

    def get_status(self, mb):
        return MbStatus.READY_FOR_BWD

    def has_more_ticks(self):
        return self.current < len(self.mbs)

    def get_next_microbatch(self):
        mb = self.mbs[self.current]
        self.current += 1
        return mb

    def reset_and_shuffle(self):
        random.shuffle(self.mbs)
        self.current = 0


class MockServerComm:
    def __init__(self, rounds, pipeline):
        self.messages = [
            ModuleExecutionRequest("module_%d" % i, (0,), {}, ["main"], 0, 0, 0, MbStatus.FWD)
            for i in range(rounds)
        ]
        self.current = 0
        self.pipeline = pipeline

    def has_message(self):
        if smp.pp_rank() == 0 and self.pipeline.has_more_ticks() and random.random() < 0.6:
            # if pipeline has more ticks, sometimes randomly return false to better mix
            # IncomingMessageTasks with NextMicrobatchTasks
            return False

        return self.current < len(self.messages)

    def get_next_message(self, block=False):
        msg = self.messages[self.current]
        self.current += 1
        return msg

    def reset_and_shuffle(self):
        random.shuffle(self.messages)
        self.current = 0


class TestDeterministicQueue(unittest.TestCase):
    def is_same(self, x, y):
        if type(x) != type(y):
            return False

        if isinstance(x, NextMicrobatchTask):
            return x.microbatch == y.microbatch

        if isinstance(x, IncomingMessageTask):
            return x.message.module == y.message.module

    def test_deterministic(self):
        smp.init({"partitions": 2, "ddp": True})

        random.seed(123 + smp.rank())

        rounds = 20
        record_step = 0
        pipeline = MockPipeline(rounds)
        comm = MockServerComm(rounds, pipeline)
        queue = DeterministicServerQueue(comm, pipeline)
        queue.record_step = record_step
        queue.set_partitioned()

        server = MockServer(queue, rounds)
        state.current_step_func = lambda: 0

        first_order = server.run()

        smp.barrier()

        queue.mark_step_done()
        queue.pipeline.reset_and_shuffle()
        queue.comm.reset_and_shuffle()

        second_order = server.run()

        # verify that the order of tasks is the same despite the shuffle
        self.assertTrue(
            all([self.is_same(first, second) for first, second in zip(first_order, second_order)])
        )


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
