# Standard Library
import unittest
from typing import Any
from unittest.mock import MagicMock, call

# Third Party
import torch
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.messages import (
    ForwardExecutionResult,
    ModuleExecutionRequest,
)
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.server import InvalidRequestError
from smdistributed.modelparallel.torch.worker import WorkerHolder


class TestServer(unittest.TestCase):
    def test_worker_exec_local_req(self):
        smp.init({"microbatches": 2, "pipeline": "simple", "partitions": 1})
        m0 = nn.Linear(3, 3)
        m0.to(torch.device("cuda", smp.local_rank()))
        m = smp.DistributedModel(m0)
        i = torch.ones(1, 3).to(torch.device("cuda", smp.local_rank()))
        smp.state.module_manager.assign_partition(m, 0)
        smp.state.module_manager.assign_partition(m0, 0)
        mer = ModuleExecutionRequest("main", (i,), {}, ["main"], 0, 0, 0, MbStatus.FWD)
        # first id is not 0 when we run many tests
        next_worker_id = WorkerHolder._id
        # in regular execution a request from same rank is never enqueued like this
        saved_process_result = smp.state.exec_server.process_result
        smp.state.exec_server.process_result = MagicMock()
        smp.state.exec_server.process_request(mer)
        # TODO(anisub): Need to switch to args instead of str of call_args_list
        # This can only be done after pipeline switches to 3.8
        assert str(smp.state.exec_server.process_result.call_args_list) == str(
            [call(ForwardExecutionResult(mer, None))]
        )
        smp.state.exec_server.process_result = saved_process_result

    def test_worker_resume_req(self):
        smp.init({"microbatches": 2, "pipeline": "simple", "partitions": 1})
        m0 = nn.Linear(3, 3)
        m0.to(torch.device("cuda", smp.local_rank()))
        m = smp.DistributedModel(m0)
        i = torch.ones(1, 3).to(torch.device("cuda", smp.local_rank()))
        mer = ModuleExecutionRequest("main", (i,), {}, ["main"], 0, 0, 0, MbStatus.FWD)
        smp.state.module_manager.assign_partition(m, 0)
        smp.state.module_manager.assign_partition(m0, 0)
        smp.state._function_call_count = 0
        # in regular execution a request from same rank is never enqueued like this
        def fake_patch(module: nn.Module, *args: Any, **kwargs: Any):
            smp.state.module_manager.record_traversal_into_module(module)
            if smp.state._function_call_count == 1:
                smp.state._function_call_count += 1
                original_forward = smp.state.patch_manager.get_original_method(
                    "forward", module.__class__
                )
                outputs = original_forward(module, *args, **kwargs)
            elif smp.state._function_call_count == 0:
                smp.state._function_call_count += 1
                request = ModuleExecutionRequest.create_forward_req(
                    module, args, kwargs, enable_grads=True, enable_autocast=False
                )
                smp.state.exec_server.record_pending_request(request)
                outputs = smp.state.current_worker.thread_get_forward_result(request)
            smp.state.module_manager.finished_module_exec()
            return outputs

        def fake_patch_module(*args, **kwargs):
            return fake_patch(smp.state.model, *args, **kwargs)

        m.forward = fake_patch_module

        next_worker_id = WorkerHolder._id
        # starts to execute the module, then hits a new request that it's waiting on, then resumes
        try:
            smp.state.exec_server.process_request(mer)
        except InvalidRequestError as exc:
            # as server thinks this is from a worker waiting for something it tries to resume worker
            assert exc.req == mer
        assert smp.state.exec_server.get_worker(next_worker_id).req == mer
        assert smp.state.exec_server.get_worker(next_worker_id + 1).req.module == "main"
        assert smp.state.exec_server.get_worker(next_worker_id + 1).req.execution_stack == [
            "main",
            "main",
        ]


if __name__ == "__main__":
    unittest.main()
