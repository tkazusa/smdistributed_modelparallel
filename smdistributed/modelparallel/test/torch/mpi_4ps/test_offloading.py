# Standard Library
import gc
import time
import unittest
from collections import namedtuple

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.offload import TensorOffloader
from smdistributed.modelparallel.torch.server import ExecutionServer


def no_clear(self):
    pass


def get_numel_tensors_in_memory(gpu=False):
    gc.collect()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    seen_elem = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                print(obj.shape, obj.device)
                if (gpu and obj.data.is_cuda) or (not gpu and not obj.data.is_cuda):
                    seen_elem += torch.numel(obj.data)
        except:
            pass
    return seen_elem


def train(shard_offloads=False):
    a = torch.rand(1024, 1024, device=0, requires_grad=True)
    b = torch.rand(1024, 1024, device=0, dtype=torch.float16, requires_grad=False)
    total_numel = a.numel() + b.numel()

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    torch.cuda.synchronize()

    TensorOffloader._clear_tensors_for_finished_offloads = no_clear
    offloader = TensorOffloader()
    task, key = offloader.save_for_backward(shard_offloads, a, b)
    time.sleep(0.01)
    del a, b

    if not shard_offloads or smp.tp_rank() == 0:
        assert torch.allclose(offloader.offloaded[task][key][0], a_cpu)
        assert torch.allclose(offloader.offloaded[task][key][1], b_cpu)

    a1, b1 = offloader.saved_tensors(shard_offloads, task, key)

    assert a1.numel() + b1.numel() == total_numel
    assert a1.requires_grad
    assert b1.requires_grad is False
    assert torch.allclose(a_cpu, a1.cpu())
    assert torch.allclose(b_cpu, b1.cpu())

    offloader.offloaded.clear()
    del a_cpu, b_cpu
    del a1, b1
    seen_elem = get_numel_tensors_in_memory(gpu=True)
    # assert that tensor offloaded was referenced and in memory
    if not shard_offloads or smp.tp_rank() == 0:
        assert seen_elem == total_numel, (seen_elem, total_numel, smp.tp_rank())
    else:
        assert seen_elem == 0

    offloader.reset()
    seen_elem_cpu = get_numel_tensors_in_memory(gpu=False)
    # assert that offloaded tensor doesn't remain on cpu
    assert seen_elem_cpu == 0, seen_elem_cpu


class TestActivationOffloading(unittest.TestCase):
    def _init_smp(self, partitions=1, tp_degree=4, offload=True):
        smp.init(
            {
                "microbatches": 1,
                "pipeline": "interleaved",
                "partitions": partitions,
                "tensor_parallel_degree": tp_degree,
                "offload_activations": offload,
                "ddp": True,
            }
        )

    def test_reference_in_memory(self, shard_offloads=False):
        torch.manual_seed(2)
        self._init_smp()
        smp.state.exec_server = ExecutionServer()
        smp.state.exec_server.current_task = namedtuple("Task", "task_metadata dummy")
        smp.state.exec_server.current_task.task_metadata = "dummy"
        smp.state.exec_server.server_queue.current_task = smp.state.exec_server.current_task
        train(shard_offloads=shard_offloads)

    def test_reference_in_memory_shard(self):
        self.test_reference_in_memory(True)


if __name__ == "__main__":
    unittest.main()
