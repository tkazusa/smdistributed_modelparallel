# Standard Library
import sys
import unittest

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import Net1
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.step import step


class TestLeak(unittest.TestCase):
    @step(detach_outputs=True)
    def train_step(model, data, out_grads):
        output = model(data)
        model.backward(output, out_grads)
        for k, v in state.exec_server.outputs.items():
            # verify that outputs detached at end of microbatch
            assert v.grad_fn is None
        return output

    PARTITIONS = 2

    def check_e2e(smp_fwd=True):
        torch.manual_seed(42)
        cfg = {
            "microbatches": 5,
            "placement_strategy": "spread",
            "pipeline": "simple",
            "optimize": "speed",
            "partitions": TestLeak.PARTITIONS,
            "auto_partition": False,
            "default_partition": 0,
        }

        smp.init(cfg)
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")
        encoder_input_dist = torch.ones(40, 1000)
        encoder_input_dist = encoder_input_dist.to(device)
        encoder_input_dist.requires_grad = True
        mask = torch.Tensor([[0]]) == 1
        mask = mask.to(device)

        def assign_recursive(model, partition):
            for child in model.children():
                state.module_manager.assign_partition(child, partition)
                assign_recursive(child, partition)

        def no_smp_forward():
            # single gpu model

            encoder_input = encoder_input_dist.detach().clone()
            encoder_input.requires_grad = True
            model2 = Net1()
            model2.to(device)
            output = model2(encoder_input)
            out_grads = torch.randn(output.size(), device=torch.device("cuda"))

        def smp_forward():
            # Distributed model
            torch.manual_seed(42)
            model = Net1()
            model.to(device)
            assign_recursive(model, TestLeak.PARTITIONS - 1)
            model = smp.DistributedModel(model)
            torch.cuda.set_device(smp.local_rank())
            out_grads = torch.randn(40, 20, device=torch.device("cuda"))
            output_dist = TestLeak.train_step(model, encoder_input_dist, out_grads)
            output_dist.clear()
            smp.state.module_manager.reset()
            smp.state.patch_manager.reset()
            del model
            smp.state.model = None
            smp.state.optimizer = None

        if smp_fwd:
            smp_forward()
        else:
            no_smp_forward()

    def test_mem_leak(self):
        TestLeak.check_e2e(smp_fwd=True)
        import gc

        gc.collect()
        gc.collect()
        gc.collect()
        print(
            f"Before emptying cache, memory reserved by cache for rank {smp.local_rank()}: {torch.cuda.memory_reserved(smp.local_rank())}"
        )
        torch.cuda.empty_cache()
        print(
            f"After emptying cache, memory reserved by cache for rank {smp.local_rank()}: {torch.cuda.memory_reserved(smp.local_rank())}"
        )
        print(
            f"rank is {smp.local_rank()} mem_allocated: {torch.cuda.memory_allocated(smp.local_rank())}"
        )
        total_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                    total_size += torch.numel(obj.data)
            except:
                pass
        print(f"rank is {smp.local_rank()}, size of unfreed tensors, python: {total_size}")
        self.assertEqual(torch.cuda.memory_reserved(smp.local_rank()), 0)
        self.assertEqual(
            torch.cuda.memory_allocated(smp.local_rank()), 0, f"leak on rank {smp.local_rank()}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TestLeak.PARTITIONS = int(sys.argv.pop())
    unittest.main()
    smp.barrier()
