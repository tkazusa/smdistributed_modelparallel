# Standard Library
import unittest

# Third Party
import numpy as np
import torch
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.state_mod import state


class TestModuleReuseHigh(unittest.TestCase):
    """Tests module reuse cases for SMP
    """

    def test_sequential(self):
        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class Net1(torch.nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x):
                with torch.no_grad():
                    out = self.linear1(x)
                return out

        class NetNonSMPSequential(torch.nn.Module):
            def __init__(self):
                super(NetNonSMPSequential, self).__init__()
                self.net1 = Net1()
                self.linear2 = nn.Linear(10, 10)
                self.linear3 = nn.Linear(10, 10)
                self.sequential = torch.nn.Sequential(
                    self.net1, self.linear2, self.net1, self.linear3
                )

            def forward(self, x):
                out2 = self.sequential(x)
                return out2

        smp.reset()
        torch.manual_seed(42)
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        model = NetNonSMPSequential()
        x = torch.randn(10, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = False
        x_nosmp = x.detach().clone()
        model = model.to(device)
        output = model(x_nosmp)
        out_grads = torch.ones(10, 10)
        out_grads = out_grads.to(device)
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.linear3.weight.grad

        cfg = {
            "microbatches": 2,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 4,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model.net1, 1)
        state.module_manager.assign_partition(model.net1.linear1, 1)
        state.module_manager.assign_partition(model.linear2, 2)
        state.module_manager.assign_partition(model.linear3, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        output_dist = train_step(model, x, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu(), atol=1e-6)
        if smp.pp_rank() == 1:
            np.testing.assert_allclose(
                nosmp_grads.cpu(), model.module.linear3.weight.grad.cpu(), atol=1e-6
            )

    def test_module_reuse_high(self):
        @smp.step
        def train_step(model, data1, data2, out_grads):
            output = model(data1, data2)
            model.backward(output, out_grads)
            return output

        class Net3(torch.nn.Module):
            def __init__(self):
                super(Net3, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x1, x2):
                with torch.no_grad():
                    z1 = self.linear1(x1)
                z2 = self.linear1(x2)
                return z1 + z2

        class Net2(torch.nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.net3 = Net3()

            def forward(self, x1, x2):
                out = self.net3(x1, x2)
                return out

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.net2 = Net2()

            def forward(self, x1, x2):
                with torch.no_grad():
                    out1 = self.net2(x1, x2)
                out2 = self.net2(out1, out1)
                return out2

        torch.manual_seed(42)
        model = Net()
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(10, 10)
        model.to(device)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        y = torch.randn(10, 10)
        y = y.to(device)
        y.detach_()
        y.requires_grad = True
        y_nosmp = y.detach().clone()
        x_nosmp.to(device)
        y_nosmp.to(device)
        x_nosmp.requires_grad = True
        y_nosmp.requires_grad = True
        output = model(x_nosmp, y_nosmp)
        out_grads = torch.ones(10, 10)
        out_grads = out_grads.to(device)
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.net2.net3.linear1.weight.grad

        cfg = {
            "microbatches": 1,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 4,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model, 0)
        state.module_manager.assign_partition(model.net2, 1)
        state.module_manager.assign_partition(model.net2.net3, 2)
        state.module_manager.assign_partition(model.net2.net3.linear1, 3)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        out_grads = torch.ones(10, 10)
        out_grads = out_grads.to(device)
        output_dist = train_step(model, x, y, out_grads)

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(
                output.detach().cpu(), output_dist.outputs[0].detach().cpu(), atol=1e-6
            )

        if smp.pp_rank() == 3:
            np.testing.assert_allclose(
                nosmp_grads.cpu(), model.module.net2.net3.linear1.weight.grad.cpu(), atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
