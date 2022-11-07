# Standard Library
import unittest

# Third Party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import NetTransformer
from smdistributed.modelparallel.torch.state_mod import state

from smdistributed.modelparallel.torch.amp import GradScaler  # noqa isort:skip


class TestParamsInForward(unittest.TestCase):
    # When the model passing the parameter to forward,
    # and the model that parameter is being passed to are on the same device
    # and the parent of model passing the parameter is on a different device
    # In this case, the outputs ancestor calculation will find the parameter
    # and increment its expected grad.
    def test_params_in_fwd(self):
        @smp.step
        def train_step(model, data, out_grads):
            output = model(data)
            model.backward(output, out_grads)
            return output

        class Net3(torch.nn.Module):
            def __init__(self):
                super(Net3, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x, aux_param):
                result = self.linear1(x) + aux_param
                return result

        class Net2(torch.nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.net3 = Net3()
                self.aux_param = nn.Parameter(torch.ones((1, 10), requires_grad=True))

            def forward(self, x):
                out2 = self.net3(x, self.aux_param)
                return out2

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.net2 = Net2()

            def forward(self, x):
                out1 = self.net2(x)
                out2 = self.net2(x)
                return out1 + out2

        cfg = {
            "microbatches": 4,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": True,
            "default_partition": 0,
        }

        torch.cuda.set_device(0)
        device = torch.device("cuda")
        smp.reset()
        torch.manual_seed(42)
        model = Net()
        model.to(device)
        out_grads = torch.ones(16, 10)
        out_grads = out_grads.to(device)
        x = torch.randn(16, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        x_nosmp.requires_grad = True
        x_nosmp = x_nosmp.to(device)
        out_grads = torch.ones(16, 10)
        out_grads = out_grads.to(device)
        output = model(x_nosmp)
        torch.autograd.backward(output, out_grads)
        nosmp_param_grad = model.net2.aux_param.grad.clone()

        smp.init(cfg)

        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        output_dist = train_step(model, x, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())
            np.testing.assert_allclose(
                nosmp_param_grad.cpu(), model.module.net2.aux_param.grad.cpu()
            )

    # Here the module to which param is passed on a different rank
    # compared to the module passing the param in forward.
    # So, the expected grads is incremented on the rank which passes
    # the param. The rank which executes the module (getting
    # the param in forward) treats it as just another tensor and doesnt
    # increment expected grads.
    def test_params_in_fwd_main(self):
        @smp.step()
        def train_step(model, data, out_grads):
            output = model(data)
            model.backward(output, out_grads)
            return output

        class Net2(torch.nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear1(x)

        class Net1(torch.nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                self.net2 = Net2()

            def forward(self, x, aux_param):
                result = self.net2(x) + aux_param
                return result

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.net1 = Net1()
                self.aux_param = nn.Parameter(torch.ones((1, 10), requires_grad=True))

            def forward(self, x):
                result = self.net1(x, self.aux_param)
                return result

        cfg = {
            "microbatches": 4,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        smp.reset()
        torch.manual_seed(42)
        x = torch.randn(16, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        x_nosmp.requires_grad = True
        x_nosmp = x_nosmp.to(device)
        out_grads = torch.ones(16, 10)
        out_grads = out_grads.to(device)
        model = Net()
        model.to(device)
        output = model(x_nosmp)
        torch.autograd.backward(output, out_grads)
        nosmp_param_grad = model.aux_param.grad.clone()
        smp.init(cfg)
        state.module_manager.assign_partition(model, 0)
        state.module_manager.assign_partition(model.net1, 1)
        state.module_manager.assign_partition(model.net1.net2, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        output_dist = train_step(model, x, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())
            np.testing.assert_allclose(nosmp_param_grad.cpu(), model.module.aux_param.grad.cpu())

    # Sequential test, to make sure that the grad counting happens correctly
    # for the param passed in forward
    def test_sequential_first_in_chain(self):
        @smp.step
        def train_step(model, out_grads):
            output = model()
            model.backward(output, out_grads)
            return output

        class NetSequential(torch.nn.Module):
            def __init__(self):
                super(NetSequential, self).__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)
                self.seq = nn.Sequential(self.linear1, self.linear2)
                self.aux_param = nn.Parameter(torch.ones((16, 10), requires_grad=True))

            def forward(self):
                return self.seq(self.aux_param)

        torch.manual_seed(42)
        cfg = {
            "microbatches": 1,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        device = torch.device("cuda")
        x = torch.ones(16, 10)
        x = x.to(device)
        x_nosmp = x.detach().clone()
        out_grads = torch.randn(16, 10)
        out_grads = out_grads.to(device)
        model = NetSequential()
        model.to(device)
        output = model()
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.linear1.weight.grad
        state.module_manager.assign_partition(model.linear1, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, out_grads)
        dist_result = output_dist.concat()
        if smp.mp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.mp_rank() == 1:
            np.testing.assert_allclose(nosmp_grads.cpu(), model.module.linear1.weight.grad.cpu())


class TestModuleBuffers(unittest.TestCase):
    """Tests module buffers for partitioned model
    """

    def test_buffers(self):
        @smp.step
        def train_step(model, data):
            output = model(data)
            return output

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.identity = nn.Identity()
                self.bn = nn.BatchNorm2d(100)

            def forward(self, x):
                return self.bn(self.identity(x))

        cfg = {
            "microbatches": 4,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.reset()
        torch.manual_seed(42)
        model = Net()
        smp.init(cfg)
        state.module_manager.assign_partition(model, 0)
        state.module_manager.assign_partition(model.identity, 0)
        state.module_manager.assign_partition(model.bn, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        x = torch.randn(20, 100, 35, 45)
        output_dist = train_step(model, x)
        dev = torch.device("cuda", 1)
        for name, buff in model._local_buffers.items():
            assert (
                buff.device == dev
            ), f"buffer: {name} is not on the right device, expected on: {dev}, present on: {buff.device}"


class TestMultipleLevels(unittest.TestCase):
    """Tests module multiple levels
    """

    def test_broken_path_check_correctness(self):
        """Test parent check correctness
        """

        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class Net3(torch.nn.Module):
            def __init__(self):
                super(Net3, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear1(x)

        class Net2(torch.nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.net3 = Net3()

            def forward(self, x):
                out2 = self.net3(x)
                return out2

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.net2 = Net2()

            def forward(self, x):
                out1 = self.net2(x)
                out2 = self.net2(x)
                return out1 + out2

        smp.reset()
        torch.manual_seed(42)
        model = Net()
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(16, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        x_nosmp.requires_grad = True
        model.to(device)
        output = model(x_nosmp)
        out_grads = torch.ones(16, 10)
        out_grads = out_grads.to(device)
        torch.autograd.backward(output, out_grads)
        cfg = {
            "microbatches": 4,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model, 0)
        state.module_manager.assign_partition(model.net2, 1)
        state.module_manager.assign_partition(model.net2.net3, 0)
        state.module_manager.assign_partition(model.net2.net3.linear1, 0)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        output_dist = train_step(model, x, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())


class TestModuleReuse(unittest.TestCase):
    """Tests module reuse cases for SMP
    """

    def test_same_child_diff_requires_grad(self):
        """Test same child called twice one with inputs: requires_grad,
        other case where inputs don't require_grads
        """

        @smp.step
        def train_step(model, data1, data2, out_grads):
            output = model(data1, data2)
            model.backward(output, out_grads)
            return output

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x, y):
                out1 = self.linear1(x)
                out2 = self.linear1(y)
                return out1 + out2

        smp.reset()
        torch.manual_seed(42)
        model = Net()
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(16, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = False
        x_nosmp = x.detach().clone()
        y = torch.randn(16, 10)
        y = y.to(device)
        y.detach_()
        y.requires_grad = True
        y_nosmp = y.detach().clone()
        y_nosmp.requires_grad = True
        model.to(device)
        output = model(x_nosmp, y_nosmp)
        out_grads = torch.ones(16, 10)
        out_grads = out_grads.to(device)
        torch.autograd.backward(output, out_grads)

        cfg = {
            "microbatches": 4,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model.linear1, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        output_dist = train_step(model, x, y, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(y_nosmp.grad.cpu(), y.grad.cpu())

    def test_multiple_child_non_requires_grad(self):
        """Test same child called twice one with inputs: requires_grad,
        other case where inputs don't require_grads
        """

        @smp.step
        def train_step(model, data1, data2, out_grads):
            output = model(data1, data2)
            model.backward(output, out_grads)
            return output

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x, y):
                out1 = self.linear1(x)
                out2 = self.linear1(y)
                return out1 + out2

        smp.reset()
        torch.manual_seed(42)
        model = Net()
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(16, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = False
        x_nosmp = x.detach().clone()
        y = torch.randn(16, 10)
        y = y.to(device)
        y.detach_()
        y.requires_grad = False
        y_nosmp = y.detach().clone()
        y_nosmp.requires_grad = False
        model.to(device)
        output = model(x_nosmp, y_nosmp)
        out_grads = torch.ones(16, 10)
        out_grads = out_grads.to(device)
        torch.autograd.backward(output, out_grads)
        nosmp_grad = model.linear1.weight.grad

        cfg = {
            "microbatches": 4,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model, 0)
        state.module_manager.assign_partition(model.linear1, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        output_dist = train_step(model, x, y, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())
        if smp.pp_rank() == 1:
            np.testing.assert_allclose(
                model.module.linear1.weight.grad.cpu(), nosmp_grad.cpu(), atol=1e-5
            )

    def test_requires_grad_reordering(self):
        """Test same child called twice one with inputs: requires_grad,
        other case where inputs don't require_grads
        """

        @smp.step
        def train_step(model, data1, data2, out_grads):
            output = model(data1, data2)
            model.backward(output, out_grads)
            return output

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = nn.Linear(10, 10)

            def forward(self, x, y):
                out2 = self.linear1(y)
                out1 = self.linear1(x)
                return out2 + out1

        smp.reset()
        torch.manual_seed(42)
        model = Net()
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(10, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = False
        x_nosmp = x.detach().clone()
        y = torch.randn(10, 10)
        y = y.to(device)
        y.detach_()
        model.to(device)
        y.requires_grad = True
        y_nosmp = y.detach().clone()
        x_nosmp.requires_grad = False
        y_nosmp.requires_grad = True
        output = model(x_nosmp, y_nosmp)
        out_grads = torch.ones(10, 10)
        out_grads = out_grads.to(device)
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.linear1.weight.grad

        cfg = {
            "microbatches": 2,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model.linear1, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        out_grads = torch.ones(10, 10)
        out_grads = out_grads.to(device)
        output_dist = train_step(model, x, y, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(y_nosmp.grad.cpu(), y.grad.cpu())

        if smp.pp_rank() == 1:
            np.testing.assert_allclose(
                nosmp_grads.cpu(), model.module.linear1.weight.grad.cpu(), atol=1e-5
            )

    def test_multiple_parents(self):
        """Test multiple parents for one child module
        """

        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class Net1(torch.nn.Module):
            def __init__(self, child=None):
                super(Net1, self).__init__()
                self.linear1 = child

            def forward(self, x):
                out1 = self.linear1(x)
                return out1

        class Net2(torch.nn.Module):
            def __init__(self, child=None):
                super(Net2, self).__init__()
                self.linear1 = child

            def forward(self, x):
                out1 = self.linear1(x)
                return out1

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(10, 10)
                self.child1 = Net1(child=self.linear)
                self.child2 = Net2(child=self.linear)

            def forward(self, x):
                out = self.child1(x) + self.child2(x)
                return out

        smp.reset()
        torch.manual_seed(42)
        model = Net()
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(10, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        model.to(device)
        x_nosmp.requires_grad = True
        output = model(x_nosmp)
        out_grads = torch.ones(10, 10)
        out_grads = out_grads.to(device)
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.linear.weight.grad
        cfg = {
            "microbatches": 2,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model.linear, 0)
        state.module_manager.assign_partition(model.child1, 0)
        state.module_manager.assign_partition(model.child2, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        out_grads = torch.ones(10, 10)
        out_grads = out_grads.to(device)
        output_dist = train_step(model, x, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())
            np.testing.assert_allclose(
                nosmp_grads.cpu(), model.module.linear.weight.grad.cpu(), atol=1e-5
            )


class TestSequentialFirstInChainNotRequiresGrad(unittest.TestCase):
    def test_sequential_first_in_chain(self):
        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class NetSequential(torch.nn.Module):
            def __init__(self):
                super(NetSequential, self).__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)
                self.seq = nn.Sequential(self.linear1, self.linear2)

            def forward(self, x):
                return self.seq(x)

        torch.manual_seed(42)
        cfg = {
            "microbatches": 2,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        device = torch.device("cuda")
        x = torch.ones(16, 10)
        x = x.to(device)
        x_nosmp = x.detach().clone()
        out_grads = torch.randn(16, 10)
        out_grads = out_grads.to(device)
        model = NetSequential()
        model.to(device)
        output = model(x_nosmp)
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.linear1.weight.grad
        state.module_manager.assign_partition(model.linear1, 1)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, x, out_grads)
        dist_result = output_dist.concat()
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 1:
            np.testing.assert_allclose(
                nosmp_grads.cpu(), model.module.linear1.weight.grad.cpu(), atol=1e-5
            )


class TestDummy(unittest.TestCase):
    def test_dummy_backward_interleaved(self):
        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class DummyEmbedding(torch.nn.Module):
            def __init__(self):
                super(DummyEmbedding, self).__init__()
                self.embedding1 = nn.Embedding(10, 3)

            def forward(self, x):
                return self.embedding1(x)

        class NetDummy(torch.nn.Module):
            def __init__(self):
                super(NetDummy, self).__init__()
                self.dummy_embedding = DummyEmbedding()

            def forward(self, x):
                return self.dummy_embedding(x)

        torch.manual_seed(42)
        cfg = {
            "microbatches": 2,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        device = torch.device("cuda")
        x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        x = x.to(device)
        x.detach_()
        out_grads = torch.randn(2, 4, 3)
        out_grads = out_grads.to(device)
        model = NetDummy()
        model.to(device)
        output = model(x)
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.dummy_embedding.embedding1.weight.grad.clone()
        model.zero_grad()
        smp.init(cfg)
        state.module_manager.assign_partition(model.dummy_embedding, 1)
        state.module_manager.assign_partition(model.dummy_embedding.embedding1, 0)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, x, out_grads)
        dist_result = output_dist.concat()

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), dist_result.detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(
                nosmp_grads.cpu(), model.module.dummy_embedding.embedding1.weight.grad.cpu()
            )

    def test_sequential_multi_inputs(self):
        @smp.step
        def train_step(model, data1, data2, out_grads):
            output = model((data1, data2))
            model.backward(output, out_grads)
            return output

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)

            def forward(self, inp):
                x, y = inp
                z = self.linear1(x)
                z2 = self.linear2(y)
                return z, z2

        class Net2(torch.nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)

            def forward(self, inp):
                x, y = inp
                with torch.no_grad():
                    z = self.linear1(x)
                    z2 = self.linear2(y)
                return z, z2

        class NetSequentialMultiInputs(torch.nn.Module):
            def __init__(self):
                super(NetSequentialMultiInputs, self).__init__()
                self.net1 = Net()
                self.net2 = Net2()
                self.sequential = torch.nn.Sequential(self.net1, self.net2, self.net1)

            def forward(self, x):
                out2 = self.sequential(x)
                return out2

        smp.reset()
        torch.manual_seed(42)
        model = NetSequentialMultiInputs()
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(10, 10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        y = torch.randn(10, 10)
        y = y.to(device)
        y.detach_()
        model.to(device)
        y.requires_grad = True
        y_nosmp = y.detach().clone()
        x_nosmp.requires_grad = True
        y_nosmp.requires_grad = True
        output = model((x_nosmp, y_nosmp))
        out_grads = [torch.ones(10, 10), torch.ones(10, 10)]
        out_grads = [out_grad.to(device) for out_grad in out_grads]
        torch.autograd.backward(output, out_grads)
        nosmp_grads = model.net1.linear1.weight.grad

        cfg = {
            "microbatches": 2,
            "placement_strategy": "cluster",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)
        state.module_manager.assign_partition(model.net1, 1)
        state.module_manager.assign_partition(model.net1.linear1, 1)
        state.module_manager.assign_partition(model.net1.linear2, 1)
        state.module_manager.assign_partition(model.net2, 1)
        state.module_manager.assign_partition(model.net2.linear1, 1)
        state.module_manager.assign_partition(model.net2.linear2, 1)
        state.module_manager.assign_partition(model.sequential, 0)
        model = smp.DistributedModel(model, average_grads_across_microbatches=False)
        out_grads = [torch.ones(10, 10), torch.ones(10, 10)]
        out_grads = [out_grad.to(device) for out_grad in out_grads]
        output_dist = train_step(model, x, y, out_grads)
        dist_result = (output_dist[0].concat(), output_dist[1].concat())
        if smp.pp_rank() == 0:
            for i in range(2):
                np.testing.assert_allclose(
                    output[i].detach().cpu(), dist_result[i].detach().cpu(), atol=1e-6
                )

        if smp.pp_rank() == 1:
            np.testing.assert_allclose(
                nosmp_grads.cpu(), model.module.net1.linear1.weight.grad.cpu(), atol=1e-6
            )


class TestSMPSequentialInputAggregation(unittest.TestCase):
    """Tests a model which passes input to SMPSequential on
    another rank, and smp sequential output goes through
    multiple paths.
    """

    def test_kwargs(self):
        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class NetNonSMPSequential(torch.nn.Module):
            def __init__(self):
                super(NetNonSMPSequential, self).__init__()
                self.linear1 = nn.Linear(10, 10)
                self.sequential = torch.nn.Sequential(nn.Linear(10, 20))
                self.linear2 = nn.Linear(20, 20)
                self.linear3 = nn.Linear(20, 20)

            def forward(self, x):
                out1 = self.linear1(x)
                out2 = self.sequential(out1)
                out3 = self.linear2(out2)
                out4 = self.linear3(out2)
                result = out3 + out4
                return result

        cfg = {
            "microbatches": 1,
            "placement_strategy": "spread",
            "pipeline": "simple",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)

        torch.manual_seed(42)
        model = NetNonSMPSequential()
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")
        x = torch.randn(10)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        x_nosmp.requires_grad = True
        model.to(device)
        output = model(x_nosmp)
        out_grads = torch.randn(output.size(), device=torch.device("cuda"))
        torch.autograd.backward(output, out_grads)
        state.module_manager.assign_partition(model.sequential, 1)
        state.module_manager.assign_partition(model.sequential[0], 1)
        state.module_manager.assign_partition(model.linear2, 1)
        model = smp.DistributedModel(model)
        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, x, out_grads)
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), output_dist.outputs[0].detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())

        smp.barrier()
        smp.reset()
        model = NetNonSMPSequential()
        model.to(device)
        x_nosmp = x_nosmp.to(device)
        x = x.to(device)
        x.grad = None
        x_nosmp.grad = None
        output = model(x_nosmp)
        out_grads = torch.randn(output.size(), device=torch.device("cuda"))
        torch.autograd.backward(output, out_grads)
        smp.init(cfg)
        state.module_manager.assign_partition(model.sequential, 1)
        state.module_manager.assign_partition(model.sequential[0], 0)
        state.module_manager.assign_partition(model.linear2, 1)
        model = smp.DistributedModel(model)
        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, x, out_grads)
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), output_dist.outputs[0].detach().cpu())

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())


class TestSMPInputAggregation(unittest.TestCase):
    """Test a model which passes single input to child module
    and is used by children of the child modules"""

    def test_kwargs(self):
        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class Net2(torch.nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.linear1 = nn.Linear(20, 10)
                self.linear2 = nn.Linear(20, 10)

            def forward(self, x):
                z1 = self.linear1(x)
                z2 = self.linear2(x)
                return z1, z2

        class NetMultiInOuts(torch.nn.Module):
            def __init__(self):
                super(NetMultiInOuts, self).__init__()
                self.child = Net2()

            def forward(self, x):
                with smp.partition(1):
                    x = self.child(x)
                return x

        cfg = {
            "microbatches": 1,
            "placement_strategy": "spread",
            "pipeline": "simple",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)

        torch.manual_seed(42)
        model = NetMultiInOuts()
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")
        x = torch.randn(20)
        y = torch.randn(10)
        x = x.to(device)
        y = y.to(device)
        x.detach_()
        y.detach_()
        x.requires_grad = True
        y.requires_grad = True
        x_nosmp = x.detach().clone()
        y_nosmp = y.detach().clone()
        x_nosmp.requires_grad = True
        y_nosmp.requires_grad = True
        model.to(device)

        def assign_recursive(model, partition):
            for child in model.children():
                state.module_manager.assign_partition(child, partition)
                assign_recursive(child, partition)

        # Single GPU model
        torch.manual_seed(42)
        # Distributed model
        model2 = NetMultiInOuts()
        model2.to(device)
        output = model2(x_nosmp)
        out_grads = (
            torch.randn(output[0].size(), device=torch.device("cuda")),
            torch.randn(output[1].size(), device=torch.device("cuda")),
        )
        torch.autograd.backward(output, out_grads)
        assign_recursive(model, 1)
        state.module_manager.assign_partition(model.child.linear1, 0)
        model = smp.DistributedModel(model)

        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, x, out_grads)
        for idx, _ in enumerate(output):
            if smp.pp_rank() == 0:
                np.testing.assert_allclose(
                    output[idx].detach().cpu(), output_dist[idx].outputs[0].detach().cpu()
                )
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())


class TestMultiInputsOutputs(unittest.TestCase):
    """Test model with multiple inputs and outputs
    """

    def test_kwargs(self):
        @smp.step
        def train_step(model, data1, data2, out_grads):
            output = model(data1, data2)
            model.backward(output, out_grads)
            return output

        class Net2(torch.nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.linear1 = nn.Linear(20, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x, y):
                z1 = self.linear1(x)
                z2 = self.linear2(y)
                return z1, z2

        class NetMultiInOuts(torch.nn.Module):
            def __init__(self):
                super(NetMultiInOuts, self).__init__()
                self.child = Net2()

            def forward(self, x, y):
                with smp.partition(1):
                    x = self.child(x, y)
                return x

        torch.manual_seed(42)
        cfg = {
            "microbatches": 1,
            "placement_strategy": "spread",
            "pipeline": "simple",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)

        model = NetMultiInOuts()
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")
        x = torch.randn(20)
        y = torch.randn(10)
        x = x.to(device)
        y = y.to(device)
        x.detach_()
        y.detach_()
        x.requires_grad = True
        y.requires_grad = True
        x_nosmp = x.detach().clone()
        y_nosmp = y.detach().clone()
        x_nosmp.requires_grad = True
        y_nosmp.requires_grad = True
        model.to(device)

        def assign_recursive(model, partition):
            for child in model.children():
                state.module_manager.assign_partition(child, partition)
                assign_recursive(child, partition)

        # Single GPU model
        torch.manual_seed(42)

        # Distributed model

        model2 = NetMultiInOuts()
        # will be ignored
        model2.to(device)
        output = model2(x_nosmp, y_nosmp)
        out_grads = (
            torch.randn(output[0].size(), device=torch.device("cuda")),
            torch.randn(output[1].size(), device=torch.device("cuda")),
        )
        torch.autograd.backward(output, out_grads)

        assign_recursive(model, 1)
        state.module_manager.assign_partition(model.child.linear1, 0)
        model = smp.DistributedModel(model, trace_device="cpu")

        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, x, y, out_grads)

        for idx, _ in enumerate(output):
            if smp.pp_rank() == 0:
                np.testing.assert_allclose(
                    output[idx].detach().cpu(), output_dist[idx].outputs[0].detach().cpu()
                )
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())
            np.testing.assert_allclose(y_nosmp.grad.cpu(), y.grad.cpu())


class TestModuleList(unittest.TestCase):
    """Test model with ModuleList
    """

    def test_module_list(self):
        @smp.step
        def train_step(model, data1, out_grads):
            output = model(data1)
            model.backward(output, out_grads)
            return output

        class GrandChildModuleList(torch.nn.Module):
            def __init__(self):
                super(GrandChildModuleList, self).__init__()
                with smp.partition(1):
                    self.nnlist = nn.ModuleList([nn.Linear(20, 20), nn.Linear(20, 20)])

            def forward(self, x):
                z1 = self.nnlist[0](x)
                z2 = self.nnlist[1](z1)
                return z2

        class IntermediateModule(torch.nn.Module):
            def __init__(self):
                super(IntermediateModule, self).__init__()
                self.grandchild = GrandChildModuleList()

            def forward(self, x):
                return self.grandchild(x)

        class NetModuleList(torch.nn.Module):
            def __init__(self):
                super(NetModuleList, self).__init__()
                with smp.partition(0):
                    self.child_module_list = IntermediateModule()

            def forward(self, x):
                z1 = self.child_module_list(x)
                return z1

        torch.manual_seed(42)
        smp.reset()
        torch.cuda.set_device(smp.local_rank())
        cfg = {
            "microbatches": 1,
            "placement_strategy": "spread",
            "pipeline": "simple",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 1,
        }

        smp.init(cfg)
        device = torch.device("cuda")
        model = NetModuleList()
        model.to(device)
        x = torch.ones(20, requires_grad=True)
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        x_nosmp.requires_grad = True
        out_grads = (torch.ones(20, device=torch.device("cuda")),)
        output = model(x_nosmp)
        torch.autograd.backward(output, out_grads)

        device = torch.device("cuda")
        state.module_manager.assign_partition(model, 0)
        state.module_manager.assign_partition(model.child_module_list, 0)
        state.module_manager.assign_partition(model.child_module_list.grandchild, 1)
        model = smp.DistributedModel(model)
        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, x, out_grads)
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), output_dist[0].detach().cpu())
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu())


class TestSMPKwargs(unittest.TestCase):
    """Test model with kwargs
    """

    def test_kwargs(self):
        cfg = {
            "microbatches": 1,
            "placement_strategy": "spread",
            "pipeline": "simple",
            "optimize": "speed",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
        }
        smp.init(cfg)

        @smp.step
        def train_step(model, data, mask, out_grads):
            output = model(encoder_input=data, src_key_padding_mask=mask)
            model.backward(output, out_grads)
            return output

        torch.manual_seed(42)
        model = NetTransformer()
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")
        encoder_input_dist = torch.Tensor([[[20, 30, 40, 50]]])
        encoder_input_dist = encoder_input_dist.to(device)
        encoder_input_dist.requires_grad = True
        encoder_input = encoder_input_dist.detach().clone()
        encoder_input.requires_grad = True

        mask = torch.Tensor([[0]]) == 1
        mask = mask.to(device)
        model.to(device)

        def assign_recursive(model, partition):
            for child in model.children():
                state.module_manager.assign_partition(child, partition)
                assign_recursive(child, partition)

        # Single GPU model
        torch.manual_seed(42)
        # Distributed model

        model2 = NetTransformer()
        model2.to(device)
        output = model2(encoder_input, src_key_padding_mask=mask)
        out_grads = torch.randn(output.size(), device=torch.device("cuda"))
        output.backward(out_grads)

        assign_recursive(model, 1)
        model = smp.DistributedModel(model)

        torch.cuda.set_device(smp.local_rank())
        output_dist = train_step(model, encoder_input_dist, mask, out_grads)

        if smp.pp_rank() == 0:
            np.testing.assert_allclose(output.detach().cpu(), output_dist.outputs[0].detach().cpu())
            np.testing.assert_allclose(encoder_input.grad.cpu(), encoder_input_dist.grad.cpu())


class TestAutocast(unittest.TestCase):
    def test_amp_resume(self):
        @smp.step
        def step(model, scaler, data, target):
            with torch.cuda.amp.autocast(True):
                output = model(data)

            loss = F.nll_loss(output, target, reduction="mean")
            scaled_loss = scaler.scale(loss)
            model.backward(scaled_loss)
            return output, loss

        class Wrapper(nn.Module):
            def __init__(self):
                super(Wrapper, self).__init__()
                self.lin = nn.Linear(28, 28)

            def forward(self, x):
                assert not torch.is_autocast_enabled()
                return self.lin(x)

        class Net1(nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                self.lin1 = nn.Linear(28, 28)
                with smp.partition(0):
                    self.lin2 = Wrapper()

            def forward(self, x):
                assert not torch.is_autocast_enabled()
                x = self.lin1(x)
                x = self.lin2(x)
                return x

        class Net2(nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.fc1 = nn.Linear(28, 28)
                self.fc2 = nn.Linear(28, 10)

            def forward(self, x):
                assert torch.is_autocast_enabled()
                x = self.fc1(x)
                x = self.fc2(x)
                output = F.log_softmax(x, 1)
                return output

        class GroupedNet(nn.Module):
            def __init__(self):
                super(GroupedNet, self).__init__()
                with smp.partition(1):
                    self.net1 = Net1()
                self.net2 = Net2()

            def forward(self, x):
                assert torch.is_autocast_enabled()
                with torch.cuda.amp.autocast(False):
                    assert not torch.is_autocast_enabled()
                    x = self.net1(x)
                assert torch.is_autocast_enabled()
                x = self.net2(x)
                x = torch.flatten(x, 1)
                return x

        smp.init(
            {"partitions": 2, "auto_partition": False, "microbatches": 4, "default_partition": 0}
        )
        model = smp.DistributedModel(GroupedNet())
        optimizer = torch.optim.Adadelta(model.parameters(), lr=2)
        x = torch.randn(64, 28, 28)
        y = torch.randint(0, 10, (64,))
        device = torch.device("cuda", smp.local_rank())
        x = x.to(device)
        y = y.to(device)
        scaler = GradScaler()
        step(model, scaler, x, y)
        scaler.step(optimizer)
        scaler.update()


@smp.step
def basic_step(model, data, target):
    output = model(data)
    loss = F.nll_loss(output, target, reduction="mean")
    model.backward(loss)
    return output, loss


class TestHook(unittest.TestCase):
    @classmethod
    def create_hook_fn(cls, hook_called):
        def hook_fn(model, optimizer):
            assert len(set([x.device for x in model.parameters()])) > 1
            assert model.partitioned
            hook_called.add(True)

        return hook_fn

    class Model(nn.Module):
        def __init__(self):
            super(TestHook.Model, self).__init__()
            self.a = nn.Linear(4, 4)
            with smp.partition(1):
                self.b = nn.Linear(4, 4)
            self.c = nn.Linear(4, 4)

        def forward(self, x):
            x = self.a(x)
            x = F.relu(x)
            x = self.b(x)
            x = self.c(x)
            x = F.relu(x)
            return x

    def test_pp_hook(self):
        x = torch.randn(16, 4)
        y = torch.randint(low=0, high=3, size=(16,))
        smp.init(
            {"partitions": 2, "auto_partition": False, "microbatches": 4, "default_partition": 0}
        )
        model = smp.DistributedModel(TestHook.Model())
        optimizer = smp.DistributedOptimizer(torch.optim.Adadelta(model.parameters(), lr=2))
        hook_called = set()
        handle = model.register_post_partition_hook(TestHook.create_hook_fn(hook_called))
        device = torch.device("cuda", smp.local_rank())
        x = x.to(device)
        y = y.to(device)
        basic_step(model, x, y)
        assert len(hook_called) == 1

    def test_pp_hook_remove(self):
        x = torch.randn(16, 4)
        y = torch.randint(low=0, high=3, size=(16,))
        smp.init(
            {"partitions": 2, "auto_partition": False, "microbatches": 4, "default_partition": 0}
        )
        model = smp.DistributedModel(TestHook.Model())
        optimizer = smp.DistributedOptimizer(torch.optim.Adadelta(model.parameters(), lr=2))
        hook_called = set()
        handle = model.register_post_partition_hook(TestHook.create_hook_fn(hook_called))
        handle.remove()
        device = torch.device("cuda", smp.local_rank())
        x = x.to(device)
        y = y.to(device)
        basic_step(model, x, y)
        assert len(hook_called) == 0


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
