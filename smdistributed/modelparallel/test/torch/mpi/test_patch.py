# Standard Library
import unittest

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import (
    Net1,
    mock_is_executor,
    mock_recv_forward,
    mock_send_backward,
)
from smdistributed.modelparallel.torch.graph_utils import get_ancestors
from smdistributed.modelparallel.torch.patches.tracing import TracingEnd


class Nest1(nn.Module):
    def __init__(self):
        super(Nest1, self).__init__()
        with smp.partition(1):
            self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Nest2(nn.Module):
    def __init__(self):
        super(Nest2, self).__init__()
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc3(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        with smp.partition(0):
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.nest1 = Nest1()

        with smp.partition(1):
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
            self.nest2 = Nest2()

    def forward(self, x, target):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.nest1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.nest2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TestDevices(unittest.TestCase):
    def check_devices_internal(self, module):
        for name, param in module.named_parameters(recurse=False):
            # print(
            #     smp.rank(),
            #     module,
            #     name,
            #     param.shape,
            #     param.device,
            #     smp.state.module_manager.get_partition(module),
            # )
            if smp.state.module_manager.get_partition(module) == smp.pp_rank():
                assert param.device == torch.device("cuda", smp.local_rank()), (
                    smp.local_rank(),
                    name,
                    param.device,
                    param.shape,
                )
            else:
                assert param.device == torch.device("cpu"), (
                    smp.local_rank(),
                    name,
                    param.device,
                    param.shape,
                )

    def check_devices(self, module):
        for n, m in module.named_children():
            self.check_devices(m)
        self.check_devices_internal(module)

    def test_parameter_devices(self):
        smp.init(
            {
                "microbatches": 2,
                "pipeline": "simple",
                "partitions": 2,
                "auto_partition": False,
                "default_partition": 0,
            }
        )
        n = Net()
        n = smp.DistributedModel(n)
        n.post_partition()
        assert len(set([x.device for x in n.parameters()])) == 2
        n.to(torch.device("cuda"))
        assert len(set([x.device for x in n.parameters()])) == 2
        n.cpu()
        assert len(set([x.device for x in n.parameters()])) == 1
        n.cuda()
        assert len(set([x.device for x in n.parameters()])) == 2
        self.check_devices(n)

    def test_parameter_devices_autop(self):
        smp.init(
            {
                "microbatches": 2,
                "pipeline": "simple",
                "partitions": 2,
                "auto_partition": True,
                "default_partition": 0,
            }
        )

        class AutopNest1(nn.Module):
            def __init__(self):
                super(AutopNest1, self).__init__()
                self.conv2 = nn.Conv2d(32, 64, 3, 1)

            def forward(self, x):
                return x

        class AutopNet(nn.Module):
            def __init__(self):
                super(AutopNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.nest1 = AutopNest1()
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                return x

        n = AutopNet()
        n = smp.DistributedModel(n)
        n.to(torch.device("cuda"))
        assert not any([x.is_cuda for x in n.parameters()])
        n.cuda()
        assert not any([x.is_cuda for x in n.parameters()])
        n.cpu()
        assert len(set([x.device for x in n.parameters()])) == 1
        self.check_devices(n)


class TestManualAssignment(unittest.TestCase):
    def test_manual_assignment(self):
        smp.init(
            {
                "microbatches": 2,
                "pipeline": "simple",
                "partitions": 2,
                "auto_partition": False,
                "default_partition": 0,
            }
        )
        n = Net()
        n = smp.DistributedModel(n)
        state = smp.state

        assert state.module_manager.get_partition(n) == 0
        assert (
            state.module_manager.get_partition(n.module) == 0
        ), state.module_manager._module_partitions
        assert state.module_manager.get_partition(n.module.conv1) == 0
        assert state.module_manager.get_partition(n.module.nest1) == 0
        assert state.module_manager.get_partition(n.module.nest1.conv2) == 1
        assert state.module_manager.get_partition(n.module.fc1) == 1
        assert state.module_manager.get_partition(n.module.fc2) == 1
        assert state.module_manager.get_partition(n.module.nest2) == 1
        assert state.module_manager.get_partition(n.module.nest2.fc3) == 1

        n.post_partition()
        if smp.pp_rank() == 0:
            assert len(list(n.local_modules())) == 3, list(n.local_modules())
        if smp.pp_rank() == 1:
            assert len(list(n.local_modules())) == 5, list(n.local_modules())


class TestTracePatch(unittest.TestCase):
    def test_trace_patch(self):
        smp.init(
            {
                "microbatches": 2,
                "pipeline": "simple",
                "partitions": 2,
                "auto_partition": False,
                "default_partition": 0,
            }
        )

        class SeqModel(nn.Module):
            def __init__(self, N, D_in, H, D_out):
                super(SeqModel, self).__init__()
                with smp.partition(1):
                    self.lin = nn.Linear(D_in, H)
                    with smp.partition(0):
                        self.relu = nn.ReLU()
                with smp.partition(0):
                    self.lin2 = nn.Linear(H, D_out)

            def forward(self, x, cool=False):
                x = self.lin(x)
                x = self.relu(x)
                x = self.lin2(x)
                return x

        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in, device=torch.device("cuda", smp.local_rank()))
        y = torch.randn(N, D_out, device=torch.device("cuda", smp.local_rank()))
        model = SeqModel(N, D_in, H, D_out)
        model = smp.DistributedModel(model)
        smp.state.current_step_func = lambda: 1
        for t in range(2):
            if t == 0:
                with smp.state.patch_manager.patch_for_trace(
                    model, torch.device("cuda", smp.local_rank())
                ):
                    with torch.no_grad():
                        with self.assertRaises(TracingEnd):
                            y_pred = model(x, cool=True)
        mm = smp.state.module_manager
        mm.name_modules_and_create_parent_map()

        names = mm._module_to_name.values()
        assert len(set(names)) == len(names)
        assert "main" in names
        assert "main/module/lin" in names
        assert mm._module_execution_order.index(model) == 0
        assert mm._module_execution_order.index(model.module.lin2) == 4

        for m in model.modules():
            assert isinstance(mm._traced_output_sizes[m], int)
            assert m in mm._traced_output_sizes
            assert m in mm._module_to_name

    def test_trace_cpu(self):
        smp.init({"microbatches": 2, "pipeline": "simple", "partitions": 2})

        class SeqModel(nn.Module):
            def __init__(self, N, D_in, H, D_out):
                super(SeqModel, self).__init__()
                self.lin = nn.Linear(D_in, H)
                self.relu = nn.ReLU()
                self.lin2 = nn.Linear(H, D_out)

            def forward(self, x, cool=False):
                x = self.lin(x)
                x = self.relu(x)
                x = self.lin2(x)
                u = x.mean(-1, keepdim=True)
                x = (x - u).pow(2).mean(-1, keepdim=True)
                return x

        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in, dtype=torch.float16)
        y = torch.randn(N, D_out, dtype=torch.float16)
        model = SeqModel(N, D_in, H, D_out)
        model.half()
        model = smp.DistributedModel(model, trace_device="cpu")
        smp.state.current_step_func = lambda: 1
        for t in range(2):
            if t == 0:
                with smp.state.patch_manager.patch_for_trace(model, torch.device("cpu")):
                    with torch.no_grad():
                        with self.assertRaises(TracingEnd):
                            y_pred = model(x, cool=True)
        mm = smp.state.module_manager
        mm.name_modules_and_create_parent_map()

        names = mm._module_to_name.values()
        assert len(set(names)) == len(names)
        assert "main" in names
        assert "main/module/lin" in names
        assert mm._module_execution_order.index(model) == 0
        assert mm._module_execution_order.index(model.module.lin2) == 4

        for m in model.modules():
            assert isinstance(mm._traced_output_sizes[m], int)
            assert m in mm._traced_output_sizes
            assert m in mm._module_to_name


class TestDistributedForwardBackward(unittest.TestCase):
    """Tests distributed_forward with the SMPInput,
    and SMPParentSend, SMPParentRecv ops"""

    @unittest.skip(
        "Test was written when many aspects of SMP PT weren't there, now many guarantees that those components expect aren't being satisfied, couldn't make the test work. I think what it verifies is covered in other tests."
    )
    def test_model_net1(self):
        # Model with two children for testing
        cfg = {
            "microbatches": 2,
            "placement_strategy": "spread",
            "partitions": 2,
            "pipeline": "simple",
            "optimize": "speed",
        }
        smp.init(cfg)
        sg1 = Net1()
        child1 = nn.Linear(1000, 40)
        child2 = nn.Linear(40, 20)

        gpu_device = torch.device("cuda", smp.local_rank())

        sg1.to(gpu_device)
        child1.to(gpu_device)
        child2.to(gpu_device)

        # Prepare input data
        x = torch.rand(40, 1000, requires_grad=True, device=gpu_device)

        # Run childs modules forward to be saved and used for mock later
        # detaching here, since we dont want the outputs to be part of a graph
        y = x.detach().clone()
        y.detach_()
        y.requires_grad = True
        out_child1 = child1(y)
        in_child2 = out_child1.clone()
        out_child2 = child2(in_child2)
        smp.patches.execution.state.model = sg1
        models = {sg1: (True, True), sg1.linear1: (False, True), sg1.linear2: (False, True)}
        module_to_outputs = {"main/linear1": out_child1, "main/linear2": out_child2}
        mock_execs = mock_is_executor(models)
        mock_forward = mock_recv_forward(module_to_outputs)
        mock_backward = mock_send_backward()

        count = 0
        with mock_execs:
            smp.patches.execution.state.patch_manager.patch_forward(model=sg1)
            # Forward pass
            with mock_forward:
                y = sg1.forward(x)
            _, _, recvs = get_ancestors(y)
            for r in recvs:
                r.next_parent_recvs_bwd_pending += 1
            # Backward pass
            with mock_backward:
                count = 0

                y.backward(torch.ones(y.size(), device=gpu_device), retain_graph=True)
                output1 = smp.patches.execution.state.module_manager.get_output(
                    0, sg1.linear2, sg1, 0
                )
                output1[0].backward(torch.ones(output1[0].size(), device=gpu_device))
                output2 = smp.patches.execution.state.module_manager.get_output(
                    0, sg1.linear1, sg1, 0
                )
                output2[0].backward(torch.ones(output2[0].size(), device=gpu_device))


class TestSequential(unittest.TestCase):
    def test_torch_sequential(self):
        @smp.step
        def train_step(model, x):
            output = model(x)
            loss = torch.sum(output)
            model.backward(loss)

        class Model(nn.Module):
            def __init__(self, N, D_in, H, D_out):
                super(Model, self).__init__()
                self.lin = nn.Linear(D_in, H)
                self.seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(H, H),
                    nn.Linear(H, H),
                    nn.Linear(H, H),
                    nn.Linear(H, H),
                    nn.Linear(H, H),
                    nn.Linear(H, H),
                    nn.Linear(H, H),
                    nn.Linear(H, H),
                    nn.ReLU(),
                )
                self.lin2 = nn.Linear(H, D_out)

            def forward(self, x):
                x = self.lin(x)
                x = self.seq(x)
                x = self.lin2(x)
                return x

        cfg = {"microbatches": 2, "partitions": 2, "pipeline": "simple"}
        smp.init(cfg)
        N, D_in, H, D_out = 64, 1000, 100, 10
        model = Model(N, D_in, H, D_out)
        model = smp.DistributedModel(model)
        x = torch.randn(N, D_in, device=torch.device("cuda", smp.local_rank()))
        train_step(model, x)


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
