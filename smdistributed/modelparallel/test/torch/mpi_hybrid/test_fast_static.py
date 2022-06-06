# Standard Library
import unittest

# Third Party
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# First Party
import smdistributed.modelparallel.torch as smp


class TestFastStaticBase(unittest.TestCase):
    def test_model(self):
        model_cls = self.get_model_class()
        mb = 8
        ambs = [mb]
        for pipeline in ["simple", "interleaved"]:
            if pipeline == "interleaved":
                ambs.append(mb // 2)
            for amb in ambs:
                for static, fast in [(True, True), (True, False), (False, True)]:
                    self.verify_configuration(model_cls, pipeline, fast, static, mb, amb)

    def verify_configuration(self, model_cls, pipeline, fast, static, mb, active_mb):
        @smp.step
        def train_step(model, data):
            output = model(data)
            model.backward(output)
            return output

        cfg = {
            "microbatches": mb,
            "active_microbatches": active_mb,
            "placement_strategy": "spread",
            "pipeline": pipeline,
            "optimize": "speed",
            "partitions": 4,
            "auto_partition": False,
            "static_mode": static,
            "fast_mode": fast,
            "default_partition": 0,
            "ddp": True,
        }
        smp.init(cfg)

        # prepare inputs
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        x = torch.randn(20 * cfg["microbatches"])
        x = x.to(device)
        x.detach_()
        x.requires_grad = True
        x_nosmp = x.detach().clone()
        x_nosmp.requires_grad = True

        # non-smp version
        torch.manual_seed(42)
        model2 = model_cls()
        model2.to(device)
        mb_outputs = []
        for mb in range(cfg["microbatches"]):
            mb_output = model2(x_nosmp[20 * mb : 20 * (mb + 1)])
            torch.autograd.backward(mb_output)
            mb_outputs.append(mb_output)
        output = sum(mb_outputs) / len(mb_outputs)

        # smp version
        torch.manual_seed(42)
        model = model_cls()
        model.to(device)
        model = smp.DistributedModel(model)
        # record the sequence at the second step
        smp.state.exec_server.server_queue.record_step = 1
        torch.cuda.set_device(smp.local_rank())

        # run 3 times so that the metadata tranmission is skipped for 3rd time
        output_dist = train_step(model, x)
        output_dist = train_step(model, x)
        output_dist = train_step(model, x)

        # check output and grads
        if smp.pp_rank() == 0:
            np.testing.assert_allclose(
                output.detach().cpu(), output_dist.reduce_mean().detach().cpu(), atol=1e-3
            )
        if smp.pp_rank() == 0:
            # divide by 2 to account for 2 runs
            np.testing.assert_allclose(x_nosmp.grad.cpu(), x.grad.cpu() / 3.0, atol=1e-3)


class TestNoDirect(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(0):
                    self.linear1 = nn.Linear(20, 10)
                with smp.partition(1):
                    self.linear2 = nn.Linear(20, 10)

            def forward(self, x):
                z1 = self.linear1(x)
                z2 = self.linear2(x)
                return torch.sum(z1 + z2)

        return Net


class TestSimpleColocatedChain(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(0):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(1):
                    self.linear2 = nn.Linear(20, 20)
                    self.linear3 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return torch.sum(x)

        return Net


class TestLongColocatedChain(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(0):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(1):
                    self.linear2 = nn.Linear(20, 20)
                    self.linear3 = nn.Linear(20, 20)
                    self.linear4 = nn.Linear(20, 20)
                    self.linear5 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = self.linear4(x)
                x = self.linear5(x)
                return torch.sum(x)

        return Net


class TestLongMultihopChain(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(0):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(1):
                    self.linear2 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear3 = nn.Linear(20, 20)
                with smp.partition(3):
                    self.linear4 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear5 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = self.linear4(x)
                x = self.linear5(x)
                return torch.sum(x)

        return Net


class TestMultiRemoteConsumer(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(0):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(1):
                    self.linear2 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear3 = nn.Linear(20, 20)
                with smp.partition(3):
                    self.linear4 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear5 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                y = self.linear2(x)
                z = self.linear3(x)
                t = self.linear4(x)
                w = self.linear5(t)
                return torch.sum(y + z + w)

        return Net


class TestMixedLocalRemoteConsumer(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(0):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(1):
                    self.linear2 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear3 = nn.Linear(20, 20)
                with smp.partition(3):
                    self.linear4 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear5 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                y = self.linear4(x)
                z = self.linear5(x)
                return torch.sum(y + z)

        return Net


class TestMixedLocalRemoteParentConsumer(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(1):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear2 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear3 = nn.Linear(20, 20)
                with smp.partition(0):
                    self.linear4 = nn.Linear(20, 20)
                with smp.partition(3):
                    self.linear5 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                y = self.linear3(x)
                z = self.linear4(x)
                t = self.linear5(x)
                return torch.sum(y + z + t)

        return Net


class TestModuleReuse(TestFastStaticBase):
    def get_model_class(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(1):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear2 = nn.Linear(20, 20)
                with smp.partition(3):
                    self.linear3 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear2(x)
                x = self.linear2(x)
                x = self.linear1(x)
                x = self.linear1(x)
                return torch.sum(x)

        return Net


class TestNestedModuleReuse(TestFastStaticBase):
    def get_model_class(self):
        class Net1(nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                self.linear1 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear2 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear2(x)
                return x

        class Net2(nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.linear1 = nn.Linear(20, 20)
                with smp.partition(3):
                    self.linear2 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return torch.sum(x)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.net1 = Net1()
                with smp.partition(1):
                    self.net2 = Net2()

            def forward(self, x):
                x = self.net1(x)
                x = self.net2(x)
                return x

        return Net


class TestNestedWithReuseAndFunctionals(TestFastStaticBase):
    def get_model_class(self):
        class Net1(nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                self.linear1 = nn.Linear(20, 20)
                with smp.partition(2):
                    self.linear2 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
                x = self.linear2(x)
                return x

        class Net2(nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                self.linear1 = nn.Linear(20, 20)
                with smp.partition(3):
                    self.linear2 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
                return torch.sum(x)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.net1 = Net1()
                with smp.partition(1):
                    self.net2 = Net2()

            def forward(self, x):
                x = self.net1(x)
                x = self.net2(x)
                return x

        return Net


class TestSequential(TestFastStaticBase):
    def get_model_class(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                with smp.partition(0):
                    self.linear1 = nn.Linear(20, 20)
                with smp.partition(1):
                    self.linear2 = nn.Linear(20, 20)
                    self.linear3 = nn.Linear(20, 20)
                    self.seq = nn.Sequential(self.linear2, self.linear3)
                with smp.partition(2):
                    self.linear4 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.linear1(x)
                x = self.seq(x)
                x = self.linear4(x)
                return torch.sum(x)

        return Net


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
