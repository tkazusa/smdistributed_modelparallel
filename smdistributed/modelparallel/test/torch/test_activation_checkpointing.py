# Standard Library
import unittest

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import ATOL
from smdistributed.modelparallel.torch.patches.checkpoint import checkpoint
from smdistributed.modelparallel.torch.state_mod import state


class TestActivationCheckpointing(unittest.TestCase):
    def test_activ_ckpt_no_kwargs(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.lin2(x)
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x = self.inner1(x)
                return torch.flatten(self.inner2(x), 1)

        self._test_act_ckpt(Model)

    def test_activ_sequential(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.lin2(x)
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()
                self.seq = nn.Sequential(self.lin0, self.lin1, self.inner1, self.inner2)

            def forward(self, x):
                x = self.seq(x)
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model)

    def test_activ_sequential_children(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.layer = nn.Linear(20, 20)
                self.num_executions = 0

            def forward(self, x):
                x = self.layer(x)
                self.num_executions += 1
                return x

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = nn.Sequential(InnerModule(), InnerModule())
                self.inner2 = nn.Sequential(InnerModule(), InnerModule())

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x = self.inner1(x)
                x = self.inner2(x)
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model, set_sequential_children=True)

    def test_activ_ckpt_no_kwargs_ckpt_api(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.lin2(x)
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                if state.model:
                    x = checkpoint(self.inner1, x)
                    x = checkpoint(self.inner2, x)
                else:
                    x = self.inner1(x)
                    x = self.inner2(x)
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model, set_activation_checkpointing=False)

    def test_activ_ckpt_args_kwargs_ckpt_api(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(self, *args, **kwargs):
                x = self.lin2(args[0])
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                if state.model:
                    x = checkpoint(self.inner1, x)
                    x = checkpoint(self.inner2, x)
                else:
                    x = self.inner1(x)
                    x = self.inner2(x)
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model, set_activation_checkpointing=False)

    def test_activ_ckpt_args_kwargs(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(self, *args, **kwargs):
                x = self.lin2(args[0])
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x = self.inner1(x)
                return torch.flatten(self.inner2(x), 1)

        self._test_act_ckpt(Model)

    def test_activ_ckpt_default_kwargs_unused(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(
                self,
                x,
                some_option=True,
                another_option=3,
                yet_another=None,
                tensor_option=torch.randn(2, 2),
            ):
                x = self.lin2(x)
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x = self.inner1(x)
                return torch.flatten(self.inner2(x), 1)

        self._test_act_ckpt(Model)

    def test_activ_ckpt_default_kwargs_used(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(
                self,
                x,
                some_option=True,
                another_option=3,
                yet_another=None,
                tensor_option=torch.ones(20, 20, dtype=torch.float32),
            ):
                if some_option:
                    x = self.lin2(x) + tensor_option.to(torch.device("cuda"))
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x = self.inner1(x)
                return torch.flatten(self.inner2(x), 1)

        self._test_act_ckpt(Model)

    def test_activ_ckpt_partially_passed_kwargs(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(
                self,
                x,
                some_option=True,
                another_option=3,
                yet_another=None,
                tensor_option=torch.ones(20, 20, dtype=torch.float32),
            ):
                x = self.lin2(x)
                if some_option:
                    x = torch.matmul(x, tensor_option.to(torch.device("cuda")))
                return self.lin3(x)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x = self.inner1(x, some_option=False)
                x = self.inner2(
                    x, some_option=True, tensor_option=0.1 * torch.ones(20, 20, dtype=torch.float32)
                )
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model)

    def test_activ_ckpt_partially_passed_kwargs_tuple_return(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(
                self,
                x,
                some_option=True,
                another_option=3,
                yet_another=None,
                tensor_option=torch.ones(20, 20, dtype=torch.float32),
            ):
                intermediate = self.lin2(x)
                if some_option:
                    x = torch.matmul(intermediate, tensor_option.to(torch.device("cuda")))
                return self.lin3(x), intermediate

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x, y = self.inner1(x, some_option=False)
                x, z = self.inner2(
                    x, some_option=True, tensor_option=0.1 * torch.ones(20, 20, dtype=torch.float32)
                )
                x = x + y + z
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model)

    def test_activ_ckpt_partially_passed_kwargs_tuple_return_structure_input(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(
                self,
                ls,
                some_option=True,
                another_option=3,
                yet_another=None,
                tensor_option=torch.ones(20, 20, dtype=torch.float32),
            ):
                x, y = ls
                intermediate = self.lin2(x) + self.lin2(y)
                if some_option:
                    x = torch.matmul(intermediate, tensor_option.to(torch.device("cuda")))
                return self.lin3(x), intermediate

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x, y = self.inner1([x, x], some_option=False)
                x, z = self.inner2(
                    [x, y],
                    some_option=True,
                    tensor_option=0.1 * torch.ones(20, 20, dtype=torch.float32),
                )
                x = x + y + z
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model)

    def test_activ_ckpt_partially_passed_kwargs_tuple_return_ckpt_api(self):
        class InnerModule(nn.Module):
            def __init__(self):
                super(InnerModule, self).__init__()
                self.lin2 = nn.Linear(20, 20)
                self.lin3 = nn.Linear(20, 20)

            def forward(
                self,
                x,
                some_option=True,
                another_option=3,
                yet_another=None,
                tensor_option=torch.ones(20, 20, dtype=torch.float32),
            ):
                intermediate = self.lin2(x)
                if some_option:
                    x = torch.matmul(intermediate, tensor_option.to(torch.device("cuda")))
                return self.lin3(x), intermediate

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.lin0 = nn.Linear(20, 20)
                self.lin1 = nn.Linear(20, 20)
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                if state.model:
                    x, y = checkpoint(self.inner1, x, some_option=False)
                    x, z = checkpoint(
                        self.inner2,
                        x,
                        some_option=True,
                        tensor_option=0.1 * torch.ones(20, 20, dtype=torch.float32),
                    )
                else:
                    x, y = self.inner1(x, some_option=False)
                    x, z = self.inner2(
                        x,
                        some_option=True,
                        tensor_option=0.1 * torch.ones(20, 20, dtype=torch.float32),
                    )
                x = x + y + z
                return torch.flatten(x, 1)

        self._test_act_ckpt(Model, set_activation_checkpointing=False)

    def _test_act_ckpt(self, model_class, set_activation_checkpointing=True, set_sequential_children=False):
        batch_size = 64

        smp.init({"pipeline_parallel_degree": 1, "tensor_parallel_degree": 1, "skip_tracing": True})

        torch.manual_seed(42)
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda", smp.local_rank())

        model = model_class().to(device)
        model_clone = model_class()
        model_clone.load_state_dict(model.state_dict())

        x = torch.randn(batch_size, 20, 20).to(device)
        y = torch.randint(0, 10, (batch_size,)).to(device)

        # get grads without smp
        out = model(x)
        loss = F.nll_loss(out, y, reduction="mean")
        loss.backward()

        original_grads = {}
        for m in model.modules():
            for n, p in m.named_parameters(recurse=False):
                original_grads[(m, n)] = p.grad.clone().detach()

        # zero grads
        for p in model.parameters():
            p.grad = None

        # run distributed
        dist_model = smp.DistributedModel(model_clone)

        if set_activation_checkpointing:
            if not set_sequential_children:
                smp.set_activation_checkpointing(dist_model.module.inner1)
                smp.set_activation_checkpointing(dist_model.module.inner2)
            else:
                for child in dist_model.module.inner1.children():
                    smp.set_activation_checkpointing(child)

        module_pairs = [
            (model.lin0, dist_model.module.lin0),
            (model.lin1, dist_model.module.lin1),
            (model.inner1, dist_model.module.inner1),
            (model.inner2, dist_model.module.inner2),
        ]

        @smp.step
        def train_step(x, y):
            out = dist_model(x)
            loss = F.nll_loss(out, y, reduction="mean")
            dist_model.backward(loss)

        device = torch.device("cuda", smp.local_rank())
        x = x.to(device)
        y = y.to(device)
        local_bs = batch_size // smp.dp_size()
        train_step(
            x[smp.dp_rank() * local_bs : (smp.dp_rank() + 1) * local_bs],
            y[smp.dp_rank() * local_bs : (smp.dp_rank() + 1) * local_bs],
        )

        for mod, dist_mod in module_pairs:
            for n, p in dist_mod.named_parameters(recurse=False):
                original_grad = original_grads[(mod, n)]

                self.assertTrue(torch.allclose(p.grad.cpu(), original_grad.cpu(), atol=ATOL))

        if set_sequential_children:
            for child in dist_model.module.inner1.children():
                self.assertEqual(child.num_executions, 2)
            for child in dist_model.module.inner2.children():
                self.assertEqual(child.num_executions, 1)


if __name__ == "__main__":
    unittest.main()
