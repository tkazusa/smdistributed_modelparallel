# Future
from __future__ import absolute_import, division, print_function, unicode_literals

# Standard Library
import unittest

# Third Party
import numpy as np
import torch
import torch.distributed as dist
from torch import nn

# First Party
import smdistributed.modelparallel.test.torch.mpi_hybrid.default_ddp_comm_hooks as default_hooks
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch import DistributedModel
from smdistributed.modelparallel.torch.optimizers.optimizer import DistributedOptimizer


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()
        torch.manual_seed(0)
        self.p = nn.Parameter(torch.randn(20, 20))

    def forward(self, x):
        return self.p * x


class TestDdpCommHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.t0 = Task()
        self.t1 = Task()

    def forward(self, x, rank):
        return self.t0(x ** (1 + rank)) + self.t1(10 * x ** (1 + rank))


@smp.step
def step(model, input, rank):
    output = model(input, rank)
    model.backward(output.mean())


@smp.step
def join_step(model, input):
    output = model(input)
    model.backward(output.sum())


class DistributedDataParallelCommHookTest(unittest.TestCase):
    def _run_and_get_grads(self, model):
        torch.manual_seed(2020)
        input = torch.randn(40, 20)
        # Run forward
        step(model, input, smp.dp_rank())
        grads = [p.grad.data.cpu().numpy() for p in model.local_parameters()]
        for p in model.local_parameters():
            p.grad.data.zero_()
        return grads

    def test_ddp_comm_hook_allreduce_hook(self):
        """
        This unit test verifies the ``allreduce`` hook registered case gives same result
        with no hook registered case.
        """
        smp.init({"partitions": 2, "ddp": True, "microbatches": 2})
        model = DistributedModel(TestDdpCommHook(), average_grads_across_microbatches=False)
        # No hook registered case, get the reference grads.
        reference_grads = self._run_and_get_grads(model)
        # Register hook case, get the hook grads.
        model.register_comm_hook(
            hook=default_hooks.allreduce_hook, state=smp.get_dp_process_group()
        )
        hook_grads = self._run_and_get_grads(model)

        np.testing.assert_allclose(hook_grads, reference_grads, rtol=1e-5, atol=0)
        smp.barrier()

    def test_ddp_comm_hook_fp16compress_hook(self):
        """
        This unit test verifies the ``fp16 compress`` hook registered case
        gives close result with no hook registered case.
        """
        smp.init({"partitions": 2, "ddp": True, "microbatches": 2})
        model = DistributedModel(TestDdpCommHook(), average_grads_across_microbatches=False)

        # No hook registered case, get the reference grads.
        reference_grads = self._run_and_get_grads(model)
        # Register hook case, get the hook grads.
        model.register_comm_hook(
            hook=default_hooks.fp16_compress_hook, state=smp.get_dp_process_group()
        )
        hook_grads = self._run_and_get_grads(model)

        # TODO: investigate why atol needed to be changed to 1e-3 from 1e-4 in a similar DDP test
        np.testing.assert_allclose(hook_grads, reference_grads, rtol=1e-5, atol=1e-3)
        smp.barrier()

    def test_ddp_comm_hook_allgather_hook(self):
        """
        This unit test verifies the ``fp16 compress`` hook registered case
        gives close result with no hook registered case.
        """
        smp.init({"partitions": 2, "ddp": True, "microbatches": 2})
        model = DistributedModel(TestDdpCommHook(), average_grads_across_microbatches=False)

        # No hook registered case, get the reference grads.
        reference_grads = self._run_and_get_grads(model)
        # Register hook case, get the hook grads.
        model.register_comm_hook(
            hook=default_hooks._allgather_then_aggregate_hook, state=smp.get_dp_process_group()
        )
        hook_grads = self._run_and_get_grads(model)

        np.testing.assert_allclose(hook_grads, reference_grads, rtol=1e-5, atol=0)
        smp.barrier()


class JoinTests(unittest.TestCase):
    class Net(nn.Module):
        def __init__(self, dim):
            super(JoinTests.Net, self).__init__()
            self.lin = nn.Linear(dim, dim, bias=False)
            self.lin2 = nn.Identity()

        def forward(self, x):
            return self.lin2(self.lin(x))

    class BiggerNet(nn.Module):
        def __init__(self, dim):
            super(JoinTests.BiggerNet, self).__init__()
            self.lin = nn.Linear(dim, dim, bias=False)
            self.lin2 = nn.Linear(dim, dim, bias=True)

        def forward(self, x):
            return self.lin2(self.lin(x))

    def setUp(self):
        self.microbatches = 2
        smp.init({"partitions": 2, "ddp": True, "microbatches": self.microbatches})
        self.rank = smp.local_rank()

    # This test also tests case where one partition has overlapping and other non overlapping reducers
    def test_ddp_grad_div_uneven_inputs(self, overlap=True):
        # Test gradient division during training with join() API. If
        # divide_by_initial_world_size=False, we scale by the effective world
        # size when allreducing grads.
        dim = 5
        batch = 1 * self.microbatches
        grad_scale = 50

        inp = torch.ones(batch, dim, device=self.rank) * grad_scale
        net = DistributedModel(
            JoinTests.Net(dim).cuda(self.rank), bucket_cap_mb=1, overlapping_allreduce=overlap
        )
        n_iters = 3
        if smp.dp_rank() > 0:
            n_iters += 2

        with net.join(divide_by_initial_world_size=False):
            for _ in range(n_iters):
                join_step(net, inp)
                # The grad is always expected_grad, since we divide by the number
                # of currently active processes and inactive processes contribute
                # zero gradient. If we kept dividing by static initial world
                # size as processes leave, the grad would be smaller.
                params = list(net.local_parameters())
                if len(params):
                    expected_grad = torch.ones(dim, dim, device=self.rank) * grad_scale
                    param = params[0]
                    np.testing.assert_allclose(
                        expected_grad.cpu().numpy(), param.grad.data.cpu().numpy()
                    )
                # Avoid accumulating grads so that it's the same every iteration
                net.zero_grad()

                torch.cuda.synchronize(device=self.rank)

        # If divide_by_initial_world_size=True (default), we always scale grads
        # by the initial world_size.
        with net.join(divide_by_initial_world_size=True):
            for i in range(n_iters):
                join_step(net, inp)
                effective_ws = dist.get_world_size(smp.get_dp_process_group())
                if i >= 3:
                    effective_ws -= 1
                expected_grad = (
                    torch.ones(dim, dim, device=self.rank) * grad_scale * effective_ws
                ) / dist.get_world_size(smp.get_dp_process_group())
                params = list(net.local_parameters())
                if len(params):
                    param = params[0]
                    np.testing.assert_allclose(
                        expected_grad.cpu().numpy(), param.grad.data.cpu().numpy()
                    )
                # Avoid accumulating grad so that it's the same every iteration.
                net.zero_grad()
                torch.cuda.synchronize(device=self.rank)

        smp.barrier()

    def test_ddp_grad_div_uneven_inputs_nooverlap(self):
        self.test_ddp_grad_div_uneven_inputs(overlap=False)
        smp.barrier()

    def test_ddp_join_model_equivalence(self):
        batch = 2 * self.microbatches
        dim = 10
        learning_rate = 0.03
        model = JoinTests.BiggerNet(dim)
        inp = torch.rand(batch, dim, device=self.rank)

        net = DistributedModel(model.cuda(self.rank))
        ddp_optim = torch.optim.SGD(
            model.parameters(), lr=learning_rate * dist.get_world_size(smp.get_dp_process_group())
        )
        ddp_optim = DistributedOptimizer(ddp_optim)

        with net.join():
            for i in range(2 * (smp.dp_rank() + 1)):
                ddp_optim.zero_grad()
                join_step(net, inp)
                torch.cuda.synchronize(device=self.rank)
                ddp_optim.step()

        local_states = smp.allgather(net.local_state_dict(), smp.CommGroup.DP_GROUP)
        if smp.dp_rank() > 0:
            for (_, t1), (_, t2) in zip(
                local_states[smp.dp_rank()].items(), local_states[0].items()
            ):
                if isinstance(t1, (bool, dict)):
                    assert isinstance(t2, (bool, dict))
                    continue
                np.testing.assert_allclose(t1.numpy(), t2.numpy())
        smp.barrier()


if __name__ == "__main__":
    unittest.main()
