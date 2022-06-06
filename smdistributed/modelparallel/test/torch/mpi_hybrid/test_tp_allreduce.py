# Standard Library
import unittest

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import ATOL, equalize_linear_weights
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.tp_registry import get_weight_slice


class TestAllreduceTensorParallelism(unittest.TestCase):
    def test_tp_allreduce(
        self, overlap=True, activation_checkpointing=True, offload_activations=False
    ):
        batch_size = 64

        smp.init(
            {
                "pipeline_parallel_degree": 2,
                "microbatches": 1,
                "tensor_parallel_degree": 2,
                "ddp": True,
                "auto_partition": False,
                "default_partition": 0,
                "offload_activations": offload_activations,
            }
        )

        torch.manual_seed(42)
        torch.cuda.set_device(smp.local_rank())

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                with smp.partition(1):
                    self.lin0 = nn.Linear(20, 20)
                    self.lin1 = nn.Linear(20, 20)
                with smp.tensor_parallelism():
                    self.lin2 = nn.Linear(20, 20)
                    self.lin3 = nn.Linear(20, 20)

            def forward(self, x):
                x = self.lin0(x)
                x = self.lin1(x)
                x = self.lin2(x)
                return torch.flatten(self.lin3(x), 1)

        model = Model()
        model_clone = Model()
        model_clone.load_state_dict(model.state_dict())

        x = torch.randn(batch_size, 20, 20)
        y = torch.randint(0, 10, (batch_size,))

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
        dist_model = smp.DistributedModel(model_clone, overlapping_allreduce=overlap)

        if activation_checkpointing:
            smp.set_activation_checkpointing(dist_model.module.module.lin1)
            smp.set_activation_checkpointing(dist_model.module.module.lin2)

        module_pairs = [
            (model.lin0, dist_model.module.module.lin0, None),
            (model.lin1, dist_model.module.module.lin1, None),
            (model.lin2, dist_model.module.module.lin2, "input"),
            (model.lin3, dist_model.module.module.lin3, "input"),
        ]

        for mod, dist_mod, ptn in module_pairs:
            equalize_linear_weights(mod, dist_mod, ptn)

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

        for mod, dist_mod, ptn in module_pairs:
            if smp.pp_rank() == state.module_manager.get_partition(dist_mod):
                for n, p in dist_mod.named_parameters(recurse=False):
                    if smp.tp_rank() != 0 and n == "bias":
                        continue

                    if n == "bias":
                        original_grad = original_grads[(mod, n)]
                    else:
                        original_grad = get_weight_slice(original_grads[(mod, n)], ptn)

                    self.assertTrue(torch.allclose(p.grad.cpu(), original_grad.cpu(), atol=ATOL))

    def test_tp_allreduce_overlap(self):
        self.test_tp_allreduce(overlap=True, activation_checkpointing=False)

    def test_tp_allreduce_nonoverlap(self):
        self.test_tp_allreduce(overlap=False, activation_checkpointing=False)

    def test_tp_allreduce_nonoverlap_activ_checkpoint(self):
        self.test_tp_allreduce(overlap=False, activation_checkpointing=True)

    def test_tp_allreduce_nonoverlap_activ_checkpoint_offload(self):
        self.test_tp_allreduce(
            overlap=False, activation_checkpointing=True, offload_activations=True
        )


if __name__ == "__main__":
    unittest.main()
