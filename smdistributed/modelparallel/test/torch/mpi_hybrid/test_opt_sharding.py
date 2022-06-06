# Standard Library
import argparse

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Adamax, AdamW, RMSprop

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import ATOL, equalize_linear_weights
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.tp_registry import get_weight_slice

NUM_STEPS = 50
ATOL = 2e-4

# (optimizer, learning_rate, weight_decay)
# Adamax and RMSprop get smaller learning rate because the lack of bias correction in gradient moments
# causes very large updates in the first steps
OPTIMIZERS = [
    (Adam, 1e-3, 0.0),
    (AdamW, 1e-3, 0.1),
    (SGD, 1e-3, 0.1),
    (Adamax, 1e-4, 0.1),
    (RMSprop, 1e-4, 0.0),
]


def print_stats(p, original_param):
    diff = torch.abs(p.cpu() - original_param.cpu())
    print("PARAMS", smp.rank(), p.cpu(), original_param.cpu())
    print("L-inf", torch.max(diff))
    print("MEAN", torch.sum(diff) / p.numel())
    print("% OUTLIERS", 100 * torch.nonzero(diff > ATOL).shape[0] / torch.numel(p))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_parallel_degree", type=int, default=1)
    parser.add_argument("--tensor_parallel_degree", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=1)
    return parser.parse_args()


def compare_params(tp_degree, pp_degree, microbatches, opt_class, lr, wd):
    batch_size = 16

    smp.init(
        {
            "pipeline_parallel_degree": pp_degree,
            "microbatches": microbatches,
            "tensor_parallel_degree": tp_degree,
            "ddp": True,
            "shard_optimizer_state": True,
            "auto_partition": False,
            "default_partition": 0,
        }
    )

    torch.manual_seed(42)
    torch.cuda.set_device(smp.local_rank())

    dim = 1024

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # assuming pp degree at most 2 here
            with smp.partition(pp_degree - 1):
                self.lin0 = nn.Linear(dim, dim)
                self.lin1 = nn.Linear(dim, dim)
            with smp.tensor_parallelism():
                self.lin2 = nn.Linear(dim, dim)
                self.lin3 = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.lin0(x)
            x = self.lin1(x)
            x = self.lin2(x)
            return torch.flatten(self.lin3(x), 1)

    model = Model()
    model_clone = Model()
    model_clone2 = Model()
    model_clone.load_state_dict(model.state_dict())
    model_clone2.load_state_dict(model.state_dict())

    optimizer = opt_class(model.parameters(), lr=lr, weight_decay=wd)

    device = torch.device("cuda", smp.local_rank())
    x_data = [torch.randn(batch_size, 20, dim).to(device) for _ in range(NUM_STEPS)]
    y_data = [torch.randint(0, 10, (batch_size,)).to(device) for _ in range(NUM_STEPS)]

    def nonsmp_train_step(x, y):
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y, reduction="mean")
        loss.backward()
        optimizer.step()

    model.to(device)

    for t in range(NUM_STEPS):
        nonsmp_train_step(x_data[t], y_data[t])

    original_params = {}
    for m, m2 in zip(model.modules(), model_clone2.modules()):
        for n, p in m.named_parameters(recurse=False):
            # use the clone module as the key since it will be checked against that
            original_params[(m2, n)] = p.data

    original_grads = {}
    for n, p in model.named_parameters():
        original_grads[n] = p.grad.detach()

    # run distributed
    torch.manual_seed(42 + smp.dp_rank())
    dist_model = smp.DistributedModel(model_clone)

    dist_optimizer = smp.DistributedOptimizer(
        opt_class(dist_model.parameters(), lr=lr, weight_decay=wd)
    )

    module_pairs = [
        (model_clone2.lin0, dist_model.module.module.lin0, None),
        (model_clone2.lin1, dist_model.module.module.lin1, None),
        (model_clone2.lin2, dist_model.module.module.lin2, "input"),
        (model_clone2.lin3, dist_model.module.module.lin3, "input"),
    ]

    for mod, dist_mod, ptn in module_pairs:
        equalize_linear_weights(mod, dist_mod, ptn)

    @smp.step
    def train_step(x, y):
        out = dist_model(x)
        loss = F.nll_loss(out, y, reduction="mean")
        dist_model.backward(loss)

    x_data = [x.to(device) for x in x_data]
    y_data = [y.to(device) for y in y_data]
    local_bs = batch_size // smp.dp_size()

    for x, y in zip(x_data, y_data):
        dist_optimizer.zero_grad()
        train_step(
            x[smp.dp_rank() * local_bs : (smp.dp_rank() + 1) * local_bs],
            y[smp.dp_rank() * local_bs : (smp.dp_rank() + 1) * local_bs],
        )

        # for red_type in dist_model.param_name_to_dp_rank:
        #    scaled_batch = red_type.value == 0
        #    param_name_to_dp_rank = dist_model.param_name_to_dp_rank[red_type]
        #    my_rank = smp.rdp_rank() if scaled_batch else smp.dp_rank()

        #    my_params_with_opt_state = {
        #        name for name, rank in param_name_to_dp_rank.items() if rank == my_rank
        #    }

        #    for n, p in dist_model.local_named_parameters():
        #        if n in my_params_with_opt_state:
        #            for org_n, org_g in original_grads.items():
        #                if n.endswith(org_n):
        #                    original_grad = org_g
        #                    break
        #            else:
        #                raise ValueError(f"Could not find {n} vs {org_n}")

        #            if scaled_batch and org_n.endswith("weight"):
        #                grad_slice = get_weight_slice(original_grad, "input")
        #                self.assertTrue(torch.allclose(p.grad.cpu(), grad_slice.cpu(), atol=ATOL))
        #                # print_stats(p.grad, grad_slice)
        #            else:
        #                self.assertTrue(
        #                    torch.allclose(p.grad.cpu(), original_grad.cpu(), atol=ATOL)
        #                )
        #                # print_stats(p.grad, original_grad)

        dist_optimizer.step()

    for mod, dist_mod, ptn in module_pairs:
        if smp.pp_rank() == state.module_manager.get_partition(dist_mod):
            for n, p in dist_mod.named_parameters(recurse=False):
                if smp.tp_rank() != 0 and n == "bias":
                    continue

                if n == "bias":
                    original_param = original_params[(mod, n)]
                else:
                    original_param = get_weight_slice(original_params[(mod, n)], ptn)
                try:
                    assert torch.allclose(
                        p.cpu(), original_param.cpu(), atol=ATOL
                    ), f"Parameter {n} outside of tolerance threshold"
                except:
                    print_stats(p, original_param)
                    raise
    smp.barrier()
    if smp.rank() == 0:
        print("OK")


def main(args):
    for opt, lr, wd in OPTIMIZERS:
        compare_params(
            args.tensor_parallel_degree,
            args.pipeline_parallel_degree,
            args.microbatches,
            opt,
            lr,
            wd,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_parallel_degree", type=int, default=1)
    parser.add_argument("--tensor_parallel_degree", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("unittest_args", nargs="*")

    args = parser.parse_args()

    main(args)
