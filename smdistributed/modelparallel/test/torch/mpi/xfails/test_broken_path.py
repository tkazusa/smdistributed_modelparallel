# Standard Library

# Third Party
import torch
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.state_mod import state

from smdistributed.modelparallel.torch.amp import GradScaler  # noqa isort:skip


@smp.step
def train_step(model, data1, out_grads):
    output = model(data1)
    model.backward(output, out_grads)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        z1 = self.linear1(x)
        z2 = self.linear2(x)
        return z1


smp.reset()
torch.manual_seed(42)
model = Net()
torch.cuda.set_device(0)
device = torch.device("cuda")
x = torch.randn(16, 10)
x = x.to(device)
out_grads = torch.ones(16, 10)
out_grads = out_grads.to(device)
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
state.module_manager.assign_partition(model.linear2, 1)
model = smp.DistributedModel(model, average_grads_across_microbatches=False)
output_dist = train_step(model, x, out_grads)
