# Standard Library

# Third Party
import torch
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.state_mod import state

from smdistributed.modelparallel.torch.amp import GradScaler  # noqa isort:skip


@smp.step
def train_step(model, data1, data2, out_grads):
    output = model(data1, data2)
    model.backward(output, out_grads)


class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x, y):
        z1 = self.linear1(x)
        _ = self.linear1(y)
        return z1


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net2 = Net2()

    def forward(self, x, y):
        z1 = self.net2(x, y)
        return z1


smp.reset()
torch.manual_seed(42)
model = Net()
torch.cuda.set_device(0)
device = torch.device("cuda")
x = torch.randn(16, 10)
x = x.to(device)
x.requires_grad = True
y = torch.randn(16, 10)
y = y.to(device)
y.requires_grad = True
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
state.module_manager.assign_partition(model.net2, 1)
state.module_manager.assign_partition(model.net2.linear1, 1)
state.module_manager.assign_partition(model.net2.linear2, 1)
model = smp.DistributedModel(model, average_grads_across_microbatches=False)
output_dist = train_step(model, x, y, out_grads)
