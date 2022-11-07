# Standard Library
import logging

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.exceptions import StepFunctionCalledError
from smdistributed.modelparallel.torch.state_mod import state

logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)


@smp.step
def train_step(x, y, dist_model):
    out = dist_model(x)
    loss = F.nll_loss(out, y, reduction="mean")
    dist_model.backward(loss)


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.lin0_0 = nn.Linear(dim, dim)
        self.lin0_1 = nn.Linear(dim, dim)
        self.lin0_2 = nn.Linear(dim, dim)
        self.lin0_3 = nn.Linear(dim, dim)
        self.lin0_4 = nn.Linear(dim, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.lin3 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.lin0_0(x)
        x = self.lin0_1(x)
        x = self.lin0_2(x)
        x = self.lin0_3(x)
        x = self.lin0_4(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return torch.flatten(self.lin3(x), 1)


torch.manual_seed(42)
batch_size = 16
pp_degree = 4
dim = 1024
smp.init(
    {
        "pipeline_parallel_degree": pp_degree,
        "ddp": True,
        "auto_partition": False,
        "default_partition": 0,
    }
)
torch.cuda.set_device(smp.local_rank())
device = torch.device("cuda", smp.local_rank())

model = Model(dim)
model = smp.DistributedModel(model)
# set_partition for rank 0 is not necessary as default_partition is 0, just put here to be clear
smp.set_partition(model.get_module().lin0_0, 0)
smp.set_partition(model.get_module().lin0_1, 0)
smp.set_partition(model.get_module().lin0_2, 0)
smp.set_partition(model.get_module().lin0_3, 0)
smp.set_partition(model.get_module().lin0_4, 0)
smp.set_partition(model.get_module().lin1, 1)
smp.set_partition(model.get_module().lin2, 2)
smp.set_partition(model.get_module().lin3, 3)

x_data = torch.randn(batch_size, 20, dim).to(device)
y_data = torch.randint(0, 10, (batch_size,)).to(device)


def verify_parition(model):
    assert state.module_manager.get_partition(model.get_module().lin0_0) == 0
    assert state.module_manager.get_partition(model.get_module().lin0_1) == 0
    assert state.module_manager.get_partition(model.get_module().lin0_2) == 0
    assert state.module_manager.get_partition(model.get_module().lin0_3) == 0
    assert state.module_manager.get_partition(model.get_module().lin0_4) == 0
    assert state.module_manager.get_partition(model.get_module().lin1) == 1
    assert state.module_manager.get_partition(model.get_module().lin2) == 2
    assert state.module_manager.get_partition(model.get_module().lin3) == 3


train_step(x_data, y_data, model)

verify_parition(model)

partition_info = state.module_manager.get_model_partition_info()

model = None
smp.reset()
smp.init(
    {
        "pipeline_parallel_degree": pp_degree,
        "ddp": True,
        "auto_partition": False,
        "default_partition": 0,
    }
)

model = Model(dim)
model = smp.DistributedModel(model)
model.load_saved_partition(partition_info)

# Set a different partition, test that loaded partition is perserved
smp.set_partition(model.get_module().lin0_0, 0)
smp.set_partition(model.get_module().lin0_1, 0)
smp.set_partition(model.get_module().lin0_2, 1)
smp.set_partition(model.get_module().lin0_3, 2)
smp.set_partition(model.get_module().lin0_4, 3)
smp.set_partition(model.get_module().lin1, 1)
smp.set_partition(model.get_module().lin2, 2)
smp.set_partition(model.get_module().lin3, 3)

train_step(x_data, y_data, model)
verify_parition(model)

try:
    model.load_saved_partition(partition_info)
except StepFunctionCalledError as e:
    print(f"Error {e} caught succesfully.")

model = None
smp.reset()
# Test load with auto_partition=True
smp.init({"pipeline_parallel_degree": pp_degree, "ddp": True, "auto_partition": True})

model = Model(dim)
model = smp.DistributedModel(model)
model.load_saved_partition(partition_info)

train_step(x_data, y_data, model)
verify_parition(model)

smp.barrier()
if smp.rank() == 0:
    print("Loading partition test finished succesfully!")
