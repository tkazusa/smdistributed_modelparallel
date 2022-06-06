# Standard Library

# Third Party
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp

smp.state.tp_registry.reset()


@smp.tp_register(smp.nn.DistributedTransformer)
class CustomModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomModule, self).__init__()
        self.id1 = nn.Identity()
        self.id2 = nn.Identity()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.id1(x)
        x = self.lin(x)
        return self.id2(x)


model = CustomModule(64, 64)
smp.init({"pipeline_parallel_degree": 1, "tensor_parallel_degree": 2, "ddp": True})
smp.set_tensor_parallelism(model, True)
model = smp.DistributedModel(model)
