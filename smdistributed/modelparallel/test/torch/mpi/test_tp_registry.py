# Standard Library
import unittest

# Third Party
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp


class TestTensorParallelismRegistry(unittest.TestCase):
    def verify_distribution(self, custom_module_cls, init_hook=True):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                if init_hook:
                    self.custom_module = custom_module_cls(64)
                else:
                    self.custom_module = custom_module_cls(64, 64)
                self.lin1 = nn.Linear(64, 64)
                self.lin2 = nn.Linear(64, 64)

            def forward(self, x):
                x = self.custom_module(x)
                x = self.lin1(x)
                return self.lin2(x)

        smp.init({"pipeline_parallel_degree": 1, "tensor_parallel_degree": 2, "ddp": True})

        smp.state.model = None
        model = Model()
        smp.set_tensor_parallelism(model.custom_module, True)
        model = smp.DistributedModel(model)

        assert type(model.module.module.custom_module) == smp.nn.DistributedLinear
        assert all([not isinstance(m, custom_module_cls) for m in model.modules()])

    def test_register(self):
        smp.state.tp_registry.reset()

        @smp.tp_register(smp.nn.DistributedLinear)
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

        self.verify_distribution(CustomModule, init_hook=False)

    def test_register_with_hooks(self):
        smp.state.tp_registry.reset()

        @smp.tp_register(smp.nn.DistributedLinear, init_hook=lambda x: ((x, x), {}))
        class CustomModule(nn.Module):
            def __init__(self, features):
                super(CustomModule, self).__init__()
                self.id1 = nn.Identity()
                self.id2 = nn.Identity()
                self.lin = nn.Linear(features, features)

            def forward(self, x):
                x = self.id1(x)
                x = self.lin(x)
                return self.id2(x)

        self.verify_distribution(CustomModule, init_hook=True)

    def test_register_with_module(self):
        smp.state.tp_registry.reset()

        class CustomModule(nn.Module):
            def __init__(self, features):
                super(CustomModule, self).__init__()
                self.id1 = nn.Identity()
                self.id2 = nn.Identity()
                self.lin = nn.Linear(features, features)

            def forward(self, x):
                x = self.id1(x)
                x = self.lin(x)
                return self.id2(x)

        smp.tp_register_with_module(
            CustomModule, smp.nn.DistributedLinear, init_hook=lambda x: ((x, x), {})
        )
        self.verify_distribution(CustomModule, init_hook=True)


if __name__ == "__main__":
    unittest.main()
