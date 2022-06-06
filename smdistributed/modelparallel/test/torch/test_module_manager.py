# Third Party
# Standard Library
import unittest

import torch
import torch.nn as nn
from torch.autograd import set_grad_enabled

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import Net1, Net2
from smdistributed.modelparallel.torch.patches.tracing import TracingEnd
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.step import step


class TestModuleManager(unittest.TestCase):
    def test_module_manager_serialization(self):
        smp.init(
            {
                "microbatches": 2,
                "pipeline": "simple",
                "partitions": 1,
                "auto_partition": False,
                "default_partition": 0,
            }
        )

        class SeqModel(nn.Module):
            def __init__(self, N, D_in, H, D_out):
                super(SeqModel, self).__init__()
                with smp.partition(0):
                    self.lin = nn.Linear(D_in, H)
                    with smp.partition(0):
                        self.relu = nn.ReLU()
                with smp.partition(0):
                    self.lin2 = nn.Linear(H, D_out)

            def forward(self, x, cool=False):
                x = self.lin(x)
                x = self.relu(x)
                x = self.lin2(x)
                return x

        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        x = x.to(torch.device("cuda", smp.local_rank()))
        y = y.to(torch.device("cuda", smp.local_rank()))
        model = SeqModel(N, D_in, H, D_out)
        model = smp.DistributedModel(model)
        smp.state.module_manager.name_modules_and_create_parent_map()

        try:
            set_grad_enabled(False)
            with state.patch_manager.patch_for_trace(
                state.model, device=torch.device("cuda", smp.local_rank())
            ):
                y_pred = model(x, cool=True)
        except TracingEnd:
            pass

        mm = smp.state.module_manager
        s = mm.get_serialized_partitioning_and_tracing_states()
        assert all([isinstance(k, str) for k in s[0]])
        mm.reset()
        mm.load_partitioning_and_trace_results(s)
        s2 = mm.get_serialized_partitioning_and_tracing_states()
        assert s == s2

    @unittest.skip
    def test_local_exec_stack(self):
        smp.init(
            {
                "microbatches": 2,
                "pipeline": "_only_forward",
                "partitions": 1,
                "auto_partition": False,
                "default_partition": 0,
            }
        )

        class Lin1(nn.Linear):
            def forward(self, x):
                assert smp.state.module_manager.execution_stack == ["main", "main/lin1"]
                assert isinstance(smp.state.module_manager.get_parent_module(self), SeqModel)
                return super(Lin1, self).forward(x)

        class Lin2(nn.Linear):
            def forward(self, x):
                assert smp.state.module_manager.execution_stack == ["main", "main/lin2"]
                smp.state.module_manager.execution_stack = ["main", "main/lin2"]
                assert smp.state.module_manager.execution_stack == ["main", "main/lin2"]
                return super(Lin2, self).forward(x)

        class SeqModel(nn.Module):
            def __init__(self, N, D_in, H, D_out):
                super(SeqModel, self).__init__()
                with smp.partition(0):
                    self.lin1 = Lin1(D_in, H)
                    self.relu = nn.ReLU()
                    self.lin2 = Lin2(H, D_out)

            def forward(self, x, cool=False):
                assert smp.state.module_manager.execution_stack == ["main"]
                x = self.lin1(x)
                x = self.relu(x)
                x = self.lin2(x)
                return x

        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = SeqModel(N, D_in, H, D_out)
        model = smp.DistributedModel(model)
        smp.state.module_manager.name_modules_and_create_parent_map()

        x = x.to(torch.device("cuda"))
        y = y.to(torch.device("cuda"))

        @step
        def s(m, i, o):
            f = m(i)
            assert smp.state.module_manager.execution_stack == []
            return f

        o = s(model, x, y)

    def test_recursive_recv_check(self):
        smp.init({"microbatches": 1, "pipeline": "_only_forward", "partitions": 1})

        class Net3(nn.Module):
            def __init__(self):
                super(Net3, self).__init__()
                self.net1 = Net1()
                self.net2 = Net2()

            def forward(self, x):
                x = self.net1(x)
                x = self.net2(x)
                return x

        class Net4(nn.Module):
            def __init__(self):
                super(Net4, self).__init__()
                self.net3 = Net3()

            def forward(self, x):
                x = self.net3(x)
                return x

        x = torch.randn(10, 1000)
        model = Net4()
        model = smp.DistributedModel(model)
        smp.state.module_manager.assign_partition(model.module.net3.net1.linear1, 1)
        smp.state.module_manager.assign_partition(model, 0)
        smp.state.module_manager.assign_partition(model.module, 0)
        smp.state.module_manager.assign_partition(model.module.net3, 1)
        smp.state.module_manager.assign_partition(model.module.net3.net1, 0)
        smp.state.module_manager.assign_partition(model.module.net3.net2, 0)
        smp.state.module_manager.assign_partition(model.module.net3.net1.linear1, 0)
        smp.state.module_manager.assign_partition(model.module.net3.net1.linear2, 0)
        smp.state.module_manager.assign_partition(model.module.net3.net2.linear1, 0)
        smp.state.module_manager.assign_partition(model.module.net3.net2.linear1, 0)
        smp.state.module_manager.assign_partition(model.module.net3.net2.linear2, 0)

        smp.state.module_manager.name_modules_and_create_parent_map()

        assert smp.state.module_manager.check_no_pending_bwd(0, model) == True
        smp.state.module_manager.increment_bwd_count(0, model.module.net3, model.module)
        smp.state.module_manager.increment_bwd_count(0, model.module.net3.net1, model.module.net3)
        assert smp.state.module_manager.check_no_pending_bwd(0, model.module) == False
        smp.state.module_manager.execution_stack = [
            "main",
            "main/module/net3",
            "main/module/net3/net2",
        ]
        smp.state.module_manager._module_execution_stack = [
            model,
            model.module,
            model.module.net3,
            model.module.net3.net2,
        ]
        mod, child_mod = smp.state.module_manager.find_boundary_ancestors(
            0, model.module.net3.net2.linear2
        )
        assert mod == model.module.net3 and child_mod == model.module.net3.net2

        # add dummy send test
        smp.state.module_manager.increment_dummy_bwd_sends(0, model.module.net3, model.module)
        assert smp.state.module_manager.num_dummy_sends(0, model.module.net3, model.module) == 1

        smp.state.module_manager.decrement_bwd_count(0, model.module.net3, model.module, 1)
        smp.state.module_manager.decrement_bwd_count(
            0, model.module.net3.net1, model.module.net3, 1
        )
        smp.state.module_manager.increment_bwd_count(
            0, model.module.net3.net1.linear1, model.module.net3.net1
        )
        assert smp.state.module_manager.check_no_pending_bwd(0, model.module) == False

        class Net5(nn.Module):
            def __init__(self):
                super(Net5, self).__init__()
                self.net3 = torch.nn.ModuleList([Net3()])

            def forward(self, x):
                out = self.net3[0](x)
                return out

        smp.init({"microbatches": 1, "pipeline": "_only_forward", "partitions": 1})
        x = torch.randn(10, 1000)
        model = Net5()
        model = smp.DistributedModel(model)
        smp.state.module_manager.assign_partition(model, 0)
        smp.state.module_manager.assign_partition(model.module, 0)
        smp.state.module_manager.assign_partition(model.module.net3, 0)
        smp.state.module_manager.assign_partition(model.module.net3[0], 1)

        smp.state.module_manager.name_modules_and_create_parent_map()

        assert smp.state.module_manager.check_no_pending_bwd(0, model) == True
        smp.state.module_manager.increment_bwd_count(0, model.module.net3[0], model.module)
        assert smp.state.module_manager.check_no_pending_bwd(0, model.module) == False
        smp.state.module_manager.execution_stack = ["main", "main/module", "main/module/net3/0"]
        grand_parent, parent = smp.state.module_manager.get_immediate_ancestors(
            model.module.net3[0]
        )
        assert grand_parent == model.module and parent == model.module.net3
        smp.state.module_manager.execution_stack = ["main", "main/module"]
        grand_parent, parent = smp.state.module_manager.get_immediate_ancestors(model.module.net3)
        assert grand_parent == model and parent == model.module
        grand_parent, parent = smp.state.module_manager.get_immediate_ancestors(model.module)
        assert grand_parent == None and parent == model

    def test_tensor_parallelism(self):
        smp.init({"partitions": 1})

        class Net3(nn.Module):
            def __init__(self):
                super(Net3, self).__init__()
                self.net1 = Net1()
                with smp.tensor_parallelism():
                    self.net2 = Net2()
                with smp.tensor_parallelism(False):
                    self.net2_copy = Net2()

            def forward(self, x):
                x = self.net1(x)
                x = self.net2(x)
                return x

        class Net4(nn.Module):
            def __init__(self):
                super(Net4, self).__init__()

                with smp.tensor_parallelism():
                    self.net3 = Net3()

            def forward(self, x):
                x = self.net3(x)
                return x

        mod = Net4()
        smp.state.model = mod
        smp.state.module_manager.simplify_tensor_parallelism_modules(mod)
        expected_tp_modules1 = {
            mod.net3.net1.linear1,
            mod.net3.net1.linear2,
            mod.net3.net2.linear1,
            mod.net3.net2.linear2,
        }
        assert smp.state.module_manager._tensor_parallelism_modules == expected_tp_modules1

        smp.state.module_manager._tensor_parallelism_modules = set()
        mod = Net3()
        smp.state.model = mod
        smp.state.module_manager.simplify_tensor_parallelism_modules(mod)
        expected_tp_modules2 = {mod.net2.linear1, mod.net2.linear2}
        assert smp.state.module_manager._tensor_parallelism_modules == expected_tp_modules2

        smp.state.module_manager._tensor_parallelism_modules = set()

        org_is_supported = smp.state.tp_registry.is_supported

        def new_is_supported(module_cls):
            if module_cls == Net3:
                return True
            else:
                return org_is_supported(module_cls)

        smp.state.tp_registry.is_supported = new_is_supported

        mod = Net4()
        smp.state.model = mod
        smp.state.module_manager.simplify_tensor_parallelism_modules(mod)
        expected_tp_modules3 = {mod.net3}
        assert smp.state.module_manager._tensor_parallelism_modules == expected_tp_modules3

        smp.state.model = None
        smp.state.module_manager._tensor_parallelism_modules = set()
        smp.state.tp_registry.is_supported = org_is_supported

        mod = Net4()
        smp.set_tensor_parallelism(mod.net3.net2_copy)
        smp.state.model = mod
        smp.state.model.partitioned = False
        smp.state.module_manager.simplify_tensor_parallelism_modules(mod)
        expected_tp_modules4 = {
            mod.net3.net1.linear1,
            mod.net3.net1.linear2,
            mod.net3.net2.linear1,
            mod.net3.net2.linear2,
            mod.net3.net2_copy.linear1,
            mod.net3.net2_copy.linear2,
        }
        assert smp.state.module_manager._tensor_parallelism_modules == expected_tp_modules4

    def test_activation_checkpointing(self):
        smp.init({"partitions": 1})
        m = nn.Linear(4, 4)
        with self.assertRaises(RuntimeError):
            smp.set_activation_checkpointing(m)


if __name__ == "__main__":
    unittest.main()
