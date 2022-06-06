# Standard Library
import unittest

# Third Party
import torch.nn as nn

# First Party
from smdistributed.modelparallel.torch.module_manager import TraceResults
from smdistributed.modelparallel.torch.module_partition import ModulePartitioner


class TestPartition(unittest.TestCase):
    def assert_correct_partition(self, model, module_order, num_partitions, expected_partition):
        trace_results = TraceResults(module_order, {}, {}, {}, {})
        partitioner = ModulePartitioner(model, num_partitions, trace_results)
        output_partition = partitioner.partition()
        self.assertEqual(output_partition, expected_partition)

    def test_simple(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.relu1 = nn.ReLU()
                self.relu2 = nn.ReLU()
                self.relu3 = nn.ReLU()

        model = Model()
        module_order = [model, model.relu1, model.relu2, model.relu3]
        num_partitions = 2
        expected_partition = {model: 0, model.relu1: 0, model.relu2: 1, model.relu3: 1}

        self.assert_correct_partition(model, module_order, num_partitions, expected_partition)

    def test_with_param(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear1 = nn.Linear(200, 15)
                self.relu2 = nn.ReLU()
                self.linear3 = nn.Linear(40, 20)
                self.linear4 = nn.Linear(10, 200)

        model = Model()
        module_order = [model, model.linear1, model.relu2, model.linear3, model.linear4]
        num_partitions = 2
        expected_partition = {
            model: 0,
            model.linear1: 0,
            model.relu2: 0,
            model.linear3: 1,
            model.linear4: 1,
        }

        self.assert_correct_partition(model, module_order, num_partitions, expected_partition)

    def test_with_imbalanced_param(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear1 = nn.Linear(2000, 15)
                self.relu2 = nn.ReLU()
                self.linear3 = nn.Linear(40, 20)
                self.linear4 = nn.Linear(10, 200)

        model = Model()
        module_order = [model, model.linear1, model.relu2, model.linear3, model.linear4]
        num_partitions = 2
        expected_partition = {
            model: 0,
            model.linear1: 1,
            model.relu2: 0,
            model.linear3: 0,
            model.linear4: 0,
        }

        self.assert_correct_partition(model, module_order, num_partitions, expected_partition)

    def test_with_submodule(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.linear1 = nn.Linear(2000, 15)
                self.relu2 = nn.ReLU()

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.submodule1 = SubModule()
                self.linear3 = nn.Linear(40, 20)
                self.submodule2 = SubModule()
                self.linear4 = nn.Linear(10, 200)

        model = Model()
        module_order = [
            model,
            model.submodule1,
            model.submodule1.linear1,
            model.submodule1.relu2,
            model.linear3,
            model.submodule2,
            model.submodule2.linear1,
            model.submodule2.relu2,
            model.linear4,
        ]
        num_partitions = 2
        expected_partition = {
            model: 0,
            model.submodule1: 0,
            model.submodule2: 1,
            model.submodule1.linear1: 0,
            model.submodule1.relu2: 0,
            model.submodule2.linear1: 1,
            model.submodule2.relu2: 1,
            model.linear3: 0,
            model.linear4: 1,
        }

        self.assert_correct_partition(model, module_order, num_partitions, expected_partition)

    def test_with_tied_parameter(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.linear1 = nn.Linear(2000, 15)
                self.relu2 = nn.ReLU()

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.submodule1 = SubModule()
                self.linear3 = nn.Linear(10, 200)
                self.submodule2 = SubModule()
                self.linear4 = nn.Linear(10, 200)
                self.linear4.weight = self.linear3.weight

        model = Model()
        module_order = [
            model,
            model.submodule1,
            model.submodule1.linear1,
            model.submodule1.relu2,
            model.linear3,
            model.submodule2,
            model.submodule2.linear1,
            model.submodule2.relu2,
            model.linear4,
        ]
        num_partitions = 2
        expected_partition = {
            model: 0,
            model.submodule1: 0,
            model.submodule2: 1,
            model.submodule1.linear1: 0,
            model.submodule1.relu2: 0,
            model.submodule2.linear1: 1,
            model.submodule2.relu2: 1,
            model.linear3: 1,
            model.linear4: 1,
        }

        self.assert_correct_partition(model, module_order, num_partitions, expected_partition)

    def test_with_w_parameter_sharing_pattern(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.linear1 = nn.Linear(2000, 15)
                self.relu2 = nn.ReLU()

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.submodule1 = SubModule()
                self.linear3 = nn.Linear(10, 200)
                self.submodule2 = SubModule()
                self.linear4 = nn.Linear(10, 200)
                self.linear5 = nn.Linear(10, 200)
                self.linear4.weight = self.linear3.weight
                self.linear4.bias = self.linear5.bias

        model = Model()
        module_order = [
            model,
            model.submodule1,
            model.submodule1.linear1,
            model.submodule1.relu2,
            model.linear3,
            model.submodule2,
            model.submodule2.linear1,
            model.submodule2.relu2,
            model.linear4,
        ]
        num_partitions = 2
        expected_partition = {
            model: 0,
            model.submodule1: 0,
            model.submodule2: 1,
            model.submodule1.linear1: 0,
            model.submodule1.relu2: 0,
            model.submodule2.linear1: 1,
            model.submodule2.relu2: 1,
            model.linear3: 1,
            model.linear4: 1,
            model.linear5: 1,
        }

        self.assert_correct_partition(model, module_order, num_partitions, expected_partition)


if __name__ == "__main__":
    unittest.main()
