# Standard Library
import unittest

# First Party
from smdistributed.modelparallel.torch.module_partition import ModulePartitioner


class MockModuleNode:
    def __init__(self, cost):
        self.normalized_cost = cost
        self.dummy = False
        self.num_descendants = 100  # set large to avoid "too small model" errors


class TestPartition(unittest.TestCase):
    def verify_partition(self, cost_list, partitions, expected_partition):
        node_list = [MockModuleNode(cost) for cost in cost_list]
        p = ModulePartitioner(None, len(partitions))  # arguments do not matter for this test
        output_partition, _ = p.partition_children(
            node_list, partitions, partitions[0], [0 for p in partitions]
        )

        print(output_partition)

        # here we only verify the partition up to a relabeling
        seen = set()
        for exp, out in zip(expected_partition, output_partition):
            self.assertEqual(len(exp), len(out))
            seen = seen.union(set(out))

        self.assertTrue(len(seen) == len(partitions))

    def test_single_node(self):
        cost_list = [1.0]
        partitions = list(range(4))
        expected_partition = [[0, 1, 2, 3]]

        self.verify_partition(cost_list, partitions, expected_partition)

    def test_balanced_even_chain(self):
        cost_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        partitions = list(range(4))
        expected_partition = [[0], [0], [1], [1], [2], [2], [3], [3]]

        self.verify_partition(cost_list, partitions, expected_partition)

    def test_even_small_group(self):
        cost_list = [0.4, 0.6]
        partitions = list(range(4))
        expected_partition = [[0, 2], [2, 3]]

        self.verify_partition(cost_list, partitions, expected_partition)

    def test_uneven_small_group(self):
        cost_list = [0.2, 0.8]
        partitions = list(range(4))
        expected_partition = [[0], [1, 2, 3]]

        self.verify_partition(cost_list, partitions, expected_partition)

    def test_balanced_uneven_chain(self):
        cost_list = [0.2, 0.2, 0.2, 0.2, 0.2]
        partitions = list(range(4))
        expected_partition = [[0], [0], [1], [2], [3]]

        self.verify_partition(cost_list, partitions, expected_partition)

    def test_imbalanced_chain(self):
        cost_list = [0.03, 0.05, 0.07, 0.4, 0.08, 0.05, 0.17, 0.04, 0.04, 0.04, 0.08]
        partitions = list(range(8))
        expected_partition = [[0], [0], [0], [0, 2, 4, 7], [5], [5], [1, 6], [1], [3], [3], [3]]

        self.verify_partition(cost_list, partitions, expected_partition)

    def test_imbalanced_small_group(self):
        cost_list = [0.1, 0.2, 0.7]
        partitions = list(range(4))
        expected_partition = [[2], [2], [0, 1, 3]]

        self.verify_partition(cost_list, partitions, expected_partition)


if __name__ == "__main__":
    unittest.main()
