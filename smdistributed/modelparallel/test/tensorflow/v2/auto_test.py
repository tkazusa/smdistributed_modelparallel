#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf
from mock import patch

# First Party
import smdistributed.modelparallel.tensorflow as smp
from smdistributed.modelparallel.tensorflow import auto
from smdistributed.modelparallel.tensorflow.state_mod import TFModelParallelState
from smdistributed.modelparallel.tensorflow.v2 import serialization
from smdistributed.modelparallel.tensorflow.v2.serialization import SerializedGraphV2
from smdistributed.modelparallel.test.tensorflow.auto_test_utils import (
    constant_return_model,
    multi_consumer_model,
    multi_model_input_consumer_model,
    one_input_model,
    set_mock_state,
    tensor_grouping_model,
    two_comm_model,
    two_input_model,
)

NUM_MICROBATCHES = 2
PARTITIONS = 2


class TestDistributedModelV2(unittest.TestCase):
    def setUp(self):
        # Patch test dependencies (this is roughly equivalent to @patch).

        state_patch = patch(
            "smdistributed.modelparallel.tensorflow.v2.model.state", autospec=TFModelParallelState()
        )
        self.addCleanup(state_patch.stop)
        self.mock_state = state_patch.start()

        # Configure test mocks.
        set_mock_state(self.mock_state)

        serialization.state = self.mock_state

        self.mock_state.serialized_graph = SerializedGraphV2()
        auto.state = self.mock_state

    def assert_has_correct_op_counts(self, partitioned_graphs, input_count, output_count):
        for mb in range(NUM_MICROBATCHES):
            partitions = [graph[mb] for graph in partitioned_graphs]

            for dev in range(len(partitioned_graphs)):
                dev_partition = [n for n in partitions[dev].node]

                self.assert_op_type_count(dev_partition, "SmpInput", input_count)
                self.assert_op_type_count(dev_partition, "SmpOutput", output_count)

    def assert_op_type_count(self, collection, op_type, count):
        self.assertEqual(len([n for n in collection if n.op == op_type]), count)

    def test_one_input_partition(self):
        class Model(smp.DistributedModel):
            def __init__(self):
                super(Model, self).__init__()

            def call(self, x):
                return one_input_model(x)

        auto.state.cfg.pipeline_parallel_degree = 2
        model = Model()
        model.partition([tf.TensorSpec(shape=[], dtype=tf.float32)], {})

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 4, 2)

    def test_two_input_partition(self):
        class Model(smp.DistributedModel):
            def __init__(self):
                super(Model, self).__init__()

            def call(self, x, y):
                return two_input_model(x, y)

        auto.state.cfg.pipeline_parallel_degree = 2
        model = Model()
        model.partition(
            [tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)],
            {},
        )

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 5, 2)

    def test_two_comm_partition(self):
        class Model(smp.DistributedModel):
            def __init__(self):
                super(Model, self).__init__()

            def call(self, x, y):
                return two_comm_model(x, y)

        auto.state.cfg.pipeline_parallel_degree = 2
        model = Model()
        model.partition(
            [tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)],
            {},
        )

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 6, 3)

    def test_multi_consumer_partition(self):
        class Model(smp.DistributedModel):
            def __init__(self):
                super(Model, self).__init__()

            def call(self, x, y):
                return multi_consumer_model(x, y)

        auto.state.cfg.pipeline_parallel_degree = 2
        model = Model()
        model.partition(
            [tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)],
            {},
        )

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 6, 3)

    def test_multi_model_input_consumer_partition(self):
        class Model(smp.DistributedModel):
            def __init__(self):
                super(Model, self).__init__()

            def call(self, x, y):
                return multi_model_input_consumer_model(x, y)

        auto.state.cfg.pipeline_parallel_degree = 2
        model = Model()
        model.partition(
            [tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)],
            {},
        )

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 6, 3)

    def test_constant_return_partition(self):
        class Model(smp.DistributedModel):
            def __init__(self):
                super(Model, self).__init__()

            def call(self, x):
                return constant_return_model(x)

        auto.state.cfg.pipeline_parallel_degree = 2
        model = Model()
        model.partition([tf.TensorSpec(shape=[], dtype=tf.float32)], {})

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 4, 3)

    def test_tensor_grouping(self):
        class Model(smp.DistributedModel):
            def __init__(self):
                super(Model, self).__init__()

            def call(self, x):
                return tensor_grouping_model(x)

        auto.state.cfg.pipeline_parallel_degree = 2
        model = Model()
        model.partition([tf.TensorSpec(shape=[], dtype=tf.float32)], {})

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 5, 3)


if __name__ == "__main__":
    unittest.main()
