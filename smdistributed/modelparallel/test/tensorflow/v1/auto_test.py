#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf
from mock import patch

# First Party
from smdistributed.modelparallel.tensorflow import auto
from smdistributed.modelparallel.tensorflow.compile import CompileStatus
from smdistributed.modelparallel.tensorflow.state_mod import TFModelParallelState
from smdistributed.modelparallel.tensorflow.v1 import model, serialization
from smdistributed.modelparallel.tensorflow.v1.serialization import SerializedGraphV1, TraceGraph
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


class TestDistributedModelV1(unittest.TestCase):
    def setUp(self):
        # Patch test dependencies (this is roughly equivalent to @patch).

        state_patch = patch(
            "smdistributed.modelparallel.tensorflow.v1.model.state", autospec=TFModelParallelState()
        )
        self.addCleanup(state_patch.stop)
        self.mock_state = state_patch.start()

        # Configure test mocks.
        set_mock_state(self.mock_state)

        self.mock_state.compile_graph = TraceGraph()
        self.mock_state.serialized_graph = SerializedGraphV1()

        serialization.state = self.mock_state
        auto.state = self.mock_state
        model.state = self.mock_state

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
        @model.distributed_model()
        def test_model(x):
            return one_input_model(x)

        auto.state.tracking_model = False
        x = tf.placeholder(tf.float32, [])
        auto.state.compile_status = CompileStatus.STEP_COMPILE
        with auto.state.compile_graph.trace():
            x = tf.identity(x)
            output = test_model(x)

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 4, 2)

    def test_two_input_partition(self):
        @model.distributed_model()
        def test_model(x, y):
            return two_input_model(x, y)

        auto.state.tracking_model = False
        x = tf.placeholder(tf.float32, [])
        y = tf.placeholder(tf.float32, [])
        auto.state.compile_status = CompileStatus.STEP_COMPILE
        with auto.state.compile_graph.trace():
            x = tf.identity(x)
            y = tf.identity(y)
            output = test_model(x, y)

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 5, 2)

    def test_two_comm_partition(self):
        @model.distributed_model()
        def test_model(x, y):
            return two_comm_model(x, y)

        auto.state.tracking_model = False
        x = tf.placeholder(tf.float32, [])
        y = tf.placeholder(tf.float32, [])
        auto.state.compile_status = CompileStatus.STEP_COMPILE
        with auto.state.compile_graph.trace():
            x = tf.identity(x)
            y = tf.identity(y)
            output = test_model(x, y)

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 6, 3)

    def test_multi_consumer_partition(self):
        @model.distributed_model()
        def test_model(x, y):
            return multi_consumer_model(x, y)

        auto.state.tracking_model = False
        x = tf.placeholder(tf.float32, [])
        y = tf.placeholder(tf.float32, [])
        auto.state.compile_status = CompileStatus.STEP_COMPILE
        with auto.state.compile_graph.trace():
            x = tf.identity(x)
            y = tf.identity(y)
            output = test_model(x, y)

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 6, 3)

    def test_multi_model_input_consumer_partition(self):
        @model.distributed_model()
        def test_model(x, y):
            return multi_model_input_consumer_model(x, y)

        auto.state.tracking_model = False
        x = tf.placeholder(tf.float32, [])
        y = tf.placeholder(tf.float32, [])
        auto.state.compile_status = CompileStatus.STEP_COMPILE
        with auto.state.compile_graph.trace():
            x = tf.identity(x)
            y = tf.identity(y)
            output = test_model(x, y)

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 6, 3)

    def test_constant_return_partition(self):
        @model.distributed_model()
        def test_model(x):
            return constant_return_model(x)

        auto.state.tracking_model = False
        x = tf.placeholder(tf.float32, [])
        auto.state.compile_status = CompileStatus.STEP_COMPILE
        with auto.state.compile_graph.trace():
            x = tf.identity(x)
            output = test_model(x)

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 4, 3)

    def test_tensor_grouping(self):
        @model.distributed_model()
        def test_model(x):
            return tensor_grouping_model(x)

        auto.state.tracking_model = False
        x = tf.placeholder(tf.float32, [])
        auto.state.compile_status = CompileStatus.STEP_COMPILE
        with auto.state.compile_graph.trace():
            x = tf.identity(x)
            output = test_model(x)

        self.assert_has_correct_op_counts(self.mock_state.serialized_graph.partitioned_graphs, 5, 3)


if __name__ == "__main__":
    unittest.main()
