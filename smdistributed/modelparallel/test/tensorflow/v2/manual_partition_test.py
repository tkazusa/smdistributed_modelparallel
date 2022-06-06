#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf
from mock import patch

# First Party
import smdistributed.modelparallel.tensorflow as smp
from smdistributed.modelparallel.backend.config import ModelParallelConfig
from smdistributed.modelparallel.tensorflow.auto import AutoPartitioner
from smdistributed.modelparallel.tensorflow.serialization import BaseSerializedGraph
from smdistributed.modelparallel.tensorflow.state_mod import state


class ManualPartitionTests(unittest.TestCase):
    """ Unit tests for the smp.partition Manual Partitioning API """

    def setUp(self):
        # Patch test dependencies (this is roughly equivalent to @patch).

        cfg_patch = patch(
            "smdistributed.modelparallel.tensorflow.utils.state.cfg", autospec=ModelParallelConfig
        )

        self.addCleanup(cfg_patch.stop)
        self.mock_cfg = cfg_patch.start()
        self.mock_cfg.pipeline_parallel_degree = 2
        self.mock_cfg.auto_partition = False

        serialized_patch = patch(
            "smdistributed.modelparallel.tensorflow.utils.state.serialized_graph",
            autospec=BaseSerializedGraph(),
        )

        self.addCleanup(serialized_patch.stop)
        self.mock_serialized = serialized_patch.start()

    def test_tensorflow_v2_api(self):
        @tf.function
        def add():
            with smp.partition(0):
                with smp.partition(1):
                    a = tf.constant(3.0)
                    b = tf.constant(4.5)

                x = tf.constant([10.0, 9.0, 8.0])
                y = tf.constant([2.0, 3.0, 4.0])
                z = x + y

        add()
        self.assertTrue(len(state.op_to_device) == 5)

        graph = add.get_concrete_function().graph

        for op in graph.get_operations():
            if op.name == "Const" or op.name == "Const_1":
                self.assertEqual(state.op_to_device[op.name], 1)
            else:
                self.assertEqual(state.op_to_device[op.name], 0)

    def test_get_layer_to_device(self):
        partition = AutoPartitioner()
        op_to_device = {"x": 0, "y": 1, "z": 0, "a": 0}
        op_to_layer = {
            "x": "layer_1",
            "y": "layer_4",
            "z": "layer_1",
            "d": "layer_3",
            "a": "layer_1",
        }

        result = partition.get_layer_to_device(op_to_layer, op_to_device, 0)
        expected_layer_to_device = {"layer_1": 0, "layer_4": 1, "layer_3": 0}

        for layer, device in expected_layer_to_device.items():
            self.assertEqual(result[layer], device)


if __name__ == "__main__":
    unittest.main()
