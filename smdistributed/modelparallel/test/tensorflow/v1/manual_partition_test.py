#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf
from mock import patch

# First Party
import smdistributed.modelparallel.tensorflow.v1 as smp
from smdistributed.modelparallel.backend.config import ModelParallelConfig
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

    def test_tensorflow_v1_api(self):

        with smp.partition(0):
            with smp.partition(1):
                self.assertTrue(len(state.op_to_device) == 0)
                a = tf.constant(20)
                b = tf.constant(30)

                self.assertEqual(state.op_to_device["Const_5"], 1)
                self.assertEqual(state.op_to_device["Const_6"], 1)
                self.assertTrue(len(state.op_to_device) == 2)

            c = tf.constant([3.0, 3.0, 3.0])
            d = tf.constant([2.0, 2.0, 2.0])

            self.assertEqual(state.op_to_device["Const_7"], 0)
            self.assertEqual(state.op_to_device["Const_8"], 0)

            sum = tf.add(c, d)

            self.assertTrue(len(state.op_to_device) == 5)
            self.assertEqual(state.op_to_device["Add"], 0)


if __name__ == "__main__":
    unittest.main()
