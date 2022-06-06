#!/usr/bin/env python3

# Third Party
import tensorflow as tf
from mock import MagicMock

# First Party
from smdistributed.modelparallel.backend.config import ModelParallelConfig
from smdistributed.modelparallel.tensorflow.auto import AutoPartitioner, PartitionType
from smdistributed.modelparallel.tensorflow.state_mod import OpIdGenerator

NUM_MICROBATCHES = 2
PARTITIONS = 2


def set_mock_state(mock_state):
    # Configure test mocks.
    mock_state.num_microbatches.return_value = NUM_MICROBATCHES
    mock_state.partition_type = PartitionType.UNIT_TEST
    mock_state.compiler = MagicMock()
    mock_state.ctrl_mgr = MagicMock()
    mock_state.cfg = MagicMock(autospec=ModelParallelConfig)
    mock_state.cfg.pipeline_parallel_degree = PARTITIONS
    mock_state.partitioner = AutoPartitioner()
    mock_state.partitioner._maybe_partition_layers = lambda: None
    mock_state.op_id_gen = OpIdGenerator()


def one_input_model(x):
    with tf.name_scope("first"):
        x = tf.add(x, tf.constant(1.0), name="add")
    with tf.name_scope("second"):
        x = tf.multiply(x, tf.constant(3.0), name="mul")
    return x


def two_input_model(x, y):
    with tf.name_scope("first"):
        x = tf.add(x, tf.constant(1.0), name="add")
    with tf.name_scope("second"):
        x = tf.multiply(x, y, name="mul")
    return x


def two_comm_model(x, y):
    with tf.name_scope("first"):
        x = x + 1.0
        y = y * 2.0
    with tf.name_scope("second"):
        x = tf.multiply(x, y, name="mul")
    return x


def multi_consumer_model(x, y):
    with tf.name_scope("first"):
        x = x + 1.0
        y = y * 2.0
    with tf.name_scope("second"):
        x = tf.multiply(x, y, name="mul")
        z = y + 1.0
    return x + z


def multi_model_input_consumer_model(x, y):
    with tf.name_scope("first"):
        x = x + 1.0
        t = x * 2.0
    with tf.name_scope("second"):
        x = tf.multiply(x + t, y, name="mul")
        z = y + 1.0
    return x + z


def constant_return_model(x):
    with tf.name_scope("first"):
        x = x + 1.0
    with tf.name_scope("second"):
        z = x + 1.0
    return z, tf.constant(2.0)


def tensor_grouping_model(x):
    with tf.name_scope("first"):
        x = x + 1.0
        y = x + 2.0
    with tf.name_scope("second"):
        z = x + 1.0
        v = x + z + 1.0
        q = x + y + v
    return q
