#!/usr/bin/env python3

# Standard Library
from contextlib import contextmanager

# Third Party
import numpy as np
import tensorflow as tf
from mock import MagicMock, patch

# First Party
from smdistributed.modelparallel.backend.config import ModelParallelConfig
from smdistributed.modelparallel.tensorflow import graph_utils
from smdistributed.modelparallel.tensorflow.state_mod import TFModelParallelState
from smdistributed.modelparallel.tensorflow.v1 import model, profiling, serialization
from smdistributed.modelparallel.tensorflow.v1.serialization import SerializedGraphV1


class ProfileTests(tf.test.TestCase):
    """ Unit Tests for profiling tensor shapes using smdebugger """

    def setUp(self):
        super(ProfileTests, self).setUp()

        state_patch = patch(
            "smdistributed.modelparallel.tensorflow.v1.model.state", autospec=TFModelParallelState()
        )
        self.addCleanup(state_patch.stop)
        self.mock_state = state_patch.start()

        self.mock_state._profile_model = None
        self.mock_state.cfg = MagicMock(autospec=ModelParallelConfig)
        self.mock_state.serialized_graph = MagicMock(autospec=SerializedGraphV1)
        self.mock_state.serialized_graph.tensor_shapes = {}
        self.mock_state.serialized_graph.default_tensor_size = 1
        self.mock_state.serialized_graph.is_profiling = False
        self.mock_state.serialized_graph.raw_profiling_results = {}
        self.mock_state.serialized_graph.profiling_context = ProfileTests._profile_context

        model.state = self.mock_state
        serialization.state = self.mock_state
        graph_utils.state = self.mock_state
        profiling.state = self.mock_state

    @staticmethod
    @contextmanager
    def _profile_context(smd=None, graph=None):
        user_hook = smd.get_hook("keras", create_if_not_exists=False)
        yield True
        smd.del_hook()
        if user_hook != None:
            smd.set_hook(user_hook)

    def assert_shape_matching(self, shape1, shape2):
        self.assertEqual(len(shape1), len(shape2))
        for key, val in shape1.items():
            if "Placeholder" not in key:
                self.assertEqual(shape1[key], shape2[key])

    def test_input_with_placeholder_right_shape(self):
        @model.distributed_model()
        def test_model(x):
            return tf.layers.dense(x, 10, activation="relu")

        model.state.cfg.microbatches = 1
        serialization.state.cfg.pipeline_parallel_degree = 1
        serialization.state.serialized_graph.is_profiling = True

        x_train = np.random.random((1000, 20))
        ph = tf.placeholder(tf.float32, [1000, 20])

        profiling.profile(test_model, x_train, placeholders=[ph])

        shapes = {
            "dense/BiasAdd:0": [(1000, 10)],
            "dense/MatMul/Placeholder:0": [(1000, 20)],
            "dense/MatMul:0": [(1000, 10)],
            "dense/Relu:0": [(1000, 10)],
            "dense/bias/Assign:0": [(10,)],
            "dense/bias/Initializer/zeros:0": [(10,)],
            "dense/bias:0": [(10,)],
            "dense/kernel/Assign:0": [(20, 10)],
            "dense/kernel/Initializer/random_uniform/RandomUniform:0": [(20, 10)],
            "dense/kernel/Initializer/random_uniform/max:0": [(1,)],
            "dense/kernel/Initializer/random_uniform/min:0": [(1,)],
            "dense/kernel/Initializer/random_uniform/mul:0": [(20, 10)],
            "dense/kernel/Initializer/random_uniform/shape:0": [(2,)],
            "dense/kernel/Initializer/random_uniform/sub:0": [(1,)],
            "dense/kernel/Initializer/random_uniform:0": [(20, 10)],
            "dense/kernel:0": [(20, 10)],
        }
        self.assert_shape_matching(model.state.serialized_graph.raw_profiling_results, shapes)

    def test_input_with_tfdataset_right_shape(self):
        @model.distributed_model()
        def test_model(x):
            return tf.layers.dense(x, 10, activation="relu")

        model.state.cfg.microbatches = 1
        serialization.state.cfg.pipeline_parallel_degree = 1
        serialization.state.serialized_graph.is_profiling = True

        data = tf.data.Dataset.from_tensors(np.random.random((1000, 20)))
        iter = data.make_one_shot_iterator()
        x_train = iter.get_next()

        profiling.profile(test_model, x_train)

        shapes = {
            "dense/BiasAdd:0": [(1000, 10)],
            "dense/MatMul/Placeholder:0": [(1000, 20)],
            "dense/MatMul:0": [(1000, 10)],
            "dense/Relu:0": [(1000, 10)],
            "dense/bias/Assign:0": [(10,)],
            "dense/bias/Initializer/zeros:0": [(10,)],
            "dense/bias:0": [(10,)],
            "dense/kernel/Assign:0": [(20, 10)],
            "dense/kernel/Initializer/random_uniform/RandomUniform:0": [(20, 10)],
            "dense/kernel/Initializer/random_uniform/max:0": [(1,)],
            "dense/kernel/Initializer/random_uniform/min:0": [(1,)],
            "dense/kernel/Initializer/random_uniform/mul:0": [(20, 10)],
            "dense/kernel/Initializer/random_uniform/shape:0": [(2,)],
            "dense/kernel/Initializer/random_uniform/sub:0": [(1,)],
            "dense/kernel/Initializer/random_uniform:0": [(20, 10)],
            "dense/kernel:0": [(20, 10)],
        }
        self.assert_shape_matching(model.state.serialized_graph.raw_profiling_results, shapes)


if __name__ == "__main__":
    tf.test.main()
