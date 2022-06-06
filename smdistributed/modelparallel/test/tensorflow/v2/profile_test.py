#!/usr/bin/env python3

# Standard Library
import time
from contextlib import contextmanager

# Third Party
import numpy as np
import smdebug.tensorflow as smd
import tensorflow as tf
from mock import MagicMock, patch
from smdebug.trials import create_trial
from tensorflow.keras.layers import Dense

# First Party
import smdistributed.modelparallel.tensorflow as smp
from smdistributed.modelparallel.backend.config import ModelParallelConfig
from smdistributed.modelparallel.tensorflow import graph_utils
from smdistributed.modelparallel.tensorflow.state_mod import TFModelParallelState
from smdistributed.modelparallel.tensorflow.utils import smdebug_name_to_layer_name
from smdistributed.modelparallel.tensorflow.v2 import model, serialization
from smdistributed.modelparallel.tensorflow.v2.serialization import SerializedGraphV2


class ProfileTests(tf.test.TestCase):
    """ Unit Tests for profiling tensor shapes using smdebugger """

    def setUp(self):
        super(ProfileTests, self).setUp()

        state_patch = patch(
            "smdistributed.modelparallel.tensorflow.v2.model.state", autospec=TFModelParallelState()
        )
        self.addCleanup(state_patch.stop)
        self.mock_state = state_patch.start()

        self.mock_state._profile_model = None
        self.mock_state.cfg = MagicMock(autospec=ModelParallelConfig)
        self.mock_state.serialized_graph = MagicMock(autospec=SerializedGraphV2)
        self.mock_state.serialized_graph.tensor_shapes = {}
        self.mock_state.serialized_graph.default_tensor_size = 1
        self.mock_state.serialized_graph.is_profiling = False
        self.mock_state.serialized_graph.raw_profiling_results = {}
        self.mock_state.serialized_graph.profiling_context = ProfileTests._profile_context

        model.state = self.mock_state
        serialization.state = self.mock_state
        graph_utils.state = self.mock_state

    @staticmethod
    @contextmanager
    def _profile_context(smd=None):
        user_hook = smd.get_hook("keras", create_if_not_exists=False)
        yield True
        smd.del_hook()
        if user_hook != None:
            smd.set_hook(user_hook)

    def assert_shape_matching(self, shape1, shape2):
        self.assertEqual(len(shape1), len(shape2))
        for key, val in shape1.items():
            for tensor_type in ["inputs", "outputs"]:
                self.assertEqual(shape1[key][tensor_type], shape2[key][tensor_type])

    def test_single_layer_single_input_right_shape(self):
        class MyModel(smp.DistributedModel):
            def __init__(self):
                super(MyModel, self).__init__()
                self.dense = Dense(10, activation="relu")

            def call(self, x):
                return self.dense(x)

        self.skipTest("Skip smdebug test. Please see: https://t.corp.amazon.com/P44514701")
        model.state.cfg.microbatches = 1
        serialization.state.cfg.pipeline_parallel_degree = 1
        serialization.state.serialized_graph.is_profiling = True
        my_model = MyModel()
        x_train = np.random.random((1000, 20))

        my_model.profile(x_train)

        shapes = {"dense": {"inputs": [[(1000.0, 20.0)]], "outputs": [[(1000.0, 10.0)]]}}
        self.assert_shape_matching(model.state.serialized_graph.raw_profiling_results, shapes)

    def test_multi_layer_single_input_right_shape(self):
        class MyModel(smp.DistributedModel):
            def __init__(self):
                super(MyModel, self).__init__()
                self.dense1 = Dense(10, activation="relu")
                self.dense2 = Dense(20, activation="relu")
                self.dense3 = Dense(30, activation="relu")
                self.dense4 = Dense(40, activation="relu")
                self.dense5 = Dense(50, activation="relu")

            def call(self, x):
                x = self.dense1(x)
                x = self.dense2(x)
                x = self.dense3(x)
                x = self.dense4(x)
                x = self.dense5(x)
                return x

        self.skipTest("Skip smdebug test. Please see: https://t.corp.amazon.com/P44514701")
        model.state.cfg.microbatches = 1
        serialization.state.cfg.pipeline_parallel_degree = 1
        serialization.state.serialized_graph.is_profiling = True
        my_model = MyModel()
        x_train = np.random.random((1000, 20))

        my_model.profile(x_train)

        shapes = {
            "dense": {"inputs": [[(1000.0, 20.0)]], "outputs": [[(1000.0, 10.0)]]},
            "dense_1": {"inputs": [[(1000.0, 10.0)]], "outputs": [[(1000.0, 20.0)]]},
            "dense_2": {"inputs": [[(1000.0, 20.0)]], "outputs": [[(1000.0, 30.0)]]},
            "dense_3": {"inputs": [[(1000.0, 30.0)]], "outputs": [[(1000.0, 40.0)]]},
            "dense_4": {"inputs": [[(1000.0, 40.0)]], "outputs": [[(1000.0, 50.0)]]},
        }

        self.assert_shape_matching(model.state.serialized_graph.raw_profiling_results, shapes)

    def test_smp_hook_with_user_hook(self):
        def _is_forward_tensor(name):
            # dmdebug forward tensor naming rules: *layername/inputs(outputs)
            return (
                ("inputs" in name or "outputs" in name)
                and "gradients" not in name
                and "weights" not in name
            )

        class MyModel(smp.DistributedModel):
            def __init__(self):
                super(MyModel, self).__init__()
                self.dense = Dense(10, activation="relu")

            def call(self, x):
                return self.dense(x)

        self.skipTest("Skip smdebug test. Please see: https://t.corp.amazon.com/P44514701")
        model.state.cfg.microbatches = 1
        serialization.state.cfg.pipeline_parallel_degree = 1
        model.state.serialized_graph.is_profiling = True

        my_model = MyModel()
        path = f"/tmp/smdebug_outputs{str(int(time.time()))}"
        user_hook = smd.KerasHook(
            path,
            save_all=True,
            save_config=smd.SaveConfig(save_steps=[0], save_interval=1),
            reduction_config=smd.ReductionConfig(save_shape=True, save_raw_tensor=True),
        )
        user_hook.register_model(my_model)

        x_train = np.random.random((1000, 20))
        y_train = np.random.random((1000, 1))

        my_model.profile(x_train)

        my_model.compile(optimizer="Adam", loss="mse", run_eagerly=True)
        my_model.fit(x_train, y_train, epochs=1, steps_per_epoch=1, callbacks=[user_hook])

        trial = create_trial(path=path, name="training_run")
        tensor_names = trial.tensor_names()
        shapes = {}
        for name in tensor_names:
            if not my_model._is_forward_tensor(name):
                continue
            shape = trial.tensor(name).shape(step_num=0)
            # Convert shape to a list of tuple for post-processing
            shape = list(shape) if (len(shape) > 0 and isinstance(shape[0], tuple)) else [shape]

            layer_name, tensor_type = smdebug_name_to_layer_name(name)
            if layer_name not in shapes:
                shapes[layer_name] = {"inputs": [], "outputs": []}
            shapes[layer_name][tensor_type].append(shape)

        self.assert_shape_matching(model.state.serialized_graph.raw_profiling_results, shapes)


if __name__ == "__main__":
    tf.test.main()
