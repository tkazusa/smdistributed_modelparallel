#!/usr/bin/env python3

# Standard Library
import inspect

# Third Party
import tensorflow as tf
from mock import patch
from tensorflow.python.framework import test_util

# First Party
from smdistributed.modelparallel.tensorflow.split import StepOutput, TFTensorSplitter


class MockSplitResult(tf.Tensor):
    def __init__(self, tensor, mb, num_splits, axis):
        self._name = "dummyName"  # dummyName
        self._shape_val = (1, 1)  # dummyShape
        self._dtype = tf.float16  # dummyType
        self.source_tensor = tensor
        self.microbatch = mb
        self.num_splits = num_splits
        self.axis = axis


def mock_slice(tensor, num_splits, mb, axis=0):
    return MockSplitResult(tensor, mb, num_splits, axis)


# Disable running tests in eager execution for TF2, since preprocess_args is not expected to run
# in eager mode.
@test_util.for_all_test_methods(test_util.deprecated_graph_mode_only)
class TestPreprocessArgs(tf.test.TestCase):
    def test_validates_non_split_inputs(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, ["unknown_arg_name"], {})

        with self.assertRaises(ValueError):
            self.call_preprocess_args(
                splitter,
                args=[tf.zeros(10, 3), tf.zeros(10, 3), tf.zeros(10, 3)],
                kwargs={},
                num_microbatches=2,
                mb=0,
            )

    def test_validates_input_split_axes(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {"unknown_arg_name": 5})

        with self.assertRaises(ValueError):
            self.call_preprocess_args(
                splitter,
                args=[tf.zeros(10, 3), tf.zeros(10, 3), tf.zeros(10, 3)],
                kwargs={},
                num_microbatches=2,
                mb=0,
            )

    def test_splits_all_args_tensors_into_microbatches(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 2), tf.zeros(10, 4), tf.zeros(10, 6)],
            kwargs={},
        )

    def test_splits_all_kwargs_tensors_into_microbatches(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[],
            kwargs={"x": tf.zeros(10, 2), "y": tf.zeros(10, 4), "z": tf.zeros(10, 6)},
        )

    def test_non_split_input_on_arg(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, ["y"], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 4), tf.zeros(10, 5)],
            kwargs={"z": tf.zeros(10, 6)},
        )

    def test_non_split_input_on_arg_with_self(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, ["y"], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 4), tf.zeros(10, 5)],
            kwargs={"z": tf.zeros(10, 6)},
        )

    def test_non_split_input_on_kwarg(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, ["z"], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 4), tf.zeros(10, 5)],
            kwargs={"z": tf.zeros(10, 6)},
        )

    def test_correct_split_arguments(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {"x": 1, "y": 2, "z": 3})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 4), tf.zeros(10, 5)],
            kwargs={"z": tf.zeros(10, 6)},
        )

    def test_correct_split_arguments_many_microbatches(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {"x": 1, "y": 2, "z": 3})

        self.assert_correct_split(
            splitter,
            num_microbatches=10,
            args=[tf.zeros(10, 4), tf.zeros(10, 5)],
            kwargs={"z": tf.zeros(10, 6)},
        )

    def test_ignores_non_tensor_args(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {"x": 1, "y": 2, "z": 3})

        self.assert_correct_split(
            splitter, num_microbatches=2, args=["string", 1234], kwargs={"k": tf.zeros(10, 6)}
        )

    def test_model_outputs_are_unwrapped(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 4), StepOutput(outputs=[tf.zeros(10, 5), tf.zeros(11, 5)])],
            kwargs={
                "z": StepOutput(outputs=[tf.zeros(10, 6), tf.zeros(11, 6)]),
                "z2": tf.zeros(10, 2),
            },
        )

    def test_nested_tensor_args(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {"x": 1, "y": 2, "z": 3})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[[tf.zeros(10, 4), tf.zeros(10, 5)], tf.zeros(10, 4)],
            kwargs={"k": tf.zeros(10, 6)},
        )

    def test_nested_tensor_kwargs(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[],
            kwargs={
                "k": tf.zeros(10, 6),
                "l": [[tf.zeros(10, 4), tf.zeros(10, 5)], tf.zeros(10, 4)],
            },
        )

    def test_nested_tensor_args_and_kwargs(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, [], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 4), "string", {"random_key": tf.zeros(10, 5)}],
            kwargs={
                "k": tf.zeros(10, 6),
                "l": [[tf.zeros(10, 4), {"random_key": tf.zeros(10, 5)}], tf.zeros(10, 4)],
            },
        )

    def test_nested_tensor_args_non_split_inputs(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, ["y"], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 6), [tf.zeros(10, 4), tf.zeros(10, 5)]],
            kwargs={"z": tf.zeros(10, 6)},
        )

    def test_nested_tensor_with_model_outputs(self):
        splitter = TFTensorSplitter(lambda x, y, z: None, ["z"], {})

        self.assert_correct_split(
            splitter,
            num_microbatches=2,
            args=[tf.zeros(10, 4), tf.zeros(10, 5)],
            kwargs={"z": [tf.zeros(10, 4), StepOutput(outputs=[tf.zeros(10, 5), tf.zeros(11, 5)])]},
        )

    def assert_correct_split(self, splitter, num_microbatches, args, kwargs):
        def _assert_split_result(mb, output, parameter_name, parameter_index, parameter_value):
            """parameter_index is None only when a recursive call is made to the _assert_split_result.
            If the arg/kwarg is not nested, we do not need to recurse. However, if any arg/kwarg is nested, then, we
            recursively check the entire structure of that arg/kwarg using the same function. `parameter_index`
            behaves as an indicator that tells if we are currently processing a complicated arg/kwarg.
            If parameter_index is None, it means we are in a recursive call which is concerned with processing the
            nested args throughout a specific arg.
            If parameter_index is int, it indicates that this is the first call to the function for a specific parameter.
            """
            scrutiny_element = output
            if parameter_index is not None:
                scrutiny_element = output[parameter_index]
            if type(parameter_value) == tf.Tensor:
                if parameter_name not in splitter.non_split_inputs:
                    self.assertEqual(type(scrutiny_element), MockSplitResult)
                    self.assertEqual(scrutiny_element.source_tensor, parameter_value)
                    self.assertEqual(scrutiny_element.microbatch, mb)
                    self.assertEqual(scrutiny_element.num_splits, num_microbatches)
                    self.assertEqual(
                        scrutiny_element.axis, splitter.input_split_axes.get(parameter_name, 0)
                    )
                else:
                    self.assertEqual(type(scrutiny_element), type(parameter_value))
                    self.assertEqual(scrutiny_element, parameter_value)

            elif type(parameter_value) == StepOutput:
                self.assertEqual(type(scrutiny_element), tf.Tensor)
                self.assertEqual(scrutiny_element, parameter_value.outputs[mb])
            else:
                self.assertEqual(type(scrutiny_element), type(parameter_value))
                if isinstance(scrutiny_element, list) or isinstance(scrutiny_element, tuple):
                    self.assertEqual(len(scrutiny_element), len(parameter_value))
                    for indx, elem in enumerate(scrutiny_element):
                        _assert_split_result(mb, elem, parameter_name, None, parameter_value[indx])
                elif type(scrutiny_element) == dict:
                    self.assertEqual(set(scrutiny_element.keys()), set(parameter_value.keys()))
                    for _k in scrutiny_element.keys():
                        _assert_split_result(
                            mb, scrutiny_element[_k], parameter_name, None, parameter_value[_k]
                        )
                else:
                    self.assertEqual(scrutiny_element, parameter_value)

        parameter_names = inspect.getfullargspec(splitter.func).args
        for mb in range(num_microbatches):
            mb_args, mb_kwargs = self.call_preprocess_args(
                splitter, args, kwargs, num_microbatches, mb
            )

            for i, arg in enumerate(args):
                _assert_split_result(mb, mb_args, parameter_names[i], i, arg)

            for k, v in kwargs.items():
                _assert_split_result(mb, mb_kwargs, k, k, v)

    def call_preprocess_args(self, splitter, args, kwargs, num_microbatches, mb):
        splitter.slice = mock_slice
        with patch(
            "smdistributed.modelparallel.tensorflow.split.state.num_microbatches",
            lambda: num_microbatches,
        ):
            mb_args, mb_kwargs = splitter.preprocess_args(args, kwargs, num_microbatches, mb)
        return mb_args, mb_kwargs


if __name__ == "__main__":
    tf.test.main()
