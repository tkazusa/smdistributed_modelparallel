#!/usr/bin/env python3

# Third Party
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util

# First Party
from smdistributed.modelparallel.tensorflow.split import TFTensorSplitter, is_last_split_op

all_except = lambda arr, except_index: arr[:except_index] + arr[except_index + 1 :]

# Disable running tests in eager execution for TF2, since split_batch_tensor_uniformly is not expected to run
# in eager mode.
@test_util.for_all_test_methods(test_util.deprecated_graph_mode_only)
class TestSplitBatchTensorUniformly(tf.test.TestCase):
    def test_even_split_on_1d_tensor(self):
        self.assert_correct_split(shape=(12,), num_microbatches=6)

    def test_even_split_on_simple_tensor(self):
        self.assert_correct_split(shape=(12, 2), num_microbatches=6)

    def test_even_split_on_nontrivial_tensor(self):
        self.assert_correct_split(shape=(7, 6, 5, 4, 3, 2), num_microbatches=7)

    def test_split_on_undefined_shape(self):
        self.assert_correct_split(
            shape=(16, 3, 2), num_microbatches=4, static_shape=(None, None, None)
        )

    def test_custom_split_axis_even_split(self):
        self.assert_correct_split(shape=(2, 10), num_microbatches=5, split_axis=1)

    def assert_fails(self, shape, num_microbatches, static_shape=None, split_axis=0):
        if None in shape:
            raise ValueError("shape must be fully defined.")

        if static_shape == None:
            static_shape = shape

        with self.assertRaises(tf.errors.InvalidArgumentError):
            with self.cached_session() as sess:
                input_tensor = tf.compat.v1.placeholder(shape=static_shape, dtype=tf.int32)
                splitter = TFTensorSplitter(lambda x: None, [], {})
                output_tensors = [
                    splitter.slice(input_tensor, num_microbatches, mb, split_axis)
                    for mb in range(num_microbatches)
                ]

                split_shapes = [
                    t.shape for t in sess.run(output_tensors, {input_tensor: np.zeros(shape)})
                ]

    def assert_correct_split(self, shape, num_microbatches, static_shape=None, split_axis=0):
        if None in shape:
            raise ValueError("shape must be fully defined.")

        if static_shape == None:
            static_shape = shape

        with self.cached_session() as sess:
            input_tensor = tf.compat.v1.placeholder(shape=static_shape, dtype=tf.int32)
            splitter = TFTensorSplitter(lambda x: None, [], {})
            output_tensors = [
                splitter.slice(input_tensor, num_microbatches, mb, split_axis)
                for mb in range(num_microbatches)
            ]

            self.assert_correct_static_shapes(output_tensors, static_shape, split_axis)
            self.assert_output_tensors_is_last_split_op(output_tensors)

            split_shapes = [
                t.shape for t in sess.run(output_tensors, {input_tensor: np.zeros(shape)})
            ]
            self.assert_correct_computed_shapes(
                split_shapes,
                unsplit_shape=shape,
                expected_splits=num_microbatches,
                split_axis=split_axis,
            )

    def assert_correct_static_shapes(self, output_tensors, static_shape, split_axis):
        for index, tensor in enumerate(output_tensors):
            actual_static_shape = tensor.shape.as_list()

            self.assertEqual(
                tuple(all_except(actual_static_shape, split_axis)),
                tuple(all_except(static_shape, split_axis)),
                "Last split op should have same static shape as input, except on split dimension",
            )

            if static_shape[split_axis] == None:
                desired_split_size = None
            else:
                unsplit_size = static_shape[split_axis]
                num_splits = len(output_tensors)
                desired_split_size = (
                    int(index < unsplit_size % num_splits) + unsplit_size // num_splits
                )

            self.assertEqual(
                actual_static_shape[split_axis],
                desired_split_size,
                "Actual static split size should equal split size if input tensor has size defined or None otherwise",
            )

    def assert_output_tensors_is_last_split_op(self, output_tensors):
        for tensor in output_tensors:
            self.assertTrue(
                is_last_split_op(tensor.op),
                "is_last_split_op should return True for the last split op for each microbatch.",
            )

    def assert_correct_computed_shapes(
        self, split_shapes, unsplit_shape, expected_splits, split_axis
    ):
        for shape in split_shapes:
            self.assertEqual(all_except(shape, split_axis), all_except(unsplit_shape, split_axis))

        split_sizes = [shape[split_axis] for shape in split_shapes]
        self.assertEqual(
            len(split_sizes),
            expected_splits,
            "`slice` should split tensor into specified number of tensors.",
        )
        self.assertEqual(
            sum(split_sizes),
            unsplit_shape[split_axis],
            "The split tensors must add up to original tensor size",
        )

        expected_max_split_difference = (
            0 if (unsplit_shape[split_axis] % expected_splits == 0) else 1
        )
        self.assertEqual(
            max(split_sizes) - min(split_sizes),
            expected_max_split_difference,
            "The split tensors should be as uniform as possible in size",
        )


if __name__ == "__main__":
    tf.test.main()
