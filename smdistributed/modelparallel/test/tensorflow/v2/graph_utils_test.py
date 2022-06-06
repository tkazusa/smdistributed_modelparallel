#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel.tensorflow.pybind import inline_funclib


class GraphdefTests(unittest.TestCase):
    """ Unit Tests for manipulating Graphdef """

    def test_funclib_inlining_with_control_flow(self):
        @tf.function
        def model(x, y, c):
            for i in range(c):
                x = x + tf.constant(2.0)
            z = x + y
            return z

        test_graph_def = model.get_concrete_function(
            x=tf.TensorSpec(shape=[], dtype=tf.float32),
            y=tf.TensorSpec(shape=[], dtype=tf.float32),
            c=tf.TensorSpec(shape=[], dtype=tf.float32),
        ).graph.as_graph_def()

        output_graph_def = inline_funclib(test_graph_def)

        # checking if all the nodes in the input graphdef are present in output_graphdef
        input_nodedef_names = [node.name for node in test_graph_def.node]
        output_nodedef_names = [node.name for node in output_graph_def.node]
        for node_name in input_nodedef_names:
            self.assertTrue(node_name in output_nodedef_names)

        self.assertNotEqual(
            output_graph_def.SerializeToString(), test_graph_def.SerializeToString()
        )
        self.assertTrue(len(test_graph_def.library.function) != 0)

        # do not lower control flow
        self.assertTrue(
            len(output_graph_def.library.function) == len(test_graph_def.library.function)
        )

    def test_funclib_inlining_with_nested_function(self):
        @tf.function
        def model(x, y):
            return x + y + tf.constant(2.0)

        @tf.function
        def outer(a, b):
            return model(a, b * 1.2) - tf.constant(0.5)

        test_graph_def = outer.get_concrete_function(
            a=tf.TensorSpec(shape=[], dtype=tf.float32), b=tf.TensorSpec(shape=[], dtype=tf.float32)
        ).graph.as_graph_def()

        output_graph_def = inline_funclib(test_graph_def)

        self.assertNotEqual(
            output_graph_def.SerializeToString(), test_graph_def.SerializeToString()
        )
        self.assertTrue(len(test_graph_def.library.function) != 0)
        self.assertTrue(len(output_graph_def.library.function) == 0)
