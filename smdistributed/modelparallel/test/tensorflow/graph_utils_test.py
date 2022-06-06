#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel.tensorflow.fusing import FusedGraph
from smdistributed.modelparallel.tensorflow.pybind import inline_funclib
from smdistributed.modelparallel.tensorflow.utils import (
    ANY_OP,
    GraphTraverser,
    _get_major_tf_version,
    _get_minor_tf_version,
)


class GraphdefTests(unittest.TestCase):
    """ Unit Tests for manipulating Graphdef """

    def test_backward_walk_simple(self):
        g = tf.Graph()
        with g.as_default():
            x = tf.constant(1.2)
            y = tf.constant(1.6)
            z = x + y
            t = x - y
            w = tf.reduce_sum(z * t - tf.cast(4, tf.float32))

        target_ops = ["AddV2", "Sub"]
        stopping_ops = ["Const"]

        t = GraphTraverser(g)
        nodes = t.backward_walk([w.op], stopping_ops, target_ops)
        expected_output = {"add", "sub", "sub_1"}

        self.assertEqual({n.name for n in nodes}, expected_output)

    def test_backward_walk_same_target(self):
        g = tf.Graph()
        with g.as_default():
            x = tf.constant(1.2)
            y = tf.constant(1.6)
            z = x + y
            t = x - y
            w = tf.reduce_sum(z * t - tf.cast(4, tf.float32))

        stopping_ops = ["AddV2", "Sub"]
        target_ops = ["AddV2", "Sub"]

        t = GraphTraverser(g)
        nodes = t.backward_walk([w.op], stopping_ops, target_ops)
        expected_output = {"sub_1"}

        self.assertEqual({n.name for n in nodes}, expected_output)

    def test_backward_walk_no_stop(self):
        g = tf.Graph()
        with g.as_default():
            x = tf.constant(1.2)
            y = tf.constant(1.6)
            z = x + y
            t = x - y
            w = tf.reduce_sum(z * t - tf.cast(4, tf.float32))

        stopping_ops = []
        target_ops = ["Const"]

        t = GraphTraverser(g)
        nodes = t.backward_walk([w.op], stopping_ops, target_ops)

        tf_minor_version = _get_minor_tf_version()
        tf_major_version = _get_major_tf_version()

        if tf_major_version == 2 and tf_minor_version >= 4:
            expected_output = {"Const", "Const_1", "Rank", "range/start", "range/delta", "Cast/x"}
        else:
            expected_output = {"Const", "Const_1", "Const_2", "Cast/x"}

        self.assertEqual({n.name for n in nodes}, expected_output)

    def test_backward_walk_no_op(self):
        g = tf.Graph()
        with g.as_default():
            x = tf.constant(1.2)
            y = tf.constant(1.6)
            z = x + y
            t = x - y
            w = tf.reduce_sum(z * t - tf.cast(4, tf.float32))

        stopping_ops = []
        target_ops = ["Conv2D"]

        t = GraphTraverser(g)
        nodes = t.backward_walk([w.op], stopping_ops, target_ops)
        expected_output = set()

        self.assertEqual({n.name for n in nodes}, expected_output)

    def test_backward_walk_any_op(self):
        g = tf.Graph()
        with g.as_default():
            x = tf.constant(1.2)
            y = tf.constant(1.6)
            z = x + y
            t = x - y
            w = tf.reduce_sum(z * t - tf.cast(4, tf.float32))

        target_ops = ANY_OP
        stopping_ops = ["Const"]

        t = GraphTraverser(g)
        nodes = t.backward_walk([w.op], stopping_ops, target_ops)

        tf_minor_version = _get_minor_tf_version()
        tf_major_version = _get_major_tf_version()

        if tf_major_version == 2 and tf_minor_version >= 4:
            expected_output = {
                "Sum",
                "Rank",
                "range",
                "range/start",
                "range/delta",
                "sub_1",
                "Cast",
                "Cast/x",
                "mul",
                "sub",
                "Const_1",
                "Const",
                "add",
            }
        else:
            expected_output = {
                "Sum",
                "Const_2",
                "sub_1",
                "Cast",
                "Cast/x",
                "mul",
                "sub",
                "Const_1",
                "Const",
                "add",
            }

        self.assertEqual({n.name for n in nodes}, expected_output)

    def test_funclib_inlining_on_graphdef_with_no_funclib(self):
        @tf.function
        def model(x, y):
            z = x + y
            return z

        test_graph_def = model.get_concrete_function(
            x=tf.TensorSpec(shape=[], dtype=tf.float32), y=tf.TensorSpec(shape=[], dtype=tf.float32)
        ).graph.as_graph_def()

        output_graph_def = inline_funclib(test_graph_def)
        self.assertTrue(len(test_graph_def.library.function) == 0)
        self.assertEqual(output_graph_def.SerializeToString(), test_graph_def.SerializeToString())

    def test_external_colocation_removal_in_fused_graph(self):

        fused_graph = FusedGraph(None, None)
        fused_node_names = ["A", "B", "C", "D"]
        fused_graph.add_empty_nodes(fused_node_names)

        fused_node = fused_graph.get_fused_node_by_name("A")
        fused_node.add_node("a1")
        fused_node.add_node("a2")
        fused_node.add_node("a3")
        fused_node.add_node("a4")

        fused_node = fused_graph.get_fused_node_by_name("B")
        fused_node.add_node("b1")
        fused_node.add_node("b2")
        fused_node.add_node("b3")
        fused_node.add_node("b4")
        # fused_node.add_external_colocation_dependency('A')

        fused_node = fused_graph.get_fused_node_by_name("C")
        fused_node.add_node("c1")
        fused_node.add_node("c2")
        fused_node.add_node("c3")
        # fused_node.add_external_colocation_dependency('A')
        fused_node.add_external_colocation_dependency("D")

        fused_node = fused_graph.get_fused_node_by_name("D")
        fused_node.add_node("d1")
        fused_node.add_node("d2")
        fused_node.add_node("d3")
        fused_node.add_external_colocation_dependency("B")

        fused_graph.remove_colocation_dependencies()

        self.assertTrue(len(fused_graph.fused_nodes) == 2)
        self.assertTrue("A" in fused_graph.fused_node_names and "B" in fused_graph.fused_node_names)
        self.assertTrue(
            {"d3", "c3", "c1", "c2", "b1", "d2", "b2", "b3", "d1", "b4"}
            == fused_graph.nodes["B"].nodes
        )

    def test_external_colocation_removal_with_circular_dependency_in_fused_graph(self):

        fused_graph = FusedGraph(None, None)
        fused_node_names = ["A", "B", "C"]
        fused_graph.add_empty_nodes(fused_node_names)

        fused_node = fused_graph.get_fused_node_by_name("A")
        fused_node.add_node("a1")
        fused_node.add_node("a2")
        fused_node.add_node("a3")
        fused_node.add_node("a4")
        fused_node.add_external_colocation_dependency("B")

        fused_node = fused_graph.get_fused_node_by_name("B")
        fused_node.add_node("b1")
        fused_node.add_node("b2")
        fused_node.add_node("b3")
        fused_node.add_node("b4")
        fused_node.add_external_colocation_dependency("C")

        fused_node = fused_graph.get_fused_node_by_name("C")
        fused_node.add_node("c1")
        fused_node.add_node("c2")
        fused_node.add_node("c3")
        fused_node.add_external_colocation_dependency("A")

        fused_graph.remove_colocation_dependencies()

        self.assertTrue(len(fused_graph.fused_nodes) == 1)
        self.assertTrue("C" in fused_graph.fused_node_names)
        self.assertTrue(
            {"a3", "b4", "c1", "c3", "b3", "a1", "b2", "a4", "b1", "a2", "c2"}
            == fused_graph.nodes["C"].nodes
        )


if __name__ == "__main__":
    unittest.main()
