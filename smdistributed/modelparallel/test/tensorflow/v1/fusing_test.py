#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel.tensorflow import graph_def_editor as gde
from smdistributed.modelparallel.tensorflow.fusing import FusedGraph


class FusingTests(unittest.TestCase):
    def test_reused_variable_fusing(self):
        g = tf.compat.v1.Graph()
        with g.as_default():
            v = tf.Variable(0.1)
            x1 = tf.constant(1.2)
            x2 = tf.constant(1.4)
            y1 = x1 * v
            y2 = x2 * v

        graph = gde.Graph(g.as_graph_def())
        op_to_layer = {op.name: op.name for op in g.get_operations()}
        fused_g = FusedGraph(graph, op_to_layer)

        expected_fused_node = {
            "mul",
            "mul_1",
            "Variable",
            "Variable/Assign",
            "Variable/read",
            "Variable/initial_value",
        }

        # ensure that all consumers of the variable are fused into single node

        self.assertEqual(len(fused_g.fused_node_names), 3)

        var_fused_node_name = sorted(fused_g.fused_node_names)[2]
        self.assertEqual(
            fused_g.get_fused_node_by_name(var_fused_node_name).nodes, expected_fused_node
        )


if __name__ == "__main__":
    unittest.main()
