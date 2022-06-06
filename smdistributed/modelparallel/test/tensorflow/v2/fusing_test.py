#!/usr/bin/env python3

# Standard Library
import unittest

# Third Party
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model

# First Party
from smdistributed.modelparallel.tensorflow import graph_def_editor as gde
from smdistributed.modelparallel.tensorflow.fusing import FusedGraph


class FusingTests(unittest.TestCase):
    def get_graph(self, model):
        mod = model()

        @tf.function
        def f():
            return mod(tf.constant([[1.0, 1.4], [-2.3, 0.4]]))

        return f.get_concrete_function().graph.as_graph_def()

    def get_op_to_layer(self, graph, layers):
        op_to_layer = {}
        for op in graph.node:
            for layer in layers:
                if layer + "/" in op.name:
                    op_to_layer[op.name] = layer
                    break
                elif layer + "_1/" in op.name:
                    op_to_layer[op.name] = layer
                    break
                else:
                    op_to_layer[op.name] = op.name
        return op_to_layer

    def test_reused_variable_fusing(self):
        layers = ["reuse_var/dense0", "reuse_var/dense1"]

        class ReusedVarMod(Model):
            def __init__(self):
                super(ReusedVarMod, self).__init__()
                self.d1 = Dense(1, name=layers[0])
                self.d2 = Dense(1, name=layers[1])

            def call(self, x):
                x = self.d1(x)
                x = self.d2(x)
                x = x + self.d1.bias + self.d2.bias
                return x

        graph = self.get_graph(ReusedVarMod)
        gde_graph = gde.Graph(graph)
        op_to_layer = self.get_op_to_layer(graph, layers)
        fused_g = FusedGraph(gde_graph, op_to_layer)

        # assert that add ops have been fused with the layer, since they consume the same variable
        self.assertTrue("reused_var_mod/add" not in fused_g.fused_node_names)
        self.assertTrue("reused_var_mod/add_1" not in fused_g.fused_node_names)
        self.assertTrue("reused_var_mod/add" in fused_g.get_fused_node_by_name(layers[0]).nodes)
        self.assertTrue("reused_var_mod/add_1" in fused_g.get_fused_node_by_name(layers[1]).nodes)

    def test_reused_layer_fusing(self):
        layers = ["reuse_layer/first", "reuse_layer/second"]

        class ReusedLayerMod(Model):
            def __init__(self):
                super(ReusedLayerMod, self).__init__()
                self.d1 = Dense(2, name=layers[0])
                self.d2 = Dense(2, name=layers[1])

            def call(self, x):
                x = self.d1(x)
                x = self.d2(x)
                x = self.d1(x)
                return x

        graph = self.get_graph(ReusedLayerMod)
        gde_graph = gde.Graph(graph)
        op_to_layer = self.get_op_to_layer(graph, layers)
        fused_g = FusedGraph(gde_graph, op_to_layer)

        # assert that ops from both calls of the layer are fused into single FusedNode
        self.assertEqual(set(fused_g.fused_node_names), set(layers).union({"Const", "Identity"}))
        first_fused_node = fused_g.get_fused_node_by_name(layers[0])
        self.assertIn("reused_layer_mod/reuse_layer/first/MatMul", first_fused_node.nodes)
        # tf 2.3 assigns '_1' to the op name while tf < 2.3 assigns '_1' to the layer name
        potential_op_names = [
            "reused_layer_mod/reuse_layer/first_1/MatMul",
            "reused_layer_mod/reuse_layer/first/MatMul_1",
        ]
        self.assertTrue(any(x in first_fused_node.nodes for x in potential_op_names))


if __name__ == "__main__":
    unittest.main()
