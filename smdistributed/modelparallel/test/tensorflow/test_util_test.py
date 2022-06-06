#!/usr/bin/env python3

# Standard Library
import unittest
from unittest.mock import MagicMock

# First Party
from smdistributed.modelparallel.tensorflow.state_mod import TFModelParallelState
from smdistributed.modelparallel.test.tensorflow.test_util import (
    GraphBuilder,
    GraphConstructionException,
)


class GraphBuilderTest(unittest.TestCase):
    """ Simple tests for the graph builder.
    """

    def test_success(self):
        # Expect no exception.
        (
            GraphBuilder()
            .with_graph_inputs(["graph_in"])
            .with_fully_connected_subgraph(
                0, inputs=["graph_in"], outputs=["intermediate1", "intermediate2"]
            )
            .with_fully_connected_subgraph(1, inputs=["intermediate1"], outputs=["relay1"])
            .with_fully_connected_subgraph(2, inputs=["intermediate2"], outputs=["relay2"])
            .with_fully_connected_subgraph(3, inputs=["relay1", "relay2"], outputs=["graph_out"])
            .build(MagicMock(autospec=TFModelParallelState))
        )

    def test_nontrivial_topology(self):
        graph = (
            GraphBuilder()
            .with_graph_inputs(["graph_in"])
            .with_fully_connected_subgraph(0, inputs=["graph_in"], outputs=["A", "B", "C"])
            .with_subgraph(1, topology={"X": ["C"], "Y": ["A", "B"], "Z": ["A", "B", "C"]})
            .with_fully_connected_subgraph(2, inputs=["X", "Y", "Z"], outputs=["graph_out"])
            .build(MagicMock(autospec=TFModelParallelState))
        )

        # Check partial connectivity.
        self.assert_op_has_inputs(graph, "partition_1_graph_op_X", ["partition_1_input_C:0"])
        self.assert_op_has_inputs(
            graph, "partition_1_graph_op_Y", ["partition_1_input_A:0", "partition_1_input_B:0"]
        )
        self.assert_op_has_inputs(
            graph,
            "partition_1_graph_op_Z",
            ["partition_1_input_A:0", "partition_1_input_B:0", "partition_1_input_C:0"],
        )

        # Check full connectivity.
        self.assert_op_has_inputs(
            graph,
            "partition_2_graph_op_graph_out",
            ["partition_2_input_X:0", "partition_2_input_Y:0", "partition_2_input_Z:0"],
        )

    def test_cyclical_failure(self):
        with self.assertRaises(GraphConstructionException):
            (
                GraphBuilder()
                .with_graph_inputs(["graph_in"])
                .with_fully_connected_subgraph(
                    0, inputs=["graph_in", "cycle"], outputs=["intermediate1", "intermediate2"]
                )
                .with_fully_connected_subgraph(
                    1, inputs=["intermediate1", "intermediate2"], outputs=["graph_out", "cycle"]
                )
                .build(MagicMock(autospec=TFModelParallelState))
            )

    def test_consume_twice_failure(self):
        with self.assertRaises(GraphConstructionException):
            (
                GraphBuilder()
                .with_graph_inputs(["graph_in"])
                .with_fully_connected_subgraph(0, inputs=["graph_in"], outputs=["intermediate1"])
                .with_fully_connected_subgraph(
                    1, inputs=["intermediate1"], outputs=["intermediate2"]
                )
                .with_fully_connected_subgraph(
                    2, inputs=["intermediate1", "intermediate2"], outputs=["graph_out"]
                )
                .build(MagicMock(autospec=TFModelParallelState))
            )

    def test_duplicate_pp_rank(self):
        with self.assertRaises(GraphConstructionException):
            (
                GraphBuilder()
                .with_graph_inputs(["graph_in"])
                .with_fully_connected_subgraph(
                    0, inputs=["graph_in"], outputs=["intermediate1", "intermediate2"]
                )
                .with_fully_connected_subgraph(
                    0, inputs=["intermediate1", "intermediate2"], outputs=["graph_out"]
                )
                .build(MagicMock(autospec=TFModelParallelState))
            )

    def test_undefined_input_edge(self):
        with self.assertRaises(GraphConstructionException):
            (
                GraphBuilder()
                .with_graph_inputs(["graph_in"])
                .with_fully_connected_subgraph(
                    0,
                    inputs=["graph_in", "undefined_input"],
                    outputs=["intermediate1", "intermediate2"],
                )
                .with_fully_connected_subgraph(
                    1, inputs=["intermediate1", "intermediate2"], outputs=["graph_out"]
                )
                .build(MagicMock(autospec=TFModelParallelState))
            )

    def test_multiple_output_edge_definitions(self):
        with self.assertRaises(GraphConstructionException):
            (
                GraphBuilder()
                .with_graph_inputs(["graph_in"])
                .with_fully_connected_subgraph(
                    0, inputs=["graph_in"], outputs=["intermediate1", "intermediate2", "common_out"]
                )
                .with_fully_connected_subgraph(
                    1,
                    inputs=["intermediate1", "intermediate2"],
                    outputs=["graph_out", "common_out"],
                )
                .build(MagicMock(autospec=TFModelParallelState))
            )

    def assert_op_has_inputs(self, graph, op_name, inputs):
        self.assertCountEqual(
            inputs,
            [input_tensor.name for input_tensor in graph.get_operation_by_name(op_name).inputs],
        )


if __name__ == "__main__":
    unittest.main()
