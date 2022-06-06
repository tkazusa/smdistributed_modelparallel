#!/usr/bin/env python3

# Standard Library
import unittest
from unittest.mock import MagicMock

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel.backend.collectives import CollectiveCommunicator
from smdistributed.modelparallel.backend.config import ModelParallelConfig
from smdistributed.modelparallel.backend.core import ModelParallelCore
from smdistributed.modelparallel.tensorflow.compile import Compiler, CompileStatus, OpBehavior
from smdistributed.modelparallel.tensorflow.pipeline import TFPipeline
from smdistributed.modelparallel.tensorflow.state_mod import OpIdGenerator, TFModelParallelState
from smdistributed.modelparallel.tensorflow.utils import INPUT_OP, OUTPUT_OP, get_attr
from smdistributed.modelparallel.test.tensorflow.test_util import GraphBuilder


class TestCompiler(unittest.TestCase):
    """ Unit test for compiler. This verifies that the compiler produces correct output when given various input graphs.
    """

    def setUp(self):
        self.mock_config = MagicMock(autospec=ModelParallelConfig)
        self.mock_config.pipeline = "simple"

        self.mock_core = MagicMock(autospec=ModelParallelCore)
        self.mock_core.cfg = self.mock_config

        # Tests should set these and not rely on default values.
        self._pp_rank_to_rank_map = None
        self.set_ranks(-1, -1)
        self.mock_transaction_id = 0

        self.setup_mock_state()

        self.graph_builder = GraphBuilder()
        self.compiler = Compiler(self.mock_core, self.mock_state)

        def get_all_op_ids_mock():
            smp_ops = self.compiler.traverser.get_ops_of_type(
                ["SmpInput", "SmpOutput"], forward_only=False
            )
            fwd_ids = [get_attr(op, "op_id") for op in smp_ops]
            bwd_ids = [-id for id in fwd_ids]
            return fwd_ids + bwd_ids

        self.compiler.get_all_op_ids = get_all_op_ids_mock

    def setup_mock_state(self):
        self.mock_state = TFModelParallelState()
        self.mock_state.cfg = self.mock_config
        self.mock_state.core = self.mock_core
        self.mock_state.comm = MagicMock(autospec=CollectiveCommunicator)
        self.mock_state.pipeline = MagicMock(autospec=TFPipeline)
        self.mock_state.op_id_gen = MagicMock(autospec=OpIdGenerator)
        self.mock_state.compile_status = CompileStatus.STEP_COMPILE
        self.mock_state.get_transaction_id = lambda subgraph, count: self.mock_transaction_id
        self.mock_state.op_id_gen.transpose_op_id = lambda op_id, mb: op_id

        # Set bcast_from return value to unique string per input, so that calls can be verified.
        self.mock_state.comm.recv_from.side_effect = lambda pp_rank, transaction_id: self.get_bcast_from_mock_result(
            pp_rank, transaction_id
        )

    def set_ranks(self, pp_rank, global_rank=None):
        """ Set mock_core pp_rank and global rank.
        pp_rank is the model partition ID, global rank (also known as simply rank) is
        the unique rank ID amongst all nodes/devices.
        """
        # If global rank is omitted, use the pp_rank_to_rank mapping to obtain it.
        if global_rank == None:
            global_rank = self.pp_rank_to_rank_map[pp_rank]

        self.mock_core.rank.return_value = global_rank
        self.mock_core.pp_rank.return_value = pp_rank

    @property
    def graph(self):
        return self.mock_state.compile_graph

    @graph.setter
    def graph(self, graph):
        self.mock_state.compile_graph = graph

    @property
    def pp_rank_to_rank_map(self):
        return self._pp_rank_to_rank_map

    @pp_rank_to_rank_map.setter
    def pp_rank_to_rank_map(self, value):
        self._pp_rank_to_rank_map = value
        self.mock_core.pp_rank_to_rank = lambda pp_rank: self.pp_rank_to_rank_map[pp_rank]

    @staticmethod
    def get_bcast_from_mock_result(pp_rank, transaction_id):
        """ Deterministic generation of bcast_from mock result given pp_rank.
        This is useful when verifying bcast_from behavior.
        """
        from collections import defaultdict

        mock_result = defaultdict(str)
        mock_result[
            0
        ] = f"<bcast_from(pp_rank={pp_rank}, transaction_id={transaction_id}) mock result>"
        return mock_result

    def get_smp_op_for_vertex(self, edge_name, vertex):
        """ Get a SMP communication op for a given edge and vertex on the edge.
        The `source` vertex is the SMP output op on the edge, and the `sink`
        vertex is the SMP input op on the edge.
        """
        source_op, sink_op = self.graph.get_comm_ops_for_edge(edge_name)

        if vertex == "source":
            return source_op
        elif vertex == "sink":
            return sink_op
        else:
            raise ValueError(f'Vertex "{vertex}" is not "source" or "sink"')

    def assert_correct_link_pair(self):
        """Assert that SMP send/recv ops have unique and same link_id
        """

        def get_link_id(node):
            if isinstance(node, tf.Tensor):
                node = node.op
            if isinstance(node, tf.Operation):
                node = node.node_def
            op_id = node.attr["op_id"].i
            return self.compiler.op_attr[0][op_id][2]

        used_links = {}
        input_ops = self.compiler.traverser.get_ops_of_type(INPUT_OP, forward_only=False)
        for input_op in input_ops:
            output_ops = self.compiler.traverser.backward_walk(input_op, stopping_op_type=OUTPUT_OP)
            if len(output_ops) > 0:
                output_op = output_ops[0]
                src = self.mock_state.op_id_to_device[output_op.get_attr("op_id")]
                dest = self.mock_state.op_id_to_device[input_op.get_attr("op_id")]
                if src != dest:
                    input_link_id, output_link_id = get_link_id(input_op), get_link_id(output_op)
                    # Check the recv/send pair have the same link id
                    self.assertEqual(input_link_id, output_link_id)
                    # Check if the link id is valid
                    self.assertNotEqual(input_link_id, -1)
                    if input_link_id in used_links:
                        raise ValueError(
                            f"Link id should be unique for all send/recv op pairs. However, link id {input_link_id} is used for "
                            + f"both op pairs {[used_links[input_link_id][0].name, used_links[input_link_id][1].name]} and {[input_op.name, output_op.name]}"
                        )
                    used_links[input_link_id] = (input_op, output_op)

    def assert_vertex_has_correct_peer(self, edge_name, vertex):
        """ Assert that the SMP communication op on the specified vertex on a graph edge has
        correct peer.
        """
        op_under_test = self.get_smp_op_for_vertex(edge_name, vertex)

        def opposite_vertex(vertex):
            return "sink" if vertex == "source" else "source"

        other_op = self.get_smp_op_for_vertex(edge_name, opposite_vertex(vertex))
        expected_peer = self.mock_state.op_id_to_device[other_op.get_attr("op_id")]

        for op_id in [op_under_test.get_attr("op_id"), -op_under_test.get_attr("op_id")]:
            self.assertEqual(self.compiler.op_attr[0][op_id][1], expected_peer)

    def assert_vertex_has_output_peer(self, edge_name, vertex, output_pp_rank):
        """ Assert that the SMP communication op on the specified vertex on a graph edge has
        correct peer.
        """
        op_under_test = self.get_smp_op_for_vertex(edge_name, vertex)
        expected_peer = output_pp_rank
        op_id = op_under_test.get_attr("op_id")

        self.assertEqual(self.compiler.op_attr[0][op_id][1], expected_peer)
        self.assertEqual(self.compiler.op_attr[0][-op_id][1], -1)

    def assert_vertex_has_dummy_peer(self, edge_name, vertex):
        """ Assert that the SMP communication op on the specified vertex on a graph edge has
        dummy peer. That is to say that the op does not have a communication peer.
        """
        op_under_test = self.get_smp_op_for_vertex(edge_name, vertex)

        expected_peer = -1

        for op_id in [op_under_test.get_attr("op_id"), -op_under_test.get_attr("op_id")]:
            self.assertEqual(self.compiler.op_attr[0][op_id][1], expected_peer)

    def assert_vertex_has_correct_op_behavior(
        self, edge_name, vertex, forward_behavior, backward_behavior
    ):
        """ Assert that the SMP communication op on the specified vertex on a graph edge
        has the specified op behavior.
        """
        op_id = self.get_smp_op_for_vertex(edge_name, vertex).get_attr("op_id")
        self.assertEqual(self.compiler.op_behavior[0][op_id], forward_behavior)
        self.assertEqual(self.compiler.op_behavior[0][-op_id], backward_behavior)

    def assert_vertex_has_correct_op_metadata(self, edge_name, vertex, expected_op_metadata):
        """ Assert that the SMP communication op on the specified vertex on a graph edge
        has the specified op metadata.
        """
        op = self.get_smp_op_for_vertex(edge_name, vertex)
        self.assertEqual(self.compiler.op_metadata[0][op.get_attr("op_id")], expected_op_metadata)

    # def assert_correct_io_map(self, bcast_pp_rank, bcast_from_pp_ranks):
    #    """ Assert that the io_map is correct for the given managed and unmanaaged model partitions.
    #    Each process is expected to manage the io_map for a single model partition, which it broadcasts
    #    to other ranks. Each process will also read the broadcasted io_map.
    #    This function asserts that the process both sends and receives the io_map data correctly.
    #    """

    #    # Assert that bcast called was correctly, and that the data is correct.
    #    correct_value = [[[self.graph.get_io_map_value_for_pp_rank(bcast_pp_rank)]]]
    #    self.assertEqual(
    #        self.compiler.io_map[self.graph.get_io_map_key_for_pp_rank(bcast_pp_rank)],
    #        correct_value,
    #    )
    #    # self.mock_state.comm.bcast.assert_called_once_with(correct_value, self.mock_transaction_id)
    #    # Assert that bcast_from was called correcetly.
    #    for pp_rank in bcast_from_pp_ranks:
    #        self.assertEqual(
    #            self.compiler.io_map[self.graph.get_io_map_key_for_pp_rank(pp_rank)],
    #            self.get_bcast_from_mock_result(pp_rank, transaction_id=self.mock_transaction_id),
    #        )

    def print_smp_op_debug_info(self):
        """ Helper function that visualizes the SMP components of a graph.
        It is helpful when debugging failed test cases.
        """

        print("\npp_rank_to_rank", self.pp_rank_to_rank_map)

        for op in self.graph.get_operations():
            try:
                op_id = op.get_attr("op_id")
            except ValueError:
                continue

            print(
                op.name,
                f"(id {op_id}) ({self.compiler.op_behavior[0][op_id]}/{self.compiler.op_behavior[0][-op_id]})",
            )
            op_attr = self.compiler.op_attr[0][op_id]
            op_metadata = self.compiler.op_metadata[0][op_id]
            print(
                "\tfwd",
                "(tick: %d, peer: %d, link_id: %d)" % op_attr,
                "(shape=%s, dtype=%r)" % op_metadata,
            )
            op_attr = self.compiler.op_attr[0][-op_id]
            print("\tbwd", "(tick: %d, peer: %d, link_id: %d)" % op_attr)


class TestCompilerWithSimpleGraph(TestCompiler):
    """ Parent class for other classes that all test the same simple graph, on different device mappings.
    """

    def set_ranks_and_create_graph(self, pp_rank):
        """ Create the simple graph, deferring device mapping to subclasses.
        The graph is a simple sequential graph divided into three subgraphs.
        """

        self.setup_device_map()
        self.set_ranks(pp_rank)

        self.graph = (
            self.graph_builder.with_graph_inputs(["graph_input"])
            .with_fully_connected_subgraph(
                pp_rank=0, inputs=["graph_input"], outputs=["first_output"]
            )
            .with_fully_connected_subgraph(
                pp_rank=1, inputs=["first_output"], outputs=["second_output"]
            )
            .with_fully_connected_subgraph(
                pp_rank=2, inputs=["second_output"], outputs=["graph_output"]
            )
            .build(self.mock_state)
        )

    def setup_device_map(self):
        raise NotImplementedError("Subclasses must define a pp_rank->device mapping")


class TestCompilerWithSimpleGraphSimpleDeviceMap(TestCompilerWithSimpleGraph):
    """ Unit test for the compiler when given a simple input graph, with each subgraph assigned to a difference device.
    """

    def setup_device_map(self):
        """ Assign every subgraph to a different device.
        """
        self.pp_rank_to_rank_map = {0: 3, 1: 5, 2: 6}
        self.compiler.state.called_sg = {"partition_0", "partition_1", "partition_2"}

    def test_correct_op_behavior_when_rank_0(self):
        self.set_ranks_and_create_graph(pp_rank=0)
        self.compiler.compile(self.graph, 0, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_correct_op_behavior(
            "graph_input", "sink", OpBehavior.VALVE, OpBehavior.MARKER
        )
        self.assert_vertex_has_correct_op_behavior(
            "first_output", "source", OpBehavior.SEND, OpBehavior.RECV
        )

        self.assert_vertex_has_correct_op_behavior(
            "first_output", "sink", OpBehavior.DUMMY, OpBehavior.DUMMY
        )
        self.assert_vertex_has_correct_op_behavior(
            "second_output", "source", OpBehavior.DUMMY, OpBehavior.DUMMY
        )

        self.assert_vertex_has_correct_op_behavior(
            "second_output", "sink", OpBehavior.DUMMY, OpBehavior.DUMMY
        )
        self.assert_vertex_has_correct_op_behavior(
            "graph_output", "source", OpBehavior.OUTPUT_RECV, OpBehavior.DUMMY
        )

    def test_correct_op_behavior_when_rank_1(self):
        self.set_ranks_and_create_graph(pp_rank=1)
        self.compiler.compile(self.graph, 1, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_correct_op_behavior(
            "graph_input", "sink", OpBehavior.DUMMY, OpBehavior.DUMMY
        )
        self.assert_vertex_has_correct_op_behavior(
            "first_output", "source", OpBehavior.DUMMY, OpBehavior.DUMMY
        )

        self.assert_vertex_has_correct_op_behavior(
            "first_output", "sink", OpBehavior.RECV, OpBehavior.SEND
        )
        self.assert_vertex_has_correct_op_behavior(
            "second_output", "source", OpBehavior.SEND, OpBehavior.RECV
        )

        self.assert_vertex_has_correct_op_behavior(
            "second_output", "sink", OpBehavior.DUMMY, OpBehavior.DUMMY
        )
        self.assert_vertex_has_correct_op_behavior(
            "graph_output", "source", OpBehavior.OUTPUT_RECV, OpBehavior.DUMMY
        )

    def test_correct_op_behavior_when_rank_2(self):
        self.set_ranks_and_create_graph(pp_rank=2)
        self.compiler.compile(self.graph, 2, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_correct_op_behavior(
            "graph_input", "sink", OpBehavior.DUMMY, OpBehavior.DUMMY
        )
        self.assert_vertex_has_correct_op_behavior(
            "first_output", "source", OpBehavior.DUMMY, OpBehavior.DUMMY
        )

        self.assert_vertex_has_correct_op_behavior(
            "first_output", "sink", OpBehavior.DUMMY, OpBehavior.DUMMY
        )
        self.assert_vertex_has_correct_op_behavior(
            "second_output", "source", OpBehavior.DUMMY, OpBehavior.DUMMY
        )

        self.assert_vertex_has_correct_op_behavior(
            "second_output", "sink", OpBehavior.RECV, OpBehavior.SEND
        )
        self.assert_vertex_has_correct_op_behavior(
            "graph_output", "source", OpBehavior.BCAST, OpBehavior.VALVE
        )

    # def test_correct_io_map_when_rank_0(self):
    #    self.set_ranks_and_create_graph(pp_rank=0)
    #    self.compiler.compile(self.graph, 0, 0, self.compiler.state.op_id_to_device, False)
    #    self.assert_correct_io_map(bcast_pp_rank=0, bcast_from_pp_ranks=[1, 2])

    # def test_correct_io_map_when_rank_1(self):
    #    self.set_ranks_and_create_graph(pp_rank=1)
    #    self.compiler.compile(self.graph, 1, 0, self.compiler.state.op_id_to_device, False)
    #    self.assert_correct_io_map(bcast_pp_rank=1, bcast_from_pp_ranks=[0, 2])

    # def test_correct_io_map_when_rank_2(self):
    #    self.set_ranks_and_create_graph(pp_rank=2)
    #    self.compiler.compile(self.graph, 2, 0, self.compiler.state.op_id_to_device, False)
    #    self.assert_correct_io_map(bcast_pp_rank=2, bcast_from_pp_ranks=[0, 1])

    def test_correct_op_metadata_when_rank_0(self):
        self.set_ranks_and_create_graph(pp_rank=0)
        self.compiler.compile(self.graph, 0, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_correct_op_metadata(
            "graph_input", "sink", self.graph.non_dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "first_output", "source", self.graph.dummy_op_metadata
        )

        self.assert_vertex_has_correct_op_metadata(
            "first_output", "sink", self.graph.dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "second_output", "source", self.graph.dummy_op_metadata
        )

        self.assert_vertex_has_correct_op_metadata(
            "second_output", "sink", self.graph.dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "graph_output", "source", self.graph.non_dummy_op_metadata
        )

    def test_correct_op_metadata_when_rank_1(self):
        self.set_ranks_and_create_graph(pp_rank=1)
        self.compiler.compile(self.graph, 1, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_correct_op_metadata(
            "graph_input", "sink", self.graph.dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "first_output", "source", self.graph.dummy_op_metadata
        )

        self.assert_vertex_has_correct_op_metadata(
            "first_output", "sink", self.graph.non_dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "second_output", "source", self.graph.dummy_op_metadata
        )

        self.assert_vertex_has_correct_op_metadata(
            "second_output", "sink", self.graph.dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "graph_output", "source", self.graph.non_dummy_op_metadata
        )

    def test_correct_op_metadata_when_rank_2(self):
        self.set_ranks_and_create_graph(pp_rank=2)
        self.compiler.compile(self.graph, 2, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_correct_op_metadata(
            "graph_input", "sink", self.graph.dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "first_output", "source", self.graph.dummy_op_metadata
        )

        self.assert_vertex_has_correct_op_metadata(
            "first_output", "sink", self.graph.dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "second_output", "source", self.graph.dummy_op_metadata
        )

        self.assert_vertex_has_correct_op_metadata(
            "second_output", "sink", self.graph.non_dummy_op_metadata
        )
        self.assert_vertex_has_correct_op_metadata(
            "graph_output", "source", self.graph.non_dummy_op_metadata
        )

    def test_correct_op_attr_when_rank_0(self):
        self.set_ranks_and_create_graph(pp_rank=0)
        self.compiler.compile(self.graph, 0, 0, self.compiler.state.op_id_to_device, False)

        self.assert_correct_link_pair()

        self.assert_vertex_has_correct_peer("first_output", vertex="source")
        self.assert_vertex_has_dummy_peer("first_output", vertex="sink")
        self.assert_vertex_has_dummy_peer("second_output", vertex="source")
        self.assert_vertex_has_dummy_peer("second_output", vertex="sink")
        self.assert_vertex_has_output_peer("graph_output", vertex="source", output_pp_rank=2)

    def test_correct_op_attr_when_rank_1(self):
        self.set_ranks_and_create_graph(pp_rank=1)
        self.compiler.compile(self.graph, 1, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_dummy_peer("first_output", vertex="source")
        self.assert_vertex_has_correct_peer("first_output", vertex="sink")
        self.assert_vertex_has_correct_peer("second_output", vertex="source")
        self.assert_vertex_has_dummy_peer("second_output", vertex="sink")
        self.assert_vertex_has_output_peer("graph_output", vertex="source", output_pp_rank=2)

    def test_correct_op_attr_when_rank_2(self):
        self.set_ranks_and_create_graph(pp_rank=2)
        self.compiler.compile(self.graph, 2, 0, self.compiler.state.op_id_to_device, False)

        self.assert_vertex_has_dummy_peer("first_output", vertex="source")
        self.assert_vertex_has_dummy_peer("first_output", vertex="sink")
        self.assert_vertex_has_dummy_peer("second_output", vertex="source")
        self.assert_vertex_has_correct_peer("second_output", vertex="sink")
        self.assert_vertex_has_dummy_peer("graph_output", vertex="source")


if __name__ == "__main__":
    unittest.main()
