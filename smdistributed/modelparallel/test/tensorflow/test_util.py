# Standard Library
from typing import List, Mapping, NamedTuple
from unittest.mock import patch

# Third Party
import tensorflow as tf
from tensorflow.python.framework import ops

# First Party
from smdistributed.modelparallel.tensorflow.ops import input as smp_input
from smdistributed.modelparallel.tensorflow.ops import output


class GraphConstructionException(Exception):
    pass


class ExtendedGraph(ops.Graph):
    """A TensorFlow graph with additional attributes to help in unit testing.
    The GraphBuilder class populates the attributes at build time.
    """

    def __init__(self):
        super().__init__()
        self._edges_to_comm_ops = {}
        self.op_to_sg = {}
        self.sg_to_pp_rank = {}

    def get_io_map_key_for_pp_rank(self, pp_rank):
        return GraphBuilder.get_partition_name(pp_rank)

    def get_io_map_value_for_pp_rank(self, pp_rank):
        return GraphBuilder.get_io_map_key(pp_rank)

    def get_comm_ops_for_edge(self, edge_name):
        return self._edges_to_comm_ops[edge_name]

    @property
    def dummy_op_metadata(self):
        return ([], tf.float32)

    @property
    def non_dummy_op_metadata(self):
        return (GraphBuilder.get_mock_tensor_shape(), GraphBuilder.get_mock_tensor_dtype())


class GraphBuilder:
    """ Helper class that builds basic TensorFlow+SMP graphs with complex topologies.
    """

    def __init__(self):
        self._subgraphs = []
        self._graph_inputs = []
        # op_id must start at 1 because negative op_id's are used to indicate
        # backprop behavior.
        self.__next_op_id = 1
        self._state = None
        self._graph = None

    @property
    def _next_op_id(self):
        """ Helper property that generates a unique op_id for graph construction.
        """
        op_id = self.__next_op_id
        self.__next_op_id += 1
        return op_id

    class _Subgraph(NamedTuple):
        """ Helper class that defines a subgraph to be constructed.
        """

        pp_rank: int
        inputs: List[str]
        outputs: List[str]
        topology: Mapping[str, List[str]]

        @property
        def name(self):
            return GraphBuilder.get_partition_name(self.pp_rank)

        @staticmethod
        def create_fully_connected_topology(inputs, outputs):
            return {output: inputs for output in outputs}

        @staticmethod
        def create_inputs_outputs_from_topology(topology):
            inputs = [input_edge for input_edges in topology.values() for input_edge in input_edges]
            outputs = topology.keys()
            return list(set(inputs)), outputs

    @staticmethod
    def get_partition_name(pp_rank):
        return f"partition_{pp_rank}"

    @staticmethod
    def get_io_map_key(pp_rank):
        return f"<mock key for subgraph in pp_rank={pp_rank}>"

    @staticmethod
    def get_mock_tensor_shape():
        return [3, 3]

    @staticmethod
    def get_mock_tensor_dtype():
        return tf.float64

    @staticmethod
    def get_mock_control_input():
        return tf.constant(0.0)

    def with_fully_connected_subgraph(self, pp_rank, inputs, outputs):
        """ Add a subgraph with the specified input and output edges. The subgraph will
        have a fully connected topology; each input is fed to a tf.add_n op for each output.
        """
        topology = self._Subgraph.create_fully_connected_topology(inputs, outputs)
        return self._with_subgraph(pp_rank, inputs, outputs, topology)

    def with_subgraph(self, pp_rank, topology):
        """ Add a subgraph with the specified topology. The topology is a dict with keys specifying
        the output edges of the subgraph, and corresponding values specifying which input edges of the
        subgraph are fed to that particular output.
        """
        inputs, outputs = self._Subgraph.create_inputs_outputs_from_topology(topology)
        return self._with_subgraph(pp_rank, inputs, outputs, topology)

    def _with_subgraph(self, pp_rank, inputs, outputs, topology):
        self._subgraphs.append(self._Subgraph(pp_rank, inputs, outputs, topology))
        return self

    def with_graph_inputs(self, inputs):
        """ Define graph inputs, to be replaced with a tf.constant tensor.
        """
        self._graph_inputs += inputs
        return self

    def build(self, state):
        """ Construct basic TensorFlow graph for unit tests.
        Each subgraph has a simple tf.math.add_n op for all outputs. The inputs to each tf.math.add_n op
        is defined by the subgraph topology. Input and output edges are replaced with SMP input and
        output ops.
        """
        self._graph = ExtendedGraph()
        self._state = state

        with patch("smdistributed.modelparallel.tensorflow.state_mod.state", self._state):
            with patch(
                "smdistributed.modelparallel.tensorflow.core_mod.pp_rank", self._state.core.pp_rank
            ):
                with self._graph.as_default():
                    self._construct_graph()

        # Prevent builder from being built again, as it has state.
        self._subgraphs = []
        return self._graph

    def _construct_graph(self):
        """ Validate and construct graph.
        """
        self._validate_unique_pp_ranks()
        self._validate_input_edges()

        scope = self._ExecutionScope()
        for graph_input in self._graph_inputs:
            # Define input edges as simple tensor with no producer op.
            scope.define_edge(
                graph_input,
                tf.constant(
                    -1.0, dtype=self.get_mock_tensor_dtype(), shape=self.get_mock_tensor_shape()
                ),
                producer_op=None,
            )

        # Attempt to construct all subgraphs, iteratively adding output edges to scope.
        sg_to_create = [sg for sg in self._subgraphs]
        while len(sg_to_create) > 0:
            # Every subgraph with inputs that are a subset of the defined edges can be constructed.
            sg_can_be_created = [sg for sg in sg_to_create if set(sg.inputs) <= scope.defined_edges]

            # If there are no such subgraphs, raise exception to user.
            if len(sg_can_be_created) == 0:
                raise GraphConstructionException(
                    f"Subgraphs have unresolved input edges, unable to construct graph. Remaining subgraphs: {sg_to_create}. Defined edges: {scope.defined_edges}. Are the subgraphs cyclical?"
                )

            for sg in sg_can_be_created:
                self._construct_subgraph(sg, scope)
                sg_to_create.remove(sg)

        # Store edges_to_comm_ops in graph so tests can use it.
        self._graph._edges_to_comm_ops = scope.edges_to_comm_ops

    def _construct_subgraph(self, subgraph, scope):
        """ Construct a TensorFlow subgraph, using the topology defined by subgraph specification.
        Values are provided by scope.
        Inputs are first loaded via SMP input ops. Outputs are first sent to SMP output ops.
        Outputs of the graph are stored in scope.
        """
        self._state.subgraph_to_device[subgraph.name] = subgraph.pp_rank

        smp_input_tensors = {}

        # Create a SMP input op for every input edge.
        for input_name in subgraph.inputs:
            op_id = self._next_op_id
            self._state.op_id_to_device[op_id] = subgraph.pp_rank
            control_inputs = []
            input_tensor = scope.peek_edge(input_name)
            smp_input_tensors[input_name] = smp_input(
                scope.peek_edge(input_name),
                control_inputs,
                op_id,
                subgraph.name,
                name=f"{subgraph.name}_input_{input_name}",
            )

            # Mark edge as consumed so other ops cannot consume it.
            scope.consume_edge(input_name, smp_input_tensors[input_name].op)

        # Create a graph op and SMP output op for every output edge.
        for output_name in subgraph.outputs:
            # Construct the "graph" op, representing an arbitrary computation. The op takes as input the edges
            # defined for the output edge in subgraph.topology.
            graph_inputs = [
                smp_input_tensors[input_edge] for input_edge in subgraph.topology[output_name]
            ]
            graph_tensor = tf.math.add_n(
                graph_inputs, name=f"{subgraph.name}_graph_op_{output_name}"
            )

            op_id = self._next_op_id
            self._state.op_id_to_device[op_id] = subgraph.pp_rank
            # Create SMP output op.
            output_tensor = output(
                graph_tensor,
                self.get_mock_control_input(),
                op_id,
                subgraph.name,
                name=f"{subgraph.name}_output_{output_name}",
                output_shape=self.get_mock_tensor_shape(),
                output_dtype=self.get_mock_tensor_dtype(),
            )

            # Define edge so another op can consume it.
            scope.define_edge(output_name, output_tensor, output_tensor.op)

    class _ExecutionScope:
        """ Helper class that keeps track of which edges are consumed.
        An edge cannot be consumed or produced twice.
        Tests can use the output property `edges_to_comm_ops` to see which output and input ops interact between subgraphs.
        """

        def __init__(self):
            self._defined_tensors = {}
            self._tensor_operations = {}

        def define_edge(self, edge_name, edge_value, producer_op):
            if edge_name in self._tensor_operations:
                raise GraphConstructionException(
                    f"Tensor {edge_name} defined more than once, not allowed by GraphBuilder"
                )

            self._defined_tensors[edge_name] = edge_value
            self._tensor_operations[edge_name] = (producer_op, None)

        def peek_edge(self, edge_name):
            return self._defined_tensors[edge_name]

        def consume_edge(self, edge_name, consumer_op):
            already_consumed = self._tensor_operations[edge_name][1] != None

            if already_consumed:
                raise GraphConstructionException(
                    f"Tensor {edge_name} consumed more than once, not allowed by GraphBuilder"
                )

            self._tensor_operations[edge_name] = (
                self._tensor_operations[edge_name][0],
                consumer_op,
            )

            return self.peek_edge(edge_name)

        @property
        def defined_edges(self):
            return self._defined_tensors.keys()

        @property
        def edges_to_comm_ops(self):
            return self._tensor_operations

    def _validate_unique_pp_ranks(self):
        """ Validate that every subgraph has distinct pp_rank.
        """
        pp_ranks = set()
        for sg in self._subgraphs:
            if sg.pp_rank in pp_ranks:
                raise GraphConstructionException(
                    f"Multiple subgraphs have pp_rank={sg.pp_rank}, unable to build graph. {self._subgraphs}"
                )
            pp_ranks.add(sg.pp_rank)

    def _validate_input_edges(self):
        """ Validate all input edges are present.
        """
        set_of_all_inputs = set()
        defined_edges = set()

        for sg in self._subgraphs:
            defined_edges.update(sg.outputs)
            set_of_all_inputs.update(sg.inputs)

        missing_inputs = set_of_all_inputs - set(self._graph_inputs) - defined_edges
        if len(missing_inputs) > 0:
            raise GraphConstructionException(f"Undefined input edges: {list(missing_inputs)}")
