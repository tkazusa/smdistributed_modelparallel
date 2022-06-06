# Third Party
# Standard Library
import math

import numpy as np
import tensorflow as tf

# First Party
from smdistributed.modelparallel.tensorflow import state
from smdistributed.modelparallel.tensorflow.graph_utils import (
    get_num_elements_in_var,
    get_tensor_shape,
    get_tensor_size,
    is_var,
    is_var_consumer,
)
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.utils import dtype_to_bytes, set_to_sorted_list


def create_fused_graph(gde_graph, op_to_layer):
    """ Function that takes a gde graph and layer_name_scopes and returns a FusedGraph Object """
    return FusedGraph(gde_graph, op_to_layer)


class FusedNode:
    def __init__(self, fused_graph, name):
        # Fused graph this node is part of
        self._graph = fused_graph

        # Name of this FusedNode
        self._name = name

        # List of constituent gde node names
        self._nodes = set()

        # List of FusedNode names feeding into this FusedNode
        self._inputs = set()

        # List of FusedNode names of control inputs feeding into this FusedNode
        self._control_inputs = set()

        # List of FusedNode names that have control inputs outcoming from this FusedNode
        self._control_outputs = set()

        # List of FusedNode names this node feeds.
        self._outputs = set()

        # map of input FusedNode name : [tensor_names] coming into this fused node.
        self._external_input_tensors = {}

        # map of output FusedNode name : [tensor_names] coming out from this fused node.
        self._external_output_tensors = {}

        # The layer names that belong to this fused node (TF2 only)
        self._layer_names = []

        # List of FusedNodes this node has colocation dependency with.
        self._external_colocation_dependency = set()

        # The total variable size/count in this fused node
        self._var_size = None
        self._var_count = None

    @property
    def info(self):
        return {
            "name": self.name,
            "nodes": self.nodes,
            "data_inputs": self.inputs,
            "control_inputs": self.control_inputs,
            "outputs": self.outputs,
            "input_tensors: ": self.external_input_tensors,
            "output_tensors": self.external_output_tensors,
        }

    @property
    def nodes(self):
        """ return all the nodes in this FusedNode"""
        return self._nodes

    @property
    def inputs(self):
        """ return a list of outpur FusedNode this is connected to """
        return list(self._inputs)

    @property
    def control_outputs(self):
        """ return a list of control input node names"""
        return list(self._control_outputs)

    @property
    def control_inputs(self):
        """ return a list of control input node names"""
        return list(self._control_inputs)

    def add_inputs(self, name):
        """ add Fused node name to the input list """
        self._inputs.add(name)

    def add_control_inputs(self, name):
        """ add control input node name """
        self._control_inputs.add(name)

    def add_control_outputs(self, name):
        """ add control output node name """
        self._control_outputs.add(name)

    @property
    def outputs(self):
        """ return a list of input FusedNode feeding into this node"""
        return list(self._outputs)

    def add_outputs(self, name):
        """ add a fused node name  to output list"""
        self._outputs.add(name)

    @property
    def name(self):
        """ returns the name of the node """
        return self._name

    @property
    def colocation_dependencies(self):
        """ Returns a list of FusedNodes that are external colcoation dependency
            to this Fused Node
        """
        return list(self._external_colocation_dependency)

    def get_var_size(self, var_shapes=None, real_size=True):
        """Returns the total variable cost of this fused node. If we are optimizing memory, the cost is
        the variable size. Otherwise it is the number of tf.Variable objects."""
        if real_size and self._var_size != None:
            return self._var_size
        if not real_size and self._var_count != None:
            return self._var_count

        result = 0
        for node in self.nodes:
            gde_node = self._graph._gde_graph.get_node_by_name(node)
            if is_var(gde_node):
                if real_size:
                    var_tensor = gde_node.outputs[0]
                    if var_shapes is not None:
                        # TF2.x
                        result += get_num_elements_in_var(var_shapes[gde_node.name])
                    else:
                        # TF1.x
                        result += get_tensor_size(var_tensor)
                else:
                    result += 1

        if real_size:
            self._var_size = result
            return self._var_size
        else:
            self._var_count = result
            return self._var_count

    def add_external_colocation_dependency(self, name):
        """ Add a FusedNode name as  external colcoation dependency """
        self._external_colocation_dependency.add(name)

    def remove_from_colocation_dependency_list(self, name):
        """ Remove a FusedNode name from colocation dependency list """
        if name in self._external_colocation_dependency:
            self._external_colocation_dependency.remove(name)

    @property
    def external_input_tensors(self):
        """ Returns a list of tensors feeding into this FusedNode """
        return self._external_input_tensors

    def add_external_input_tensor(self, src_node_name, tensor_name):
        """ Add tensor_name that is feeding this FusedNode """
        if src_node_name not in self._external_input_tensors:
            self._external_input_tensors[src_node_name] = []
        self._external_input_tensors[src_node_name].append(tensor_name)

    @property
    def external_output_tensors(self):
        return self._external_output_tensors

    def add_external_output_tensor(self, dst_node_name, tensor_name):
        """ Add tensor_name that is going out of the FusedNode """
        if dst_node_name not in self._external_output_tensors:
            self._external_output_tensors[dst_node_name] = []
        self._external_output_tensors[dst_node_name].append(tensor_name)

    def add_node(self, node_name):
        """ Add a node to this FusedNode """
        self.nodes.add(node_name)

    def get_input_connection(self, fusedNode):
        """ return the internal node fusednode is connected to """

    def print(self):
        print(f"------ FusedNodeName: {self.name} -------")
        print(f"Nodes: {self.nodes}")
        print(f"Inputs: {self.inputs}")
        print(f"Outputs: {self.outputs}")
        print(f"external_colocation_dependency: {self.colocation_dependencies}")
        print(f"control_inputs: {self.control_inputs}")

    def fill_colocation_attr(self):
        """ Iterate through the gde graph and populate FusedNode with colocation dependencies """
        gde_graph = self._graph._gde_graph

        # Case 1: Fuse nodes based on TF colocation groups
        for node_name in self._nodes:
            gde_node = gde_graph.get_node_by_name(node_name)
            colocation_grp_names = gde_node.colocation_groups
            for coloc_node_name in colocation_grp_names:
                if coloc_node_name not in self._nodes:
                    # getting the fused node this node is part of:
                    fused_node_name = self._graph.get_fused_node_name_by_gde_node_name(
                        coloc_node_name
                    )
                    self.add_external_colocation_dependency(fused_node_name)

        # Case 2: Fuse variables with the nodes that consume them
        # Case 3: Fuse variable consumers with the nodes that consume them
        # Case 4: Fuse nodes that exchange tf.string or tf.variant tensors
        incoming_tensor_names = self.get_incoming_tensor_names()
        for node_name in self._nodes:
            for tensor_name in incoming_tensor_names:
                tensor = gde_graph.get_tensor_by_name(tensor_name)
                src_node = tensor.node
                if (
                    is_var(src_node)
                    or is_var_consumer(src_node)
                    or tensor.dtype in [tf.string, tf.variant]
                ):
                    fused_node_name = self._graph.get_fused_node_name_by_gde_node_name(
                        src_node.name
                    )
                    self.add_external_colocation_dependency(fused_node_name)

    def get_incoming_tensor_names(self):
        """ Returns a list of incoming tensor names to this FusedNode """
        incoming_tensor_names = set()
        gde_graph = self._graph._gde_graph
        for node_name in self._nodes:
            gde_node = gde_graph.get_node_by_name(node_name)
            input_tensors = list(gde_node.inputs)
            # input_nodes_names = [ tensor.node.name for tensor in input_tensors]
            for tensor in input_tensors:
                if tensor.node.name not in self._nodes:
                    incoming_tensor_names.add(tensor.name)

        return list(incoming_tensor_names)

    def get_outgoing_tensor_names(self):
        """ Return a list of outgoing tensor names from this FusedNode """
        outgoing_tensor_names = set()
        gde_graph = self._graph._gde_graph
        for node_name in self._nodes:
            gde_node = gde_graph.get_node_by_name(node_name)
            output_tensors = list(gde_node.outputs)
            for tensor in output_tensors:
                consumers = tensor.consumers()
                for consumer in consumers:
                    if consumer.name not in self._nodes:
                        outgoing_tensor_names.add(tensor.name)

        return list(outgoing_tensor_names)

    def fill_fused_node_info(self):
        """ fill all the Fused Node Info """
        gde_graph = self._graph._gde_graph

        # Finding FusedNodes that are feeding into this FusedNode
        for tensor_name in self.get_incoming_tensor_names():
            src_node_name = gde_graph.get_tensor_by_name(tensor_name).node.name
            src_fused_node_name = self._graph.get_fused_node_name_by_gde_node_name(src_node_name)
            self.add_inputs(src_fused_node_name)
            self.add_external_input_tensor(src_fused_node_name, tensor_name)

        # Getting the FusedNodes that has control inputs outcoming from this FusedNode
        for gde_node in self._graph._gde_graph.nodes:
            control_inputs = list(gde_node.control_inputs)
            for node in control_inputs:
                if node.name in self._nodes and gde_node.name not in self._nodes:
                    self.add_control_outputs(
                        self._graph.get_fused_node_name_by_gde_node_name(gde_node.name)
                    )

        # Getting control inputs that are incoming to this FusedNode
        for gde_node_name in self._nodes:
            gde_node = gde_graph.get_node_by_name(gde_node_name)
            control_inputs = list(gde_node.control_inputs)
            for node in control_inputs:
                if node.name not in self._nodes:
                    self.add_control_inputs(
                        self._graph.get_fused_node_name_by_gde_node_name(node.name)
                    )

        # Finding FusedNodes that are being feed by this FusedNode
        for tensor_name in self.get_outgoing_tensor_names():
            # adding fusedNode name (corresponding to consumer name) to the output of this node
            out_tensor = gde_graph.get_tensor_by_name(tensor_name)
            dst_node_names = out_tensor.consumers()
            for consumer in dst_node_names:
                consumer_fused_name = self._graph.get_fused_node_name_by_gde_node_name(
                    consumer.name
                )
                self.add_external_output_tensor(consumer_fused_name, tensor_name)
                if consumer.name not in self._nodes:
                    self.add_outputs(consumer_fused_name)

        # If this FusedNode does'nt take any inputs, marking it as input
        # node of the FusedGraph
        if not self.inputs:
            self._graph.add_inputs(self.name)

        # if there are no consumers of outputs of this FusedNode, marking it
        # as output nodes of the FusedGraph
        if not self.outputs:
            self._graph.add_outputs(self.name)


class FusedGraph:
    def __init__(self, gde_graph, op_to_layer, fused_node_name_to_device=None):
        # Map of Node name -> FusedNode Object
        self._nodes = {}

        # List of FusedNodes that contains input nodes of the graph
        self._inputs = set()

        # List of FusedNodes that contain the output nodes of the graph
        self._outputs = set()

        self._gde_graph = gde_graph
        self._op_to_layer = op_to_layer
        self._node_name_to_fused_node = {}
        self._layer_name_to_fused_node_name = {}
        self._fused_node_name_to_device = fused_node_name_to_device

        # Create Fused Graph only when these are present.
        # This check is mostly for writing unit tests where we can create dummy FusedGraph/FusedNodes.
        if gde_graph and op_to_layer != None:
            sorted_nodes = sorted(set(op_to_layer.values()))
            self.create_fused_graph(sorted_nodes)

    @property
    def fused_node_name_to_device(self):
        """ return fused node name to device mapping """
        return self._fused_node_name_to_device

    @property
    def nodes(self):
        """  Return map of fused_node_name to fused node object in the graph """
        return self._nodes

    @property
    def fused_nodes(self):
        """ return a list of FusedNodes in the graph """
        return [self._nodes[name] for name in self.fused_node_names]

    @property
    def fused_node_names(self):
        """ return name of all the fused nodes in the graph """
        # Sort it so that everytime it returns the same sequence
        # Required for auto-partition
        return sorted(list(self._nodes.keys()))

    @property
    def inputs(self):
        """ return list of input segments """
        return self._inputs

    @property
    def outputs(self):
        """ return list of output segments """
        return self._outputs

    def add_inputs(self, name):
        """ Add Fused node name that are inputs to this Fused Graph"""
        self._inputs.add(name)

    def add_outputs(self, name):
        """ Add Fused Node that are output to the Fused Graph """
        self._outputs.add(name)

    def add_empty_nodes(self, fused_node_names):
        """ Create FusedNodes with node_names """
        for fname in fused_node_names:
            fused_node = FusedNode(self, fname)
            fused_node._layer_names.append(fname)
            self.add_node(fused_node)

    def update_node_name_to_fused_node(self, node_name, fused_node):
        self._node_name_to_fused_node[node_name] = fused_node

    def add_node(self, fused_node):
        """ Add a node to the Graph """
        self._nodes[fused_node.name] = fused_node

    def get_fused_node_for_node(self, name):
        """ Given a node name return the FusedNode this node should belong to
            Returns None, if no match is found
        """
        if name in self._op_to_layer:
            return self._nodes[self._op_to_layer[name]]
        else:
            return None

    def get_fused_node_by_name(self, name):
        """Returns the fused node object given the fused node name """
        return self._nodes[name]

    def get_fused_node_name_by_gde_node_name(self, name):
        """ Returns the fused node name to which this node belongs"""
        return self._node_name_to_fused_node[name].name

    def delete_node(self, name):
        """ Delete fused node from the graph """
        fused_node = self.get_fused_node_by_name(name)
        del self._nodes[name]
        del fused_node

    def print(self):
        print(f"Fused Node names: ", self.fused_node_names)
        print(f"Graph Inputs: ", self.inputs)
        print(f"Graph Outputs: ", self.outputs)

        for fused_node in self.fused_nodes:
            fused_node.print()

    def get_connected_components(self):
        visited = set()
        components = []
        for node in self.fused_nodes:
            if node.name not in visited:
                stack = [node.name]
                component = []
                while len(stack) > 0:
                    name = stack.pop()
                    if name not in visited:
                        visited.add(name)
                        node = self._nodes[name]
                        component.append(name)
                        stack.extend(node.inputs + node.control_inputs)
                        stack.extend(node.outputs + node.control_outputs)
                components.append(component)
        return components

    def layer_name_to_fused_node_name(self, name):
        """Get the fused node that has the layer name"""
        if name in self._layer_name_to_fused_node_name:
            return self._layer_name_to_fused_node_name[name]

        for node in self._nodes.values():  # _nodes.values() are fused nodes
            for layer_name in node._layer_names:
                self._layer_name_to_fused_node_name[layer_name] = node.name
        return self._layer_name_to_fused_node_name[name]

    def convert_to_metis_graph(self, fused_node_names, var_shapes=None, compute_cons=True):
        """Converting a fused graph into the CSR graph format
           For the details of CSR graph format please check:
           http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf
           Chapter 5.5"""
        nodes = [self._nodes[name] for name in fused_node_names]
        nvtxs = len(nodes)
        node_name_to_idx = {}
        for idx, name in enumerate(fused_node_names):
            node_name_to_idx[name] = idx
        xadj, adjncy, adjwgt, vwgt = [], [], [], []
        acc_edge_idx = 0
        ncon = 2 if compute_cons else 1
        # (TODO) Get the real compute_intensity for balancing constraint
        # (TODO) Get the real edge weight from SM debugger
        dummy_compute_intensity = 1
        var_size_scale_factor = 1.0
        if state.cfg.optimize == "memory":
            var_size_scale_factor = self.get_metis_scale_factor(nodes, var_shapes)
        get_logger().debug(
            f"Applying a var size scale factor {var_size_scale_factor} to normalize metis balancing constraint"
        )
        for node in nodes:
            xadj.append(acc_edge_idx)
            # if optimize == "speed", each tf.Variable object counts as 1
            node_var_size = node.get_var_size(
                var_shapes=var_shapes, real_size=(state.cfg.optimize == "memory")
            )
            if state.cfg.optimize == "memory":
                node_var_size = round(node_var_size * var_size_scale_factor)
            vwgt.append(node_var_size)
            if compute_cons:
                vwgt.append(dummy_compute_intensity)
            out_coming_nodes = set()
            out_coming_nodes = out_coming_nodes.union(node._inputs, node._outputs)
            for out_coming_node in set_to_sorted_list(out_coming_nodes):
                adjncy.append(node_name_to_idx[out_coming_node])
                if out_coming_node not in node._control_inputs.union(node._control_outputs):
                    adjwgt.append(
                        self.get_edge_weight(node, out_coming_node, out_coming_node in node._inputs)
                    )
                else:
                    adjwgt.append(1)
                acc_edge_idx += 1

        # post process edge weights if using tensor shapes as edge weights (if profiling with SM debugger)
        if state.serialized_graph.default_tensor_size != 1:
            get_logger().info(f"Using logarithmic edge weights for enhanced metis partitioning.")
            adjwgt = [round(max(math.log10(w + 1), 1)) for w in adjwgt]
        xadj.append(acc_edge_idx)
        num_edges = len(adjncy) // 2
        return nvtxs, num_edges, xadj, adjncy, adjwgt, vwgt, ncon

    def get_metis_scale_factor(self, nodes, var_shapes, max_deviation=2):
        """Calculate metis scale factor for variable sizes to be used as metis balancing constraint"""
        var_sizes = [
            node.get_var_size(var_shapes=var_shapes, real_size=(state.cfg.optimize == "memory"))
            for node in nodes
        ]
        # remove outliers and compute mean
        data = np.array(var_sizes)
        data = data[abs(data - np.mean(data)) < max_deviation * np.std(data)]
        metis_scale_factor = np.mean(data).item()
        # The var_size balancing constraint needs to be int after scaling. Hence adding a factor of 10.0 to ensure
        # we do not get all zeros for values less than mean.
        if metis_scale_factor > 1.0:
            return 10.0 / metis_scale_factor
        return 1.0

    def get_communication_volume_between_nodes(self, from_node, to_node, unit="MB"):
        """Calculate the tensor size send from from_node to to_node"""
        if to_node.name not in from_node.external_output_tensors:
            return 0
        output_size = 0
        for output_tensor_name in from_node.external_output_tensors[to_node.name]:
            dtype = self._gde_graph.get_tensor_by_name(output_tensor_name).dtype
            tensor_size = self._get_tensor_size_from_name(output_tensor_name)
            real_size = tensor_size * dtype_to_bytes(dtype)
            output_size += real_size
        if unit == "MB":
            return output_size / 1e6
        elif unit == "GB":
            return output_size / 1e9
        else:
            raise ValueError(f"Unsupported unit {unit}")

    def _get_tensor_size_from_name(self, tensor_name):
        tensor = self._gde_graph.get_tensor_by_name(tensor_name)
        tensor_shape = get_tensor_shape(tensor, self._op_to_layer)
        return get_tensor_size(tensor_shape)

    def get_edge_weight(self, src_node, dst_node_name, is_input):
        # Put all edge with 1 if there is no profiling
        if state.serialized_graph.default_tensor_size == 1:
            return 1

        tensor_map = (
            src_node._external_input_tensors if is_input else src_node._external_output_tensors
        )

        if dst_node_name not in tensor_map:
            raise ValueError(
                f"{dst_node_name} is not an {'input' if is_input else 'output'} of fused node {src_node.name}"
            )
        tensor_size = sum(
            [self._get_tensor_size_from_name(name) for name in tensor_map[dst_node_name]]
        )
        return tensor_size

    def update_external_colocation_attr(self):
        """ Fill colocation dependency for each FusedNode in FusedGraph"""
        for fused_node in self.fused_nodes:
            fused_node.fill_colocation_attr()

    def handle_external_colocation(self):
        """ Finds external colocation dependency of each FusedNode and
            Merges two FusedNodes if they have colocation dependency
        """
        self.update_external_colocation_attr()
        self.remove_colocation_dependencies()

    def remove_colocation_dependencies(self):
        """
            Remove colocation dependencies between FusedNodes.
            Using UnionFind algo to figure out disjoint components.
            Two components/FusedNodes are connected if they share
            colocation dependency.
            Using UnionFind to find common parent and merge them if needed.
        """
        # list of tuple of edges ( node_names ), created from colocation_dependencies
        edge_list = []
        fused_node_name_list = []
        for fused_node in self.fused_nodes:
            fused_node_name_list.append(fused_node.name)
            for node_name in fused_node.colocation_dependencies:
                edge_list.append((fused_node.name, node_name))

        if edge_list:
            from smdistributed.modelparallel.tensorflow.utils import UnionFind

            # UnionFind to do the unify.
            union_find = UnionFind(len(fused_node_name_list))
            for t in edge_list:
                union_find.unify(fused_node_name_list.index(t[0]), fused_node_name_list.index(t[1]))

            merge_map = {}  # map of source to destination.
            for idx, name in enumerate(fused_node_name_list):
                dest = union_find.find_parent(idx)
                if idx != dest:
                    merge_map[fused_node_name_list[idx]] = fused_node_name_list[dest]

            # merging nodes, merge k to v
            for k, v in merge_map.items():
                source = self._nodes[k]  # source is a FusedNode object
                dest = self._nodes[v]  # dest is a FusedNode object
                # copy src to dest
                for val in source.nodes:
                    dest.nodes.add(val)

                dest._layer_names.extend(source._layer_names)

                # Remove src as colocation dependency from dest.
                dest.remove_from_colocation_dependency_list(source.name)

                # update node_name_to_fusedNode for all the nodes in source.
                for node_name in source.nodes:
                    self.update_node_name_to_fused_node(node_name, dest)

            # delete the key nodes in merge_map from graph._nodes
            for node_name in merge_map.keys():
                # reassign device to node that is being fused to
                if self._fused_node_name_to_device:
                    self.handle_fused_node_name_to_device_conflicts(node_name, merge_map)
                self.delete_node(node_name)

    def handle_fused_node_name_to_device_conflicts(self, node_name, merge_map):
        """ If a node that exists in the mapping is being merged to a node that does not exist in the mapping,
        add the destination node to the device mapping with the source node's device. If both nodes exist in the
        mapping and they are mapped to different devices return a warning.
        """
        if node_name in self._fused_node_name_to_device:
            if merge_map[node_name] in self._fused_node_name_to_device:
                if (
                    self._fused_node_name_to_device[merge_map[node_name]]
                    != self._fused_node_name_to_device[node_name]
                ):
                    get_logger().warning(
                        "Violating user device specification: Any ops defined in %s are being assigned to device %d"
                        % (node_name, self._fused_node_name_to_device[merge_map[node_name]])
                    )
            else:
                self._fused_node_name_to_device[
                    merge_map[node_name]
                ] = self._fused_node_name_to_device[node_name]

    def create_fused_graph(self, fused_node_names):
        """ Fills all the details in FusedGraph object """
        self.add_empty_nodes(fused_node_names)
        self.add_orig_nodes_to_fused_nodes()

        self.handle_external_colocation()

        # fill node data
        for fused_node in self.fused_nodes:
            fused_node.fill_fused_node_info()

    def add_orig_nodes_to_fused_nodes(self):
        """ Add gde nodes to the FusedGraph """
        for node in self._gde_graph.nodes:
            fused_node = self.get_fused_node_for_node(node.name)

            if fused_node is None:
                # if Node cannot be put under any existing FusedNode
                # create a new FusedNode containing just this node.
                fused_node = FusedNode(self, node.name)
                self.add_node(fused_node)

            fused_node.add_node(node.name)
            self.update_node_name_to_fused_node(node.name, fused_node)
