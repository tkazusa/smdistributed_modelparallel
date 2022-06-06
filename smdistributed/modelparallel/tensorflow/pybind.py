# Third Party
import tensorflow as tf
from tensorflow.core.framework import graph_pb2, node_def_pb2

# First Party
from smdistributed.modelparallel.tensorflow import graph_def_editor as gde
from smdistributed.modelparallel.tensorflow import smp_tf_pybindlib

# This is in a separate file because it requires
# importing from the main package or init file of modelparallel/tensorflow
# Separation makes it easier to prevent circular imports.


def inline_funclib(in_graph_def):
    """ Inlines all the function library in input graphdef
        Args:
            in_graph_def: Input Graphdef
        Returns:
            Returns an output graphdef with all the FunctionLibrary Inlined.
            If input graphdef does not have a function library , will return the input graphdef.
    """

    if in_graph_def.library.function:
        graph_def_str = in_graph_def.SerializeToString()
        return_str = smp_tf_pybindlib.inline_funclib(graph_def_str)
        output_graph_def = graph_pb2.GraphDef()
        output_graph_def.ParseFromString(return_str)
        return output_graph_def
    else:
        return in_graph_def


def metis_partition(num_parts, nvtxs, n_edges, ncons, xadj, adjncy, adjwgt, vwgt, contiguous):
    """Run metis partition for the graph provided"""
    partition_and_edgecut = smp_tf_pybindlib.metis_partition(
        num_parts, nvtxs, n_edges, ncons, xadj, adjncy, adjwgt, vwgt, contiguous
    )
    partitions = partition_and_edgecut[:-1]
    edge_cut = partition_and_edgecut[-1]
    return [int(x) for x in partitions], int(edge_cut)


def backward_walk(graph_id, start_nodes, stopping_ops, target_ops):
    def _node_def(node):
        if isinstance(node, gde.Node):
            return node.to_node_def()
        elif isinstance(node, tf.Operation):
            return node.node_def
        else:
            return node

    start_nodes = [_node_def(node).SerializeToString() for node in start_nodes]
    return smp_tf_pybindlib.backward_walk(graph_id, start_nodes, stopping_ops, target_ops)


def set_graph(graph_id, graph_def):
    graph_def_str = graph_def.SerializeToString()
    smp_tf_pybindlib.set_graph(graph_id, graph_def_str)


def get_ops_of_type(graph_id, op_types, microbatch, forward_only):
    mb = microbatch if microbatch is not None else -1
    return smp_tf_pybindlib.get_ops_of_type(graph_id, op_types, mb, forward_only)


def get_nodes_by_names(graph_id, names):
    if len(set(names)) != len(names):
        raise ValueError("Duplicate names are not supported.")

    node_def_strs = smp_tf_pybindlib.get_nodes_by_names(graph_id, names)
    node_defs = [node_def_pb2.NodeDef() for name in names]
    [
        node_def.ParseFromString(node_def_str)
        for node_def, node_def_str in zip(node_defs, node_def_strs)
    ]
    return node_defs


def register_xla_optimizer():
    smp_tf_pybindlib.register_xla_optimizer()
