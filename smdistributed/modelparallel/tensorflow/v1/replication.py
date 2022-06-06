"""
Given graph definition for a single microbatch, replicate operations for 1..N-1 microbatches
"""
# Third Party
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.contrib.graph_editor import subgraph
from tensorflow.python.ops.control_flow_ops import CondContext

# First Party
from smdistributed.modelparallel.tensorflow import core
from smdistributed.modelparallel.tensorflow.attrs import (
    LINK_ID_ATTR,
    MICROBATCH_ATTR,
    OP_ID_ATTR,
    PEER_ATTR,
    TICK_ATTR,
)
from smdistributed.modelparallel.tensorflow.compile import CompileStatus
from smdistributed.modelparallel.tensorflow.split import is_last_split_op
from smdistributed.modelparallel.tensorflow.state_mod import state
from smdistributed.modelparallel.tensorflow.utils import get_op

VARIABLE_TYPES = [
    "Variable",
    "VariableV2",
    "AutoReloadVariable",
    "VarHandleOp",
    "_VarHandlesOp",
    "ReadVariableOp",
    "_ReadVariablesOp",
]


def copy_with_input_replacement(outputs, input_mapping, mb):
    """Replicate the graph, going backward from 'outputs', until a SMPSplit operation is encountered. tf.Variables
    are excluded from replication, and remain connected to both replicas of the ops that consume them. input_mapping
    is a dict that maps tf.Tensors to tf.Tensors. If input_mapping[x] = y, all ops that consume Tensor x in the original
    graph will instead consume Tensor y in the replicated graph."""

    ops = tf.get_default_graph().get_operations()
    smp_split_ops = [x.outputs[0] for x in ops if is_last_split_op(x)]
    stopping_ops = smp_split_ops

    flat_outputs = tf.nest.flatten(outputs)
    flat_outputs = [out for out in flat_outputs if out is not None]
    flat_output_ops = tf.nest.map_structure(get_op, flat_outputs)

    ops_to_copy = ge.get_backward_walk_ops(
        flat_output_ops, inclusive=True, stop_at_ts=stopping_ops, control_inputs=True
    )
    ops_to_copy = list(filter(lambda x: x.type not in VARIABLE_TYPES, ops_to_copy))
    copied_ops, op_mapping = replicate_graph_ops(ops_to_copy, input_mapping, mb)

    return tf.nest.map_structure(lambda x: map_tensor_or_op(op_mapping, x), outputs)


def map_tensor_or_op(op_mapping, t):
    if isinstance(t, tf.Tensor):
        if t.op in op_mapping:
            return op_mapping[t.op].outputs[t._value_index]
        else:
            return t
    elif isinstance(t, tf.Operation):
        if t in op_mapping:
            return op_mapping[t]
        else:
            return t
    else:
        return t


def replicate_graph_ops(ops, input_mapping, mb):
    """Replicate the list of ops, subject to input_mapping"""
    copy_transformer = ge.Transformer()
    nodedef_mapper = _get_nodedef_mapper(mb)
    copy_transformer.transform_op_handler = inject_param_decorator(
        ge.copy_op_handler, nodedef_fn=nodedef_mapper
    )
    copy_transformer.transform_external_input_handler = _get_input_replacement_func(input_mapping)
    sgv = subgraph.make_view(ops, graph=tf.get_default_graph())
    ops_copy, mapping = copy_transformer(
        sgv, dst_graph=tf.get_default_graph(), dst_scope="", src_scope=""
    )
    add_to_cond_context_collection(mapping)
    return ops_copy._ops, mapping._transformed_ops


def add_to_cond_context_collection(mapping):
    """After the graph has been replicated, it is necessary to make the Conditional Contexts for the replicated
       tensors and include them in the COND_CONTEXT collection of the graph. Otherwise control flow ops fail to work
       properly.
       The following function iterates through all the existing conditional contexts. For every conditional context
       (cond_cont) it verifies if the tensors within the context have been replicated or not. This is confirmed by
       looking for the presence of cond_cont's pivot (switch_t/switch_f) and the pred tensors in the mapping. If both
       these tensors have been mapped, it means that both these tensors have been replicated and that we should make
       a new CondContext that should include the newly replicated tensors. Once the new CondContext is made, it is
       added to the CondContext collection.
    """
    cond_lst = tf.get_collection(tf.GraphKeys.COND_CONTEXT)
    for cond_cont in cond_lst:
        if cond_cont.pivot in mapping._transformed_ts and cond_cont.pred in mapping._transformed_ts:
            tf.add_to_collection(
                tf.GraphKeys.COND_CONTEXT,
                CondContext(
                    mapping._transformed_ts[cond_cont.pred],
                    mapping._transformed_ts[cond_cont.pivot],
                    cond_cont.branch,
                    name=cond_cont.name,
                    import_scope="",
                ),
            )


def _get_input_replacement_func(tensor_mapping):
    """Build mapper for input replacement used by ge.Transformer"""

    def replace_t(info, t):
        if t in tensor_mapping:
            return tensor_mapping[t]
        else:
            return ge.keep_t_if_possible_handler(info, t)

    return replace_t


def _get_nodedef_mapper(mb):
    """Build a function to be used by ge.Transformer to modify node attributes when copying"""

    def nodedef_mapper(node_def):
        if is_smp_op(node_def):

            if state.compile_status == CompileStatus.STEP_COMPILE:
                node_def.attr[OP_ID_ATTR].i = state.op_id_gen.transpose_op_id(
                    node_def.attr[OP_ID_ATTR].i, mb
                )
                node_def.attr[MICROBATCH_ATTR].i = mb
            else:
                op_id = state.op_id_gen.transpose_op_id(node_def.attr[OP_ID_ATTR].i, mb)
                tick, peer, link_id = state.compiler.op_attr[mb][op_id]

                node_def.attr[MICROBATCH_ATTR].i = mb
                node_def.attr[OP_ID_ATTR].i = op_id
                if not is_smp_allgather_op(node_def):
                    node_def.attr[TICK_ATTR].i = tick
                    node_def.attr[PEER_ATTR].i = core.get_pp_group()[peer]
                node_def.attr[LINK_ID_ATTR].i = link_id

        return node_def

    return nodedef_mapper


def is_smp_op(node):
    return (
        is_smp_input_op(node)
        or is_smp_output_op(node)
        or is_smp_comm_op(node)
        or is_smp_allgather_op(node)
    )


def is_smp_input_op(node):
    if isinstance(node, tf.Tensor):
        node = node.op
    if isinstance(node, tf.Operation):
        node = node.node_def
    return node.op == "SmpInput"


def is_smp_output_op(node):
    if isinstance(node, tf.Tensor):
        node = node.op
    if isinstance(node, tf.Operation):
        node = node.node_def
    return node.op == "SmpOutput"


def is_smp_comm_op(node):
    if isinstance(node, tf.Tensor):
        node = node.op
    if isinstance(node, tf.Operation):
        node = node.node_def
    return node.op == "SmpComm"


def is_smp_allgather_op(node):
    if isinstance(node, tf.Tensor):
        node = node.op
    if isinstance(node, tf.Operation):
        node = node.node_def
    return node.op == "SmpAllgather"


def inject_param_decorator(target_func, *inject_args, **inject_kwargs):
    """Inject args and kwargs on every call to function"""

    def wrapper(*args, **kwargs):
        return target_func(*args, *inject_args, **kwargs, **inject_kwargs)

    return wrapper
