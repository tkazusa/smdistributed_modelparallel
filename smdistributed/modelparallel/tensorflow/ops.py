# Third Party
import tensorflow as tf
from tensorflow.python.eager.backprop import _MockOp as MockOp
from tensorflow.python.framework import ops

# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup, RankType
from smdistributed.modelparallel.backend.utils import flatten, get_ext_suffix
from smdistributed.modelparallel.tensorflow import core_mod, state_mod
from smdistributed.modelparallel.tensorflow.attrs import (
    LINK_ID_ATTR,
    MICROBATCH_ATTR,
    OP_ID_ATTR,
    PEER_ATTR,
)
from smdistributed.modelparallel.tensorflow.compile import CompileStatus, OpBehavior
from smdistributed.modelparallel.tensorflow.utils import (
    get_dummy_spec,
    load_lib,
    make_tf_compatible_name,
)

# If Horovod is in the environment, load Horovod library first before loading SMP library.
# Otherwise TF cannot find symbols in Horovod library when it is imported afterwards.
# This has to do with the fact that we are linking against Horovod library while building SMP.
try:
    import horovod.tensorflow as hvd  # noqa
except ImportError:
    pass

INPUT_OP = "SmpInput"
OUTPUT_OP = "SmpOutput"
BUNDLE_OP = "SmpBundle"
COMM_OP = "SmpComm"
IDENTITY_OP = "SmpIdentity"
REGISTER_OP = "SmpRegister"
ALLGATHER_OP = "SmpAllgather"

SMPLIB = load_lib(
    "./smplib" + get_ext_suffix(),
    [INPUT_OP, OUTPUT_OP, BUNDLE_OP, COMM_OP, IDENTITY_OP, REGISTER_OP, ALLGATHER_OP],
)

# Above core_mod and state_mod were imported instead of actual attributes or methods here
# because state and pp_rank are modified by patch during tests
# if we imported those objects/methods here we will not get the patched objects/methods


def _get_true_op(op):
    if isinstance(op, MockOp):
        return op.outputs[0].op
    else:
        return op


def input(tensor, control_input, op_id, sg, dummy=False, name=None):
    """A wrapper for a SmpInput op, which is placed at the inputs of subgraphs."""
    name = name or "SmpInput_%s" % make_tf_compatible_name(tensor.name)

    # check CompileStatus, decide whether to use shape/dtype or not
    if state_mod.state.compile_status == CompileStatus.TRAIN:
        tick, peer, link_id = state_mod.state.compiler.op_attr[op_id]
        shape, dtype = state_mod.state.compiler.op_metadata[op_id]
        # bundle the control input with a placeholder constant - if XLA is enabled, this constant will later
        # be replaced by the XLA-induced control input from another microbatch
        bundled_ctrl_input = bundle([control_input, tf.constant(0.0, dtype=control_input.dtype)])
        output = SMPLIB.smp_input(
            tensor=tensor,
            control_input=bundled_ctrl_input,
            op_id=op_id,
            tick=tick,
            peer=peer,
            link_id=link_id,
            expected_shape=shape,
            out_type=dtype,
            forward=True,
            dummy=dummy,
            microbatch=state_mod.state.microbatch,
            name=name,
        )
    else:
        if state_mod.state.subgraph_to_device[sg] == core_mod.pp_rank():
            # infer shape and dtype of the expected tensor
            shape = [(dim if dim is not None else -1) for dim in tensor.shape.as_list()]
            dtype = tensor.dtype
        else:
            shape, dtype = get_dummy_spec()
        output = SMPLIB.smp_input(
            tensor=tensor,
            control_input=control_input,
            op_id=op_id,
            tick=-1,
            peer=-1,
            link_id=-1,
            expected_shape=shape,
            out_type=dtype,
            forward=True,
            dummy=dummy,
            microbatch=state_mod.state.microbatch,
            name=name,
        )
    # state_mod.state.op_to_sg[output.op] = sg
    return output


def output(tensor, control_input, op_id, sg, output_shape=None, output_dtype=None, name=None):
    """A wrapper for a SmpOutput op, which is placed at the outputs of subgraphs."""

    name = name or "SmpOutput_%s" % make_tf_compatible_name(tensor.name)

    # check CompileStatus, decide whether to use shape/dtype or not
    if state_mod.state.compile_status == CompileStatus.TRAIN:
        tick, peer, link_id = state_mod.state.compiler.op_attr[op_id]
        shape, dtype = state_mod.state.compiler.op_metadata[op_id]
        output = SMPLIB.smp_output(
            tensor,
            control_input,
            op_id=op_id,
            tick=tick,
            peer=peer,
            link_id=link_id,
            expected_shape=shape,
            out_type=dtype,
            forward=True,
            microbatch=state_mod.state.microbatch,
            name=name,
        )
    else:
        if state_mod.state.subgraph_to_device[sg] == core_mod.pp_rank():
            # infer shape and dtype of the expected tensor
            shape = [(dim if dim is not None else -1) for dim in tensor.shape.as_list()]
            dtype = tensor.dtype
        else:
            dtype = output_dtype
            shape = [(dim if dim is not None else -1) for dim in output_shape]
        output = SMPLIB.smp_output(
            tensor,
            control_input,
            op_id=op_id,
            tick=-1,
            peer=-1,
            link_id=-1,
            expected_shape=shape,
            out_type=dtype,
            forward=True,
            microbatch=state_mod.state.microbatch,
            name=name,
        )
    # state_mod.state.op_to_sg[output.op] = sg
    return output


@ops.RegisterGradient("SmpInput")
def input_grad(op, grad):
    state_mod.state.backward = True
    op_ = _get_true_op(op)
    fwd_op_id = op_.get_attr(OP_ID_ATTR)
    microbatch = op_.get_attr(MICROBATCH_ATTR)
    peer = op_.get_attr(PEER_ATTR)
    link_id = op_.get_attr(LINK_ID_ATTR)

    bwd_op_id = state_mod.state.get_bwd_id(fwd_op_id)
    state_mod.state.op_id_to_device[bwd_op_id] = state_mod.state.op_id_to_device[fwd_op_id]
    # If you find a bwd op behavior with STOP_GRADIENT, just return None.
    # This could potentially happen depending on how graph gets partitioned.
    if bwd_op_id in state_mod.state.compiler.op_behavior[microbatch]:
        if state_mod.state.compiler.op_behavior[microbatch][bwd_op_id] == OpBehavior.STOP_GRADIENT:
            return [None, None]

    maintain_spec = (
        state_mod.state.compile_status == CompileStatus.STEP_COMPILE
        or state_mod.state.compiler.op_behavior[microbatch][fwd_op_id] != OpBehavior.DUMMY
    )
    if maintain_spec:
        shape = [(dim if dim is not None else -1) for dim in op_.inputs[0].shape.as_list()]
        dtype = op_.inputs[0].dtype
    else:
        shape, dtype = get_dummy_spec()

    tick = state_mod.state.pipeline.get_tick(bwd_op_id, microbatch)

    output = SMPLIB.smp_output(
        grad,
        tf.constant(0.0),
        op_id=bwd_op_id,
        tick=tick,
        peer=peer,
        link_id=link_id,
        expected_shape=shape,
        out_type=dtype,
        microbatch=microbatch,
        forward=False,
    )

    bundled_control_grad = bundle([tf.cast(output, dtype=tf.float32)])
    if maintain_spec or len(op_.inputs[0].shape.as_list()) == 0:
        return [output, bundled_control_grad]
    else:
        # if we converted a real input into dummy, gradient does not matter for that input in this rank
        return [None, bundled_control_grad]


@ops.RegisterGradient("SmpOutput")
def output_grad(op, grad):
    state_mod.state.backward = True
    op_ = _get_true_op(op)
    fwd_op_id = op_.get_attr(OP_ID_ATTR)
    microbatch = op_.get_attr(MICROBATCH_ATTR)
    peer = op_.get_attr(PEER_ATTR)
    link_id = op_.get_attr(LINK_ID_ATTR)

    shape = [(dim if dim is not None else -1) for dim in op_.inputs[0].shape.as_list()]
    dtype = op_.inputs[0].dtype
    bwd_op_id = state_mod.state.get_bwd_id(fwd_op_id)
    tick = state_mod.state.pipeline.get_tick(bwd_op_id, microbatch)

    bundled_ctrl_input = bundle([op_.outputs[0], tf.constant(0.0, dtype=op_.outputs[0].dtype)])

    output = SMPLIB.smp_input(
        grad,
        control_input=bundled_ctrl_input,
        op_id=bwd_op_id,
        tick=tick,
        peer=peer,
        link_id=link_id,
        expected_shape=shape,
        out_type=dtype,
        microbatch=microbatch,
        forward=False,
        dummy=False,
    )
    state_mod.state.op_id_to_device[bwd_op_id] = state_mod.state.op_id_to_device[fwd_op_id]
    return output, bundle([tf.cast(output, dtype=tf.float32)])


def bundle(inputs, name=None):
    if inputs == None or len(inputs) == 0:
        return SMPLIB.smp_bundle([tf.constant(0.0)], name=name)
    for i in range(len(inputs)):
        if inputs[i].dtype != tf.float32:
            inputs[i] = tf.cast(inputs[i], dtype=tf.float32)
    return SMPLIB.smp_bundle([inp for inp in inputs], name=name)


@ops.RegisterGradient("SmpBundle")
def bundle_grad(op, grad):
    op_ = _get_true_op(op)
    return [grad for _ in range(op_.get_attr("N"))]


def identity(tensor):
    return SMPLIB.smp_identity(tensor)


def register_state():
    _, op_ids, behaviors = state_mod.state.compiler.get_op_behaviors()
    in_counts = state_mod.state.compiler.io_counts[0]
    out_counts = state_mod.state.compiler.io_counts[1]
    recv_link_ids = flatten(state_mod.state.compiler.recv_link_ids)

    return SMPLIB.smp_register(
        in_counts=in_counts,
        out_counts=out_counts,
        op_ids=op_ids,
        behaviors=behaviors,
        recv_link_ids=recv_link_ids,
    )


@ops.RegisterGradient("SmpIdentity")
def identity_grad(op, grad):
    return SMPLIB.smp_identity(grad)


def allgather_tensor_mpgroup(tensor):
    return allgather_tensor_ppgroup(tensor)


def allgather_tensor_ppgroup(tensor):
    """
    Allgather one tensor from all pp ranks

    Input: the tensor to be allgathered
    Output: A tf.stack() of all tensors gathered
    """

    if tensor.dtype == tf.bool:
        tensor_cast = tf.cast(tensor, dtype=tf.int32)
    else:
        tensor_cast = tensor

    # get op id
    op_id = state_mod.state.op_id_gen.get_op_id(
        state_mod.state.microbatch, -1, state_mod.state.allgather_idx
    )

    # get link id
    if core_mod.pp_rank() == 0:
        link_id = state_mod.state.get_comm_link_id(0, state_mod.state.allgather_idx)
        state_mod.state.comm.broadcast(link_id, group=CommGroup.PP_GROUP)
    else:
        link_id = state_mod.state.comm.recv_from(0, rank_type=RankType.PP_RANK)

    # pp_group is the ranks that belong to this model replica
    pp_group = core_mod.get_pp_group()

    allgather_outputs = SMPLIB.smp_allgather(
        tensor_cast,
        rank=core_mod.rank(),
        pp_group=pp_group,
        link_id=link_id,
        op_id=op_id,
        pp_size=core_mod.pp_size(),
        microbatch=state_mod.state.microbatch,
    )

    state_mod.state.allgather_idx += 1
    if tensor.dtype == tf.bool:
        return tf.stack([tf.cast(out, dtype=tensor.dtype) for out in allgather_outputs])
    else:
        return tf.stack(list(allgather_outputs))


def is_all_finite_distributed(grads):
    is_finite_per_grad = [tf.reduce_all(tf.math.is_finite(g)) for g in grads if g is not None]
    is_finite_per_pp_rank = tf.reduce_all(is_finite_per_grad)
    return tf.reduce_all(allgather_tensor_ppgroup(is_finite_per_pp_rank))


def accumulate(grad_vars, method="variable"):
    return [(g.accumulate(var=v.outputs[0], method=method), v.outputs[0]) for g, v in grad_vars]
