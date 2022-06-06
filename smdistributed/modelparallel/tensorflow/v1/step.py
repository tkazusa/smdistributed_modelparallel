# Third Party
from tensorflow.python.framework import ops

# First Party
from smdistributed.modelparallel.tensorflow import core, state
from smdistributed.modelparallel.tensorflow.compile import CompileStatus
from smdistributed.modelparallel.tensorflow.graph_utils import _get_op_name
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.split import TFTensorSplitter, postprocess_outputs
from smdistributed.modelparallel.tensorflow.utils import is_tensor_or_var
from smdistributed.modelparallel.tensorflow.v1.replication import copy_with_input_replacement


def step(non_split_inputs=None, input_split_axes=None):
    """ A decorator that defines the scope over which pipelined execution will occur. Typically
    will include the forward and backward passes during training, but can also include forward only
    for evaluation etc.

    All tf.Tensor inputs will be split along the batch dimension, unless the argument name is included
    in `non_split_inputs`. The batch dimension will be assumed to be 0, unless specified otherwise in
    `input_split_axes`. Each resulting microbatch will be executed sequentially, according to the selected
    pipeline type. All returned tensors will be converted to StepOutput objects, even when they are
    inside a nested structure.
    """

    def decorate(step_fn):
        splitter = TFTensorSplitter(step_fn, non_split_inputs, input_split_axes)

        def multi_mb_call(unsplit_args, unsplit_kwargs):
            mb_zero_args, mb_zero_kwargs = splitter.preprocess_args(
                unsplit_args, unsplit_kwargs, state.num_microbatches(), 0
            )
            state.microbatch = 0
            outputs = [step_fn(*mb_zero_args, **mb_zero_kwargs)]

            if any([(out is None) for out in outputs]):
                raise ValueError("Function decorated by @smp.step must always have a return value!")

            for mb in range(1, state.num_microbatches()):
                state.microbatch = mb
                mb_args, mb_kwargs = splitter.preprocess_args(
                    unsplit_args, unsplit_kwargs, state.num_microbatches(), mb
                )
                input_mapping = {
                    arg: other_arg
                    for arg, other_arg in zip(mb_zero_args, mb_args)
                    if is_tensor_or_var(arg)
                }
                input_mapping.update(
                    {
                        mb_zero_kwargs[key]: mb_kwargs[other_key]
                        for key, other_key in zip(mb_zero_kwargs, mb_kwargs)
                        if is_tensor_or_var(mb_zero_kwargs[key])
                    }
                )

                mb_outputs = copy_with_input_replacement(outputs[0], input_mapping, mb)
                outputs.append(mb_outputs)

            return outputs

        def wrapper(*args, **kwargs):
            state.generate_trace_graph()
            # state.compile_graph = TraceGraph()

            with state.reset_allgather_index():
                state.compile_status = CompileStatus.STEP_COMPILE
                with state.compile_graph.trace():
                    multi_mb_call(args, kwargs)
                _warn_if_trainable_var_outside_model(state.compile_graph)

                for mb in range(state.num_microbatches()):
                    state.compiler.compile(
                        state.compile_graph, core.pp_rank(), mb, state.op_id_to_device
                    )

            state.compile_status = CompileStatus.TRAIN
            step_outputs = multi_mb_call(args, kwargs)

            return postprocess_outputs(step_outputs)

        return wrapper

    return decorate


def _warn_if_trainable_var_outside_model(graph):
    for var in graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES):
        op_name = _get_op_name(var.name)
        if "SMPDistributedModel/" not in op_name and "SMPDummy" not in op_name:
            get_logger().warning(
                f"Trainable variable {var.name} is not part of smp.DistributedModel. This might create duplicate versions of the variable in different ranks, and lead to unintended consequences."
            )
