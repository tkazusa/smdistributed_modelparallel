# Standard Library
from typing import Any, Sequence, Union

# Third Party
import torch
from torch.nn import Module

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import pp_rank
from smdistributed.modelparallel.torch.exceptions import (
    InvalidExecutor,
    MissingPathFromComputationToModuleOutput,
    MissingPathFromModuleInputToModuleOutput,
)
from smdistributed.modelparallel.torch.graph_utils import get_ancestors
from smdistributed.modelparallel.torch.messages import ModuleExecutionRequest
from smdistributed.modelparallel.torch.module_manager import TensorModuleInfo
from smdistributed.modelparallel.torch.offload import can_shard_activation_offloading
from smdistributed.modelparallel.torch.ops import (
    SMPInput,
    SMPParentRecv,
    increment_prev_parent_recv_counts,
)
from smdistributed.modelparallel.torch.patches.checkpoint import (
    CheckpointTupledFunction,
    checkpoint_function,
    validate_checkpointable,
)
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import (
    check_env_var_truthy,
    check_requires_grad,
    check_supported,
    convert_args_to_device,
    flatten_structure,
    flattened,
    map_structure,
    rmsg,
    unflatten_structure,
)

logger = get_logger()

"""
Patch Manager uses the functions defined here to override actual methods
"""

_skip_validation_graph = check_env_var_truthy("SMP_SKIP_GRAPH_VALIDATION", "false")


def _validate_smpparents(module_name):
    smpparents = state.module_manager.pop_smpparents(state.microbatch, module_name)
    if not _skip_validation_graph:
        for smpparent in smpparents:
            if smpparent.next_parent_recvs_bwd_pending <= 0:
                raise MissingPathFromComputationToModuleOutput(
                    module_name, state.module_manager.get_module_name(smpparent.module)
                )


def _validate_smpinputs(module_name):
    smpinputs = state.module_manager.pop_smpinputs(state.microbatch, module_name)
    if not _skip_validation_graph:
        for smpinput in smpinputs:
            if smpinput.pending_smpinput_bwds <= 0:
                raise MissingPathFromModuleInputToModuleOutput(module_name, smpinput.idx)


def insert_smp_input_op(module: Module, *args: Any, **kwargs: Any):
    # SMPInput op helps in sending backward pass requests to the moduels above this module
    # if current module is not main
    smp_inputs, structure_id = flatten_structure((args, kwargs))

    smp_outs = []
    tensors_from_smp_input = []
    for idx, smp_input in enumerate(smp_inputs):
        if (
            isinstance(smp_input, torch.Tensor)
            and smp_input.requires_grad
            and torch.is_grad_enabled()
            and not state.module_manager.is_parent_executor(module)
        ):
            smp_out = SMPInput.apply(module, idx, smp_input)
            smp_outs.extend(list(smp_out))
            tensors_from_smp_input.extend(list(smp_out))
        else:
            smp_outs.append(smp_input)
    args, kwargs = unflatten_structure(smp_outs, structure_id)
    module_name = state.module_manager.get_module_name(module)
    state.module_manager.push_smpinputs(state.microbatch, module_name, tensors_from_smp_input)
    return args, kwargs


def pass_custom_attributes(tensor_in, tensor_out):
    if hasattr(tensor_in, "_smp_module_info"):
        tensor_out._smp_module_info = tensor_in._smp_module_info
    if hasattr(tensor_in, "_smp_is_dummy"):
        tensor_out._smp_is_dummy = tensor_in._smp_is_dummy
        state.serialization_manager.update_dummy(tensor_in, tensor_out)


def detach_outputs(outputs):
    # detach the outputs from computation graph and make it leaves
    # This will help us pause backward execution here, until new request
    # for next backward on this module comes
    # child_output_leaves will have same requires_grad as outputs.
    def _tensor_detach(x):
        if isinstance(x, torch.Tensor):
            y = x.detach()
            y.requires_grad = x.requires_grad
            pass_custom_attributes(x, y)
            return y
        else:
            return x

    return map_structure(_tensor_detach, outputs)


def increment_smp_input_bwd_counts(smp_inputs):
    # Increment bwd count
    for node in smp_inputs:
        node.pending_smpinput_bwds += 1


def maybe_push_outputs(module, outputs):
    if (
        not state.module_manager.is_main_module(module)
        and check_requires_grad(outputs)
        and torch.is_grad_enabled()
    ):
        flat_outputs, _ = flatten_structure(outputs)
        flat_outputs = tuple(out for out in flat_outputs if isinstance(out, torch.Tensor))
        parent_module = state.module_manager.get_parent_module(module)
        state.module_manager.push_output(state.microbatch, module, parent_module, flat_outputs)
        position = (
            state.module_manager.output_stack_size(state.microbatch, module, parent_module) - 1
        )

        params, smp_inputs, parent_recvs = get_ancestors(
            flat_outputs, checkpoint_nodes_cache=state.checkpoint_nodes_cache
        )
        logger.debug(
            rmsg(
                f"{len(params)} params are reachable from output of {state.module_manager.get_module_name(module)}"
            )
        )
        increment_smp_input_bwd_counts(smp_inputs)
        _validate_smpinputs(state.module_manager.get_module_name(module))
        state.model.grad_counter.increment_expected_num_grads(
            state.microbatch, [state.model.get_param_name(p) for p in params]
        )
        increment_prev_parent_recv_counts(parent_recvs)
        state.module_manager.save_smpinput_bwd_count(
            state.microbatch, position, module, parent_module, smp_inputs
        )


def execute_as_parent(module: Module, *args: Any, **kwargs: Any):
    check_supported(
        state.module_manager.get_module_name(module), args, kwargs, skip_tensor_check=True
    )
    # forward dispatches request to the module (another rank)
    # and returns the outputs. Also args are added to stack
    # so that backward execution can be resumed from this point
    # by popping the stack

    parent_module = state.module_manager.get_parent_module(module)

    # if checkpoint is with set_activation_checkpointing, then remote rank will set the checkpoint config to state
    # if it was checkpoint fn, then below call will forward that config to remote rank
    request = ModuleExecutionRequest.create_forward_req(
        module,
        args,
        kwargs,
        enable_grads=torch.is_grad_enabled(),
        enable_autocast=torch.is_autocast_enabled(),
        checkpoint_activations_config=state.checkpoint_activations_config
        if state.checkpoint_activations_config.enabled_for_module(module)
        else None,
        position=state.module_manager.output_stack_size(state.microbatch, module, parent_module),
    )
    outputs = state.current_worker.thread_get_forward_result(request)

    smp_child_inputs, structure_id = flatten_structure((args, kwargs))

    # Need to set these additional params and add these to forward.
    # This is because the get_ancestors function when called on
    # the smp_child_inputs, wont return the torch.nn.Parameters
    # already in the inputs, i.e. passed in the forward of a module.
    # It only returns the ancestors to the node and not the node itself.
    additional_params = set()
    for smp_child_input in smp_child_inputs:
        if isinstance(smp_child_input, torch.nn.parameter.Parameter):
            additional_params.add(smp_child_input)

    # only push to stack if the output is pushed to stack on the rank
    # executing the child module. This ensures the stack size to be
    # the same on both modules
    if check_requires_grad(outputs) and torch.is_grad_enabled():
        smp_child_inputs = tuple(smp_child_inputs)
        state.module_manager.push_output(state.microbatch, module, parent_module, smp_child_inputs)

        params, smp_inputs, parent_recvs = get_ancestors(
            smp_child_inputs, checkpoint_nodes_cache=state.checkpoint_nodes_cache
        )
        logger.debug(
            rmsg(
                f"{len(params)} params are reachable from inputs of {state.module_manager.get_module_name(module)}"
            )
        )
        params = params.union(additional_params)
        state.model.grad_counter.increment_expected_num_grads(
            state.microbatch, [state.model.get_param_name(p) for p in params]
        )
        increment_smp_input_bwd_counts(smp_inputs)
        child_output_leaves = detach_outputs(outputs)

        count = sum(
            [1 if isinstance(x, torch.Tensor) and x.requires_grad else 0 for x in smp_child_inputs]
        )
        flat_leaves, structure_id = flatten_structure(child_output_leaves)

        filtered_flat_leaves = [
            (i, leaf) for i, leaf in enumerate(flat_leaves) if isinstance(leaf, torch.Tensor)
        ]
        indices = [i for i, _ in filtered_flat_leaves]
        leaves = [leaf for _, leaf in filtered_flat_leaves]

        tensor_outputs = SMPParentRecv.apply(
            parent_module, module, count, parent_recvs, False, *leaves
        )
        _, ancestor = state.module_manager.find_boundary_ancestors(
            state.microbatch, state.module_manager.get_parent_module(module)
        )
        ancestor_name = state.module_manager.get_module_name(ancestor)
        state.module_manager.push_smpparents(state.microbatch, ancestor_name, tensor_outputs)

        # Workaround for https://issues.amazon.com/issues/P40416033
        tensor_outputs_cln = [output.clone() for output in tensor_outputs]

        # copy the module info from the original tensor
        for leaf, tensor in zip(leaves, tensor_outputs_cln):
            pass_custom_attributes(leaf, tensor)

        outputs = flat_leaves
        for i, ind in enumerate(indices):
            outputs[ind] = tensor_outputs_cln[i]

        outputs = unflatten_structure(outputs, structure_id)
    return outputs


def actual_forward(module, *args, **kwargs):
    if state.checkpoint_activations_config.enabled_for_module(module) and torch.is_grad_enabled():
        assert (
            len(kwargs) == 0
        ), "modules with kwargs can not be checkpointed. Dev note: Error should have been thrown before this point."
        outputs = CheckpointTupledFunction.apply(
            module._orig_forward,
            state.checkpoint_activations_config.preserve_rng_state,
            state.checkpoint_activations_config.pack_args_as_tuple,
            can_shard_activation_offloading(module),
            *args,
        )
    else:
        outputs = module._orig_forward(*args, **kwargs)
    return outputs


def execute_module(module: Module, *args, **kwargs: Any):
    check_supported(state.module_manager.get_module_name(module), args, kwargs)

    if state.cfg.fast_mode and state.current_minibatch() == 0 and state.microbatch == 0:
        # in most cases, we will have already called this method before in worker.py,
        # and this will be a no-op. the only known scenario where this is useful is the following:
        # parent (0)
        #     y = mod1(x) (1)
        #     z = child1(y) (0)
        #     t = child2(y) (2)
        # in this case, the _smp_module_info attribute coming out of child0 will be parsed for
        # the first time in child1. in this case, since the parent is on the same rank as child1,
        # a child-to-child transmission will NOT take place, and it is the parent that sends y even to child2.
        # this call here ensures this.

        with flattened((args, kwargs)) as flat_args:
            # if the current rank is the generator already, do not add to direct consumer map
            state.current_step_func().update_direct_fwd_consumers(
                flat_args, module=module, exclude_partitions=[pp_rank()]
            )

    if not state.is_in_fwd_on_checkpointed_fn:
        if not state.module_manager.is_main_module(
            module
        ) and not state.module_manager.is_parent_executor(module):
            parent_module = state.module_manager.get_parent_module(module)
            if not state.module_manager.is_correct_parent(module, parent_module):
                # torch.nn.ModuleList is an edge case since it doesnt have a forward
                # and the parent may be wrong. Get the actual parent, assert that
                # its module list and insert smp input op if current rank is not the
                # executor of grand_parent
                grand_parent, parent = state.module_manager.get_immediate_ancestors(module)
                assert isinstance(parent, torch.nn.ModuleList), (
                    "actual_parent should be different than module_execution_stack parent only for torch.nn.ModuleList",
                    state.module_manager.get_module_name(parent),
                )
                if grand_parent and not state.module_manager.is_executor(grand_parent):
                    args, kwargs = insert_smp_input_op(module, *args, **kwargs)
            else:
                args, kwargs = insert_smp_input_op(module, *args, **kwargs)

    # helps keep track of parents of a module execution
    state.module_manager.record_traversal_into_module(module)

    if (
        state.module_manager.should_checkpoint_activations(module)
        and not state.checkpoint_activations_config.enabled_for_module(module)
        and validate_checkpointable(module)
    ):
        # only modules set with set_activation_checkpointing will match this condition
        # for calls of checkpoint and checkpoint_sequential, this context manager will already be set
        config = state.module_manager.get_checkpoint_activations_config(module)

        if kwargs:
            raise ValueError(
                "Checkpointed module can not have kwargs, please restructure the module to take only non keyword args"
            )

        outputs = checkpoint_function(
            actual_forward,
            module,
            *args,
            module_name=state.module_manager.get_module_name(module),
            preserve_rng_state=config.preserve_rng_state,
            pack_args_as_tuple=config.pack_args_as_tuple,
        )
    else:
        outputs = actual_forward(module, *args, **kwargs)

    if not state.is_in_fwd_on_checkpointed_fn:
        module_name = state.module_manager.get_module_name(module)
        if not state.module_manager.is_parent_executor(module):
            parent_module = state.module_manager.get_parent_module(module)
            if not state.module_manager.is_main_module(
                module
            ) and not state.module_manager.is_correct_parent(module, parent_module):
                # torch.nn.ModuleList is an edge case since it doesnt have a forward
                # and the parent may be wrong. Get the actual parent, assert that
                # its module list and push outputs only if current rank is not the
                # executor of grand_parent
                grand_parent, parent = state.module_manager.get_immediate_ancestors(module)
                assert isinstance(
                    parent, torch.nn.ModuleList
                ), "actual_parent should be different than module_execution_stack parent only for torch.nn.ModuleList"
                if grand_parent and not state.module_manager.is_executor(grand_parent):
                    maybe_push_outputs(module, outputs)
            else:
                maybe_push_outputs(module, outputs)

            if not state.module_manager.is_main_module(module):
                _validate_smpparents(module_name)

    if state.cfg.fast_mode:
        # mark module info
        with flattened(outputs) as flat_outputs:
            for i, output in enumerate(flat_outputs):
                if isinstance(output, torch.Tensor) and not hasattr(output, "_smp_module_info"):
                    module_name = state.module_manager.get_module_name(module)
                    count = state.current_step_func().get_fwd_module_execution_count(
                        module, state.microbatch
                    )
                    state.current_step_func().increment_fwd_module_execution_count(
                        module, state.microbatch
                    )
                    flat_outputs[i]._smp_module_info = TensorModuleInfo(
                        module_name, count, i, True, None
                    )

    state.module_manager.finished_module_exec()

    return outputs


def distributed_forward(module: Module, *args: Any, **kwargs: Any):
    """
    Executes the module on the device which was assigned to it.
    This function should only do local compute directly. For any other IO tasks
    use the current_worker's comm_queue to send messages to <coordinator thread in worker> and then to server.
    """
    if state.module_manager.is_executor(module):
        outputs = module.execute_locally(*args, **kwargs)
    elif state.module_manager.is_parent_executor(module):
        outputs = execute_as_parent(module, *args, **kwargs)
    else:
        raise InvalidExecutor(
            state.module_manager.get_module_name(module), state.module_manager.get_partition(module)
        )
    return outputs


def distributed_backward(
    mod: Module,
    tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
    grad_tensors: Union[torch.Tensor, Sequence[torch.Tensor], None] = None,
) -> None:
    exec_stack = [state.module_manager.get_module_name(mod)]

    tensors, _ = flatten_structure(tensors)
    grad_tensors, _ = flatten_structure(grad_tensors)

    # push final output
    state.module_manager.push_output(state.microbatch, mod, mod, tensors)

    # update parent recv count for parent to child aggregation
    params, _, parent_recvs = get_ancestors(
        tensors, checkpoint_nodes_cache=state.checkpoint_nodes_cache
    )
    logger.debug(rmsg(f"{len(params)} params are reachable from final outputs"))

    state.model.grad_counter.increment_expected_num_grads(
        state.microbatch, [state.model.get_param_name(p) for p in params]
    )
    increment_prev_parent_recv_counts(parent_recvs)
    _validate_smpparents(state.module_manager.get_module_name(mod))

    # update bwd count for child to parent aggregation
    state.module_manager.increment_bwd_count(state.microbatch, mod, mod)

    # wait for actual backward start
    state.current_worker.thread_wait_for_backward_start()

    request = ModuleExecutionRequest.create_backward_req(
        mod, grad_tensors, position=-1, sender_module=mod, execution_stack=exec_stack
    )

    state.current_worker.thread_get_backward_result(request)

    # Push wait only if there is a dummy result or another backward request we are waiting for
    if not state.module_manager.check_no_pending_bwd(request.mb, mod):
        state.current_worker.thread_wait_for_backward_done()
