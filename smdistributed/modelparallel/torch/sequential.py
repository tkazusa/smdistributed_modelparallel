# Standard Library
from typing import Any

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import pp_rank
from smdistributed.modelparallel.torch.exceptions import (
    CheckpointingError,
    InvalidExecutorError,
    SequentialBackwardBrokenError,
)
from smdistributed.modelparallel.torch.graph_utils import get_ancestors
from smdistributed.modelparallel.torch.messages import SequentialModulesExecutionRequest
from smdistributed.modelparallel.torch.module_manager import TensorModuleInfo
from smdistributed.modelparallel.torch.offload import can_shard_activation_offloading
from smdistributed.modelparallel.torch.ops import SMPParentRecv, SMPSequentialInput
from smdistributed.modelparallel.torch.patches.checkpoint import (
    CheckpointTupledFunction,
    checkpoint_function,
    validate_checkpointable,
    validate_checkpointable_sequential,
)
from smdistributed.modelparallel.torch.patches.execution import (
    actual_forward,
    detach_outputs,
    execute_as_parent,
    increment_smp_input_bwd_counts,
    insert_smp_input_op,
    maybe_push_outputs,
    pass_custom_attributes,
)
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import (
    check_requires_grad,
    check_supported,
    flatten_structure,
    flattened,
    map_structure,
    rmsg,
    unflatten_structure,
)

logger = get_logger()


def insert_input_op(module, parent_module, *args, **kwargs):
    # SMPInput op helps in sending backward pass requests to the moduels above this module
    # if current module is not main
    smp_inputs, structure_id = flatten_structure((args, kwargs))
    smp_outs = []
    num_tensors_requiring_grad = 0
    for idx, smp_input in enumerate(smp_inputs):
        if (
            isinstance(smp_input, torch.Tensor)
            and smp_input.requires_grad
            and torch.is_grad_enabled()
            and not state.module_manager.is_executor(parent_module)
        ):
            smp_out = SMPSequentialInput.apply(module, parent_module, idx, smp_input)
            smp_outs.extend(list(smp_out))
            num_tensors_requiring_grad += 1
        else:
            smp_outs.append(smp_input)
    args, kwargs = unflatten_structure(smp_outs, structure_id)
    return args, kwargs, num_tensors_requiring_grad


def execute_chain_maybe_with_checkpointing(self, exec_fn, start_index, end_index, *args):
    if (
        state.module_manager.should_checkpoint_activations(self)
        and not state.checkpoint_activations_config.enabled_for_module(self)
        and validate_checkpointable_sequential(self)
    ):

        # only modules set with set_activation_checkpointing will match this condition
        # for calls of checkpoint and checkpoint_sequential, this context manager will already be set
        config = state.module_manager.get_checkpoint_activations_config(self)
        outputs = checkpoint_function(
            exec_fn,
            (start_index, end_index, *args),
            {},
            module_name=state.module_manager.get_module_name(self),
            preserve_rng_state=config.preserve_rng_state,
            strategy=config.strategy,
        )
    else:
        outputs = exec_fn(start_index, end_index, *args)
    return outputs


def execute_chain(self, start_index, end_index, *args):
    check_supported(state.module_manager.get_module_name(self), args, {})
    if state.cfg.fast_mode and state.current_minibatch() == 0 and state.microbatch == 0:
        with flattened(args) as flat_args:
            # if the current rank is the generator already, do not add to direct consumer map

            state.current_step_func().update_direct_fwd_consumers(
                flat_args, module=self, exclude_partitions=[pp_rank()], start_index=start_index
            )

    if (
        not state.checkpoint_activations_config.enabled_for_module(self)
        or not torch.is_grad_enabled()
    ):
        send_real_bwd = True
        for idx in range(start_index, end_index):
            module = self[idx]
            state.module_manager.record_traversal_into_module(module)

            if idx == start_index:
                # args has an extra tuple around input
                # num_tensors_requiring_grad will be used in case dummy backward needs to be sent to parent executor
                input, _, num_tensors_requiring_grad = insert_input_op(
                    self[end_index - 1], self, *args
                )
                # sequential only supports one arg so unpack
                input = input[0]
                # If no tensors require grads, we still need to send a dummy request back from the child
                if num_tensors_requiring_grad == 0:
                    num_tensors_requiring_grad += 1
            send_real_bwd = send_real_bwd and check_requires_grad(input)
            kwargs = {}
            if (
                state.module_manager.should_checkpoint_activations(module)
                and not state.checkpoint_activations_config.enabled_for_module(module)
                and validate_checkpointable(module)
                and torch.is_grad_enabled()
            ):
                # only modules set with set_activation_checkpointing will match this condition
                # for calls of checkpoint and checkpoint_sequential, this context manager will already be set
                config = state.module_manager.get_checkpoint_activations_config(module)
                input = checkpoint_function(
                    actual_forward,
                    (module, input),
                    kwargs,
                    module_name=state.module_manager.get_module_name(module),
                    preserve_rng_state=config.preserve_rng_state,
                )
            else:
                input = actual_forward(module, input, **kwargs)

            if idx == end_index - 1:
                outputs = input
                if not state.module_manager.is_parent_executor(module):
                    # records output in stack, which can be popped later for backward call
                    maybe_push_outputs(module, outputs)

            state.module_manager.finished_module_exec()

        _maybe_mark_module_info(self, outputs, end_index)
        return outputs, send_real_bwd, num_tensors_requiring_grad
    else:
        preserve_rng_state = state.checkpoint_activations_config.preserve_rng_state
        strategy = state.checkpoint_activations_config.strategy  # can take contiguous or each

        # args has an extra tuple around input
        input, _, num_tensors_requiring_grad = insert_input_op(self[end_index - 1], self, *args)
        # sequential only supports one arg so unpack
        input = input[0]

        def run_function(start, end, functions):
            # runs start to end both inclusive
            def forward(input):
                for j in range(start, end + 1):
                    state.module_manager.record_traversal_into_module(functions[j])
                    original_forward = state.patch_manager.get_original_method(
                        "forward", functions[j].__class__
                    )
                    input = original_forward(functions[j], input)
                    # if seq is being executed in a no_grad block, then we don't checkpoint so it doesn't come here
                    if not check_requires_grad(input):
                        raise SequentialBackwardBrokenError(
                            state.module_manager.get_module_name(functions[j]),
                            torch.is_grad_enabled(),
                            input,
                        )
                    state.module_manager.finished_module_exec()
                _maybe_mark_module_info(self, input, end + 1)
                return input

            return forward

        functions = [self[i] for i in range(start_index, end_index)]
        if strategy == "contiguous":
            # checkpoints all minus last
            if len(functions) > 0:
                logger.debug(rmsg(f"Checkpointing {len(functions)} module(s) in sequential module"))
                # unpack here so autograd fn doesnt get a tuple of tensors
                # if autograd fn gets tuple backward gets messed up.
                # it needs top level tensors

                wrapping_obj, tensors = state.serialization_manager.serialize(
                    ((input,), {}), for_transmission=False
                )

                input = CheckpointTupledFunction.apply(
                    run_function(0, len(functions) - 1, functions),
                    preserve_rng_state,
                    can_shard_activation_offloading(functions[0]),
                    wrapping_obj,
                    *tensors,
                )
        elif strategy == "each":
            # checkpoints one layer at a time
            if len(functions) > 0:
                logger.debug(
                    rmsg(
                        f"Checkpointing {len(functions)} module(s) in sequential module individually"
                    )
                )
            for j in range(0, len(functions)):
                wrapping_obj, tensors = state.serialization_manager.serialize(
                    ((input,), {}), for_transmission=False
                )

                # unpack here so autograd fn doesnt get a tuple of tensors
                input = CheckpointTupledFunction.apply(
                    run_function(j, j, functions),
                    preserve_rng_state,
                    can_shard_activation_offloading(functions[j]),
                    wrapping_obj,
                    *tensors,
                )
        elif "group_" in strategy:
            try:
                group_by_n = int(strategy.split("group_")[1])
            except ValueError:
                raise CheckpointingError(
                    f"Expected strategy to be of the form `group_x` where x is an integer. Found strategy value to be {strategy}"
                )

            if len(functions) > 0:
                logger.debug(
                    rmsg(
                        f"Checkpointing {len(functions)} module(s) in sequential module {group_by_n} at a time"
                    )
                )

            for j in range(0, len(functions), group_by_n):
                # unpack here so autograd fn doesnt get a tuple of tensors
                start = j
                end = j + group_by_n - 1  # -1 as below run_function expects both inclusive indices
                if end > len(functions) - 1:
                    end = len(functions) - 1

                if end >= start:
                    wrapping_obj, tensors = state.serialization_manager.serialize(
                        ((input,), {}), for_transmission=False
                    )

                    input = CheckpointTupledFunction.apply(
                        run_function(start, end, functions),
                        preserve_rng_state,
                        can_shard_activation_offloading(functions[start]),
                        wrapping_obj,
                        *tensors,
                    )
                else:
                    break
        else:
            raise CheckpointingError(f"Invalid strategy: {strategy}")

        outputs = input

        last_module = self[end_index - 1]
        if not state.module_manager.is_parent_executor(last_module):
            # records output in stack, which can be popped later for backward call
            maybe_push_outputs(last_module, outputs)

        _maybe_mark_module_info(self, outputs, end_index)

        # HACK: Added assumption that when checkpointing is used, backward doesnt break
        # within a sequential chain
        # TODO: Need to revisit when adding checkpointing functionality
        # Setting num_inputs to high number so that it fails instead of hanging

        # TODO: validate that at least one tensor requires grad in `input` so backward doesn't break within sequential
        return outputs, True, num_tensors_requiring_grad


def _maybe_mark_module_info(module, outputs, end_index):
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
                        module_name, count, i, True, end_index
                    )


def execute_chain_remotely(self, start_index, end_index, input):
    check_supported(state.module_manager.get_module_name(self), input, {}, skip_tensor_check=True)
    # by now if sequential was checkpointed, state.checkpoint_activations_config will be set
    request = SequentialModulesExecutionRequest.create_forward_req(
        self,
        start_index,
        end_index,
        input,
        enable_grads=torch.is_grad_enabled(),
        enable_autocast=torch.is_autocast_enabled(),
        checkpoint_activations_config=state.checkpoint_activations_config
        if state.checkpoint_activations_config.enabled_for_module(self)
        else None,
        position=state.module_manager.output_stack_size(
            state.microbatch, self[end_index - 1], self
        ),
    )
    outputs = state.current_worker.thread_get_forward_result(request)

    # only push to stack if the output is pushed to stack on the rank
    # executing the child module. This ensures the stack size to be
    # the same on both modules
    if check_requires_grad(outputs) and torch.is_grad_enabled():
        flat_args, _ = flatten_structure(input)
        smp_child_inputs = tuple(arg for arg in flat_args if isinstance(arg, torch.Tensor))
        state.module_manager.push_output(
            state.microbatch, self[end_index - 1], self, smp_child_inputs
        )
        additional_params = set()
        for smp_child_input in smp_child_inputs:
            if isinstance(smp_child_input, torch.nn.parameter.Parameter):
                additional_params.add(smp_child_input)

        params, smp_inputs, parent_recvs = get_ancestors(
            smp_child_inputs, checkpoint_nodes_cache=state.checkpoint_nodes_cache
        )
        params = params.union(additional_params)
        if not state.cfg.zero2d_enabled():
            state.model.grad_counter.increment_expected_num_grads(
                state.microbatch, [state.model.get_param_name(p) for p in params]
            )
        logger.debug(
            rmsg(
                f"{len(params)} params are reachable from inputs of {state.module_manager.get_module_name(self[start_index])}"
            )
        )
        increment_smp_input_bwd_counts(smp_inputs)

        child_output_leaves = detach_outputs(outputs)
        count = sum([1 if x.requires_grad else 0 for x in smp_child_inputs])
        flat_leaves, structure_id = flatten_structure(child_output_leaves)

        filtered_flat_leaves = [
            (i, leaf) for i, leaf in enumerate(flat_leaves) if isinstance(leaf, torch.Tensor)
        ]
        indices = [i for i, _ in filtered_flat_leaves]
        leaves = [leaf for _, leaf in filtered_flat_leaves]

        tensor_outputs = SMPParentRecv.apply(
            self, self[end_index - 1], count, parent_recvs, True, *leaves
        )

        # copy the module info from the original tensor
        for leaf, tensor in zip(leaves, tensor_outputs):
            pass_custom_attributes(leaf, tensor)

        outputs = flat_leaves
        for i, ind in enumerate(indices):
            outputs[ind] = tensor_outputs[i]

        outputs = unflatten_structure(outputs, structure_id)
    return outputs


def sequential_distributed_forward(self, input: Any):
    if state.module_manager.is_executor(self):
        if not state.module_manager.is_main_module(
            self
        ) and not state.module_manager.is_parent_executor(self):
            input, _ = insert_smp_input_op(self, input)
            # sequential only supports one arg so unpack
            input = input[0]

        state.module_manager.record_traversal_into_module(self)
        chains = []
        for i, (name, mod) in enumerate(self._modules.items()):
            p = state.module_manager.get_partition(mod)
            if len(chains) == 0 or chains[-1][0] != p:
                chains.append((p, i))
        # now chains holds entries of form (partition, start_index)
        # where for a given entry,
        # partition is the executor for modules from start_index to the next entry's start_index
        for idx, chain in enumerate(chains):
            executor, start_index = chain
            if idx + 1 < len(chains):
                # start index of next chain
                end_index = chains[idx + 1][1]
            else:
                end_index = len(self)

            if executor == pp_rank():
                input, _, _ = execute_chain_maybe_with_checkpointing(
                    self, self.execute_chain, start_index, end_index, input
                )
            else:
                input = execute_chain_maybe_with_checkpointing(
                    self, self.execute_chain_remotely, start_index, end_index, input
                )
                # TODO skip send to this executor when moving from one chain to next both on other ranks

        outputs = input
        if not state.module_manager.is_parent_executor(self):
            maybe_push_outputs(self, outputs)
        state.module_manager.finished_module_exec()
        return outputs
    elif state.module_manager.is_parent_executor(self):
        return execute_as_parent(self, input)
    else:
        raise InvalidExecutorError(
            state.module_manager.get_module_name(self), state.module_manager.get_partition(self)
        )
