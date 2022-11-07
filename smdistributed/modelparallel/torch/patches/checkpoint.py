# Third Party
# Standard Library
import collections
import warnings
from functools import lru_cache
from typing import Dict, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    check_backward_validity,
    detach_variable,
    get_device_states,
    set_device_states,
)

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.exceptions import (
    CheckpointingError,
    InvalidReturnTypeFromCheckpointedModuleError,
)
from smdistributed.modelparallel.torch.graph_utils import get_ancestors
from smdistributed.modelparallel.torch.utils import map_structure, rmsg

logger = get_logger()


class CheckpointNodesCache:
    def __init__(self):
        self._params_used: Dict["_ContextMethodMixin", Set["Parameter"]] = collections.defaultdict(
            set
        )
        self._input_nodes_used: Dict[
            "_ContextMethodMixin", Set["_ContextMethodMixin"]
        ] = collections.defaultdict(set)
        self._parent_recv_nodes_used: Dict[
            "_ContextMethodMixin", Set["_ContextMethodMixin"]
        ] = collections.defaultdict(set)

    def reset(self):
        self._params_used.clear()
        self._input_nodes_used.clear()
        self._parent_recv_nodes_used.clear()

    def add(self, node, params, inputs, parent_recvs):
        self._params_used[node] = params
        self._input_nodes_used[node] = inputs
        self._parent_recv_nodes_used[node] = parent_recvs

    def get_params(self, node):
        return self._params_used[node]

    def get_inputs(self, node):
        return self._input_nodes_used[node]

    def get_parent_recvs(self, node):
        return self._parent_recv_nodes_used[node]


class CheckpointConfig:
    def __init__(self, enabled=False, preserve_rng_state=True, module_name=None, strategy="each"):
        self.enabled = enabled
        self.preserve_rng_state = preserve_rng_state
        self.module_name = module_name
        self.strategy = strategy

    def enabled_for_module(self, module):
        from smdistributed.modelparallel.torch.state_mod import state

        return self.enabled and state.module_manager.get_module_name(module) == self.module_name

    def reset(self):
        self.enabled = False
        self.module_name = None
        self.preserve_rng_state = True
        self.strategy = "each"

    def __hash__(self):
        return (
            hash(self.enabled)
            + hash(self.preserve_rng_state)
            + hash(self.module_name)
            + hash(self.strategy)
        )

    def __eq__(self, other):
        return (
            self.enabled == other.enabled
            and self.preserve_rng_state == other.preserve_rng_state
            and self.module_name == other.module_name
            and self.strategy == other.strategy
        )

    def __str__(self):
        if self.enabled:
            options = ""
            if self.preserve_rng_state:
                options += f"preserve_rng_state,"
            if self.strategy:
                options += f"strategy: {self.strategy}"

            return f"enabled: {options}"
        else:
            return "disabled"


class CheckpointTupledFunction(torch.autograd.Function):
    """
    This is similar to the checkpoint function in torch.utils.checkpoint with a couple of changes.
    1. It supports arbitary Python structures as arguments, as well as keyword arguments and variable-length arguments
    2. It restores execution stack during backward so we can execute fwd pass there.
    3. Instead of executing forward pass in no_grad context, this function runs the forward pass with enable_grad context
     and detaches the outputs after collecting information about ancestors for backward pass bookkeeping and overlapping allreduce.

    """

    @staticmethod
    def forward(
        ctx, run_function, preserve_rng_state, shard_activation_offloading, wrapping_obj, *args
    ):
        from smdistributed.modelparallel.torch.state_mod import state

        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.shard_activation_offloading = shard_activation_offloading
        ctx.seed = state.rng_manager.get_state()
        ctx.wrapping_obj = wrapping_obj
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

        if state.cfg.offload_activations:
            ctx.task, ctx.item_id = state.current_offloader().save_for_backward(
                shard_activation_offloading, *args
            )
        else:
            ctx.save_for_backward(*args)

        ctx.exec_stack = state.module_manager.execution_stack

        with torch.enable_grad():
            _args, kwargs = state.serialization_manager.deserialize(wrapping_obj, args)
            outputs = run_function(*_args, **kwargs)
            if isinstance(outputs, (dict, set)):
                raise InvalidReturnTypeFromCheckpointedModuleError(
                    "The return value from the forward method of a checkpointed module cannot be of type dict or set."
                )

        if state.model:
            # there can't be another checkpoint within this fwd pass subgraph so dont need to pass
            # checkpoint_nodes_cache to get_ancestors
            params, input_nodes, parent_recv_nodes = get_ancestors(outputs, upto_inputs=args)
            # this cache serves as a shortcut when traversing backward graph later
            state.checkpoint_nodes_cache.add(ctx, params, input_nodes, parent_recv_nodes)
            logger.debug(rmsg(f"{len(params)} params are reachable from output of checkpointed fn"))

            if not state.cfg.zero2d_enabled():
                state.model.grad_counter.increment_expected_num_grads(
                    state.microbatch, [state.model.get_param_name(p) for p in params]
                )

        def _detach(x):
            if isinstance(x, torch.Tensor):
                out = x.detach()
                out.requires_grad = x.requires_grad
                return out
            return x

        return map_structure(_detach, outputs)

    @staticmethod
    def backward(ctx, *args):
        from smdistributed.modelparallel.torch.state_mod import state

        if not torch.autograd._is_checkpoint_valid():
            raise CheckpointingError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )

        if state.cfg.offload_activations:
            inputs = state.current_offloader().saved_tensors(
                ctx.shard_activation_offloading, ctx.task, ctx.item_id
            )
        else:
            inputs = ctx.saved_tensors

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices

        # set the smp's rng seed as the forward one temporarily during the second forward pass
        with state.fork_smp_rng_state(seed=ctx.seed, enabled=ctx.preserve_rng_state):
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

                # Compute the forward pass.
                detached_inputs = detach_variable(tuple(inputs))
                state.module_manager.execution_stack = ctx.exec_stack

                # we dont need to disable backward counting here
                # as this backward will be only within a single partition
                # this checkpoint function is assumed to be for a module within a single partition
                with state.rerunning_fwd_on_checkpointed_fn(), torch.enable_grad():
                    _args, kwargs = state.serialization_manager.deserialize(
                        ctx.wrapping_obj, detached_inputs
                    )
                    outputs = ctx.run_function(*_args, **kwargs)

            _, output_list = state.serialization_manager.serialize(outputs, for_transmission=False)

            filtered_outps = []
            filtered_args = []
            ct = 0
            for i in range(len(output_list)):
                if output_list[i].requires_grad:
                    filtered_outps.append(output_list[i])
                    if ct == len(args):
                        raise InvalidReturnTypeFromCheckpointedModuleError(
                            "No gradient received for one of the outputs of a checkpointed module that require grad. A common cause for this is that one of the output tensors is an ancestor of another. If you are using HuggingFace transformers, try setting use_cache=False, and avoid returning internal tensors from checkpointed modules."
                        )

                    filtered_args.append(args[ct])
                    ct += 1

            torch.autograd.backward(tuple(filtered_outps), tuple(filtered_args))
            grads = tuple(
                inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs
            )
        return (None, None, None, None) + grads


@lru_cache(maxsize=2048)
def validate_checkpointable(module):
    from smdistributed.modelparallel.torch.state_mod import state

    if not isinstance(module, nn.Module):
        raise CheckpointingError(f"{module} is not a Module")
    if state.model:
        # Check if all sub-modules are in the same partition
        if not state.module_manager.check_module_partition(module):
            message = (
                f"The checkpointed module {state.module_manager.get_module_name(module)} has child "
                + "modules which are not on the same partition as the module. "
                + "Only modules which are fully in one partition can be checkpointed."
            )
            if state.cfg.auto_partition:
                message += " Skipping activation checkpointing for this module."
                warnings.warn(message)
                return False
            else:
                raise CheckpointingError(message)
    return True


def validate_checkpointable_sequential(sequential_module):
    from smdistributed.modelparallel.torch.state_mod import state

    if not isinstance(sequential_module, nn.Sequential):
        raise CheckpointingError(
            f"`checkpoint_sequential` can only be used with nn.Sequential modules. Passed module is of type {type(sequential_module)}."
        )
    if state.model:
        # Check if children of the sub-modules are in the same partition
        for module in sequential_module.children():
            if not state.module_manager.check_module_partition(module):
                message = (
                    f"The checkpointed sequential module {state.module_manager.get_module_name(sequential_module)} "
                    + f"has a child module {state.module_manager.get_module_name(module)} which does not have all its "
                    + "sub modules on the same partition as the child module. Sequential module can only "
                    + "be checkpointed if each of its child modules have all sub modules on the same "
                    + "partition as itself."
                )
                if state.cfg.auto_partition:
                    message += " Skipping activation checkpointing for this module."
                    warnings.warn(message)
                    return False
                else:
                    raise CheckpointingError(message)
    return True


def checkpoint_function(fn, args, kwargs, module_name, preserve_rng_state=True, strategy="each"):
    from smdistributed.modelparallel.torch.state_mod import state

    with state.enable_activation_checkpoints(
        CheckpointConfig(
            enabled=True,
            preserve_rng_state=preserve_rng_state,
            strategy=strategy,
            module_name=module_name,
        )
    ):
        return fn(*args, **kwargs)


def checkpoint(module, *args, preserve_rng_state=True, **kwargs):
    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing() and validate_checkpointable(module):
        return checkpoint_function(
            module.forward,
            args,
            kwargs,
            module_name=state.module_manager.get_module_name(module),
            preserve_rng_state=preserve_rng_state,
        )
    else:
        return module.forward(*args)


def checkpoint_sequential(
    sequential_module: nn.Sequential,
    input: Union[torch.Tensor, Tuple[torch.Tensor]],
    strategy: str = "each",
    preserve_rng_state: bool = True,
    pack_args_as_tuple: bool = False,
):
    from smdistributed.modelparallel.torch.state_mod import state

    if pack_args_as_tuple:
        logger.warning(
            "pack_args_as_tuple argument is deprecated, and will be removed in a future version of smp. This argument is a no-op and is not required."
        )

    if validate_checkpointable_sequential(sequential_module):
        return checkpoint_function(
            sequential_module.forward,
            (input,),
            {},
            module_name=state.module_manager.get_module_name(sequential_module),
            strategy=strategy,
            preserve_rng_state=preserve_rng_state,
        )
        # else:
        #     return checkpoint_function(
        #         sequential_module.forward,
        #         input,
        #         module_name=state.module_manager.get_module_name(sequential_module),
        #         strategy=strategy,
        #         preserve_rng_state=preserve_rng_state,
        #     )
    else:
        sequential_module.forward(*input)
