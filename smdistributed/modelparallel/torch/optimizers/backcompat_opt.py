# Standard Library
import copy

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.comm import (
    CommGroup,
    RankType,
    broadcast,
    get_dp_process_group,
    get_rdp_process_group,
    recv_from,
    send,
)
from smdistributed.modelparallel.torch.core import (
    core,
    dp_rank,
    dp_size,
    pp_size,
    rank,
    rdp_rank,
    rdp_size,
    tp_rank,
    tp_size,
)
from smdistributed.modelparallel.torch.fp16.fp16util import (
    get_pp_merged_fp32_from_fp16_param_groups,
    get_tp_merged_fp32_from_fp16_param_groups,
    master_params_to_model_params,
    model_grads_to_master_grads,
    model_params_to_master_params,
    register_optimizer_hooks,
)
from smdistributed.modelparallel.torch.nn.utils import get_local_channels, get_start_pos_for_slicing
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import collect_and_merge, get_distribution_axis, rmsg

logger = get_logger()


def _local_state_dict(self, cast_to_cpu=True, gather_if_shard=True, fp32_states_only=False):
    """
    Return the state_dict that only contains the parameter states belong to this device, also used for getting final states for saving optimizer
    Args:
        cast_to_cpu: bool, set to True so that all state will be stored on CPU
        fp32_states_only: bool, set to True to only get the parameter states for fp32 states, or we only
        want to get optimizer_state_dict for fp16 optimizer rather than save checkpoint.
    (WARNING) Be aware of that parameter id might change before/after partition when torchdistx is used for delayed param init
    since before partition all parameters are fake tensors
    """

    local_dict = self.orig_state_dict()
    param_index_to_param = self._param_index_to_param_local()
    param_index_to_name_tp_group = state.param_index_to_name_tp_group
    param_name_to_index = self.param_name_to_index()
    for param_idx, opt_state in local_dict["state"].items():
        # Obtain param name corresponding to param_idx
        param_name = param_index_to_name_tp_group[tp_rank()][param_idx]
        # Obtain local_index from the param name
        local_index = param_name_to_index[param_name]
        param = param_index_to_param.get(local_index, None)
        # If its not a distributed param, we can set the state to be empty. This will reduce the size of the serialized file.
        # During load, the state corresponding to non distributed param will be obtained
        # from tp_rank 0. For shard_optimizer_state=True there will be no allgather during loading, so we still need to keep non-distributed params
        if (
            get_distribution_axis(param) is None
            and tp_rank() != 0
            and not state.cfg.shard_optimizer_state
        ):
            local_dict["state"][param_idx] = {}
            continue
        cpu_opt_state = {}
        if cast_to_cpu:
            for name, tensor in opt_state.items():
                if isinstance(tensor, torch.Tensor):
                    cpu_opt_state[name] = tensor.cpu()
                else:
                    cpu_opt_state[name] = tensor
            local_dict["state"][param_idx] = cpu_opt_state
    if state.cfg.shard_optimizer_state and gather_if_shard:
        from smdistributed.modelparallel.torch.comm import gather, RDP_GROUP

        msg = "rdp" if tp_size() > 1 else "dp"
        logger.info(
            rmsg(
                f"{msg} rank 0 is gathering optimizer state_dict from other {msg} ranks with optimizer state sharding. To prevent hangs, please ensure optimizer.local_state_dict() "
                f"(where optimizer is the object returned by the DistributedOptimizer wrapper) "
                f"is called on all the ranks. Only {msg} rank 0 will contain full optimizer state dict of that rdp group."
            )
        )
        gathered_local_dict = gather(local_dict, RDP_GROUP, rank=0)
        if rdp_rank() == 0:
            full_local_dict = {
                "state": [dic["state"] for dic in gathered_local_dict],
                "param_groups": [dic["param_groups"] for dic in gathered_local_dict],
            }
        else:
            full_local_dict = local_dict
    else:
        full_local_dict = local_dict

    full_local_dict["_smp_is_partial"] = True

    # process the stats for saving
    if state.cfg.fp16 and not fp32_states_only:
        logger.info(rmsg(f"saving fp16 optimizer locally on rank {rank()}"))
        optimizer_state_dict, cpu_fp32_from_fp16_groups = _prepare_fp16_state_dict(self)
        optimizer_state_dict["optimizer_state_dict"] = full_local_dict
        if state.cfg.shard_optimizer_state and gather_if_shard:
            if rdp_rank() == 0:
                print(
                    "With shard_optimizer_state=True, gather full fp32_from_fp16_groups for the rdp_group on rdp rank 0"
                )
                gathered_cpu_fp32_from_fp16_groups = [cpu_fp32_from_fp16_groups]
                for src in range(1, rdp_size()):
                    gathered_cpu_fp32_from_fp16_groups.append(recv_from(src, RankType.RDP_RANK))
                optimizer_state_dict["fp32_from_fp16"] = gathered_cpu_fp32_from_fp16_groups
            else:
                send(cpu_fp32_from_fp16_groups, 0, RankType.RDP_RANK)
                optimizer_state_dict["fp32_from_fp16"] = cpu_fp32_from_fp16_groups
        else:
            optimizer_state_dict["fp32_from_fp16"] = cpu_fp32_from_fp16_groups
        if pp_size() > 1:
            print(
                "WARNING: Ensure that partition decision doesnt change between runs (you can ensure this by setting use_times=False in smp config)."
                "If you want to save and load with partition decision changing between runs, use full save and load instead."
            )
        full_local_dict = optimizer_state_dict

    return full_local_dict


def _prepare_fp16_state_dict(optimizer):
    optimizer_state_dict = {}
    loss_scaler = optimizer.loss_scaler
    _model = loss_scaler.model
    loss_scaler.model = None
    _loss_scaler = copy.deepcopy(loss_scaler)
    loss_scaler.model = _model
    optimizer_state_dict["loss_scaler"] = _loss_scaler
    optimizer_state_dict["dynamic_loss_scale"] = optimizer.dynamic_loss_scale
    optimizer_state_dict["overflow"] = optimizer.overflow
    optimizer_state_dict["first_closure_call_this_step"] = optimizer.first_closure_call_this_step
    cpu_fp32_from_fp16_groups = [
        [param.cpu() for param in group] for group in optimizer.fp32_from_fp16_groups
    ]
    if optimizer.master_params_created:
        register_optimizer_hooks(state.model)
    return optimizer_state_dict, cpu_fp32_from_fp16_groups


def _state_dict(self, cast_to_cpu=True, fp32_states_only=False):  # pragma: no cover
    local_dict = self.local_state_dict(cast_to_cpu=cast_to_cpu, fp32_states_only=True)

    if state.cfg.shard_optimizer_state:
        # doing this, because for a large (~200B) model, gathering the full optimizer states at cpu will
        # consume the entire CPU memory, even on a p4d
        logger.warning(
            "Since shard_optimizer_state is enabled, only returning local state_dict from state_dict() call on smp.DistributedOptimizer."
        )
        return local_dict

    if tp_size() > 1:

        out = None
        # tp_rank() == 0, has all the params: 1. for params which are distributed
        # it has a slice, 2. single rank params are guaranteed to be on tp rank 0,
        # 3. Also, non distributed params are on all tp_ranks
        # broadcast optimizer state, so that param_groups consider the full set
        # of params (present in tp_rank 0) instead of partial params
        if tp_rank() == 0:
            out = local_dict["param_groups"]
            broadcast(out, CommGroup.TP_GROUP)
        else:
            out = recv_from(0, RankType.TP_RANK)
            local_dict["param_groups"] = out

    del local_dict["_smp_is_partial"]

    msg = "rdp_rank()" if tp_size() > 1 else "dp_rank()"
    logger.info(
        rmsg(
            f"Gathering optimizer state_dict from different ranks. To prevent hangs, please ensure optimizer.state_dict() "
            f"(where optimizer is the object returned by the DistributedOptimizer wrapper) "
            f"is called on all the ranks with {msg} == 0"
        )
    )
    full_state = collect_and_merge(local_dict["state"], optimizer=True)

    full_global_state = {"state": full_state, "param_groups": local_dict["param_groups"]}

    if state.cfg.fp16 and not fp32_states_only:
        if rank() == 0:
            logger.info(rmsg("saving fp16 optimizer"))
        optimizer_state_dict, cpu_fp32_from_fp16_groups = _prepare_fp16_state_dict(self)
        optimizer_state_dict["optimizer_state_dict"] = full_global_state
        if tp_size() > 1 and not state.cfg.shard_optimizer_state:
            tp_merged_fp32_from_fp16_groups, param_name_groups = get_tp_merged_fp32_from_fp16_param_groups(
                state.optimizer, cpu_fp32_from_fp16_groups
            )
            pp_merged_fp32_from_fp16_groups, param_name_groups = get_pp_merged_fp32_from_fp16_param_groups(
                state.optimizer, tp_merged_fp32_from_fp16_groups, param_name_groups
            )
        else:
            raise ValueError(
                "Loading full optimizer state is not supported, when TP is not enabled or shard_optimizer_state is enabled"
            )
        optimizer_state_dict["fp32_from_fp16"] = pp_merged_fp32_from_fp16_groups
        optimizer_state_dict["param_name_groups"] = param_name_groups
        full_global_state = optimizer_state_dict

    return full_global_state


def _load_state_dict(self, state_dict, gather_if_shard=True):
    """
    Load the state_dict. This function first check if the state_dict contains full states,
    if no, will allgather the states to create a full state_dict.
    This function does not load the state_dict into the optimizer. The real loading happens
    after the partition is finished, so each rank knows which parameters' states to load.
    """

    if state.cfg.fp16:
        self.loss_scaler = state_dict["loss_scaler"]
        self.loss_scaler.model = state.model
        self.dynamic_loss_scale = state_dict["dynamic_loss_scale"]
        self.overflow = state_dict["overflow"]
        self.first_closure_call_this_step = state_dict["first_closure_call_this_step"]

    def load_hook_fn(mod, opt):
        nonlocal state_dict
        opt_state_dict = None

        if state.cfg.fp16:
            if rank() == 0:
                logger.info(rmsg("loading fp16 optimizer"))
            opt_state_dict = state_dict
            state_dict = opt_state_dict["optimizer_state_dict"]
            if self.master_params_created:
                register_optimizer_hooks(state.model)

        is_partial = "_smp_is_partial" in state_dict and state_dict["_smp_is_partial"]
        if "_smp_is_partial" in state_dict:
            del state_dict["_smp_is_partial"]

        if is_partial and not state.cfg.shard_optimizer_state:
            msg = "rdp_rank()" if tp_size() > 1 else "dp_rank()"
            logger.info(
                rmsg(
                    f"Gathering optimizer state_dict from different ranks. To prevent hangs, please ensure optimizer.load_state_dict() "
                    f"(where optimizer is the object returned by the DistributedOptimizer wrapper) "
                    f"is called on all the ranks with {msg} == 0"
                )
            )
            logger.debug(rmsg(f"Gathering optimizer state_dict during loading"))
            full_state = collect_and_merge(state_dict["state"], optimizer=True)
            state_dict = {"state": full_state, "param_groups": state_dict["param_groups"]}

        if state.cfg.fp16:
            opt_state_dict["optimizer_state_dict"] = state_dict
            state_dict = opt_state_dict

        if state.model.partitioned:
            logger.debug(rmsg(f"Model already partitioned, loading state dict"))
            self._actual_load_state_dict(state_dict, is_partial, gather_if_shard)
        else:

            def _actual_load_callable(model, optimizer, is_partial):
                logger.debug(rmsg(f"Loading state dict after partitioning"))
                optimizer._actual_load_state_dict(state_dict, is_partial, gather_if_shard)

            state.model.register_post_partition_hook(_actual_load_callable)

    state.model.register_post_step_hook(load_hook_fn)


def _actual_load_state_dict(self, state_dict, partial=False, gather_if_shard=True):

    opt_state_dict = None

    if state.cfg.fp16:
        opt_state_dict = state_dict
        state_dict = opt_state_dict["optimizer_state_dict"]

    param_name_to_index = self.param_name_to_index()
    local_opt_state = {}
    local_param_names = [x[0] for x in state.model.local_named_parameters()]
    param_index_to_param = self._param_index_to_param_local()
    param_name_to_index_tp_group = state.param_name_to_index_tp_group

    def maybe_slice_state(idx, param_state_dict):
        param = param_index_to_param.get(idx, None)
        weight_split_shapes = state.module_manager.weight_split_shapes.get(param, None)
        axis = get_distribution_axis(param)
        for state_name in param_state_dict:
            param_state = param_state_dict[state_name]
            if not (isinstance(param_state, torch.Tensor) and param_state.shape != torch.Size([])):
                continue
            if axis is not None:
                slice_size = (
                    get_local_channels(param_state.size(axis))
                    if weight_split_shapes == None
                    else weight_split_shapes[tp_rank()]
                )
                start = (
                    get_start_pos_for_slicing(param_state.size(axis))
                    if weight_split_shapes == None
                    else sum(weight_split_shapes[: tp_rank()])
                )
                param_state_dict[state_name] = torch.narrow(
                    param_state, axis, start, slice_size
                ).contiguous()
        return param_state_dict

    if state.cfg.shard_optimizer_state:
        if not isinstance(state_dict["state"], list):
            local_state_dict = {
                "state": state_dict["state"],
                "param_groups": state_dict["param_groups"],
            }
        else:
            assert rdp_size() == len(
                state_dict["state"]
            ), f"Loading with shard_optimizer_state=True must use the checkpoint that was trained with the same rdp size! The training rdp size {len(state_dict['state'])}, the loading rdp size {rdp_size()}"
            local_state_dict = {
                "state": state_dict["state"][rdp_rank()],
                "param_groups": state_dict["param_groups"][rdp_rank()],
            }
    else:
        global_to_local_idx_map = {}
        for name in local_param_names:
            if name in param_name_to_index:
                idx = param_name_to_index[name]
                # assumes non distributed params only on tp_rank 0
                global_idx = param_name_to_index_tp_group[0][name]
                global_to_local_idx_map[global_idx] = idx
                if idx in state_dict["state"]:
                    local_opt_state[idx] = maybe_slice_state(idx, state_dict["state"][global_idx])
                else:
                    logger.warning(
                        rmsg(
                            f"Optimizer state for the param {name} was not loaded as it was not part of the optimizer state"
                        )
                    )
        local_state_dict = {"state": local_opt_state, "param_groups": state_dict["param_groups"]}
    self.orig_load_state_dict(local_state_dict)
    log = rmsg("Loaded optimizer state_dict")
    logger.info(log) if rank() == 0 else logger.debug(log)

    if state.cfg.fp16:
        if partial:
            if state.cfg.shard_optimizer_state and gather_if_shard > 0:
                self.fp32_from_fp16 = opt_state_dict["fp32_from_fp16"][rdp_rank()]
            else:
                self.fp32_from_fp16 = opt_state_dict["fp32_from_fp16"]

            for current_group, saved_group in zip(self.fp32_from_fp16_groups, self.fp32_from_fp16):
                for current, saved in zip(current_group, saved_group):
                    current.data.copy_(saved.data)

        else:
            self.fp32_from_fp16 = opt_state_dict["fp32_from_fp16"]
            param_name_groups = opt_state_dict["param_name_groups"]
            param_id_to_index = self._param_id_to_index()
            param_index_to_name_tp_group = state.param_index_to_name_tp_group
            param_index_to_name = param_index_to_name_tp_group[tp_rank()]
            for group_idx, (current_group, saved_group) in enumerate(
                zip(self.fp32_from_fp16_groups, self.fp32_from_fp16)
            ):
                for current in current_group:
                    param_id = id(current)
                    param_index = param_id_to_index[param_id]
                    param_name = param_index_to_name[param_index]
                    arr_index = param_name_groups[group_idx][param_name]
                    saved = saved_group[arr_index]
                    if self.master_distribution_axis[param_id] is not None:
                        axis = self.master_distribution_axis[param_id]
                        slice_size = saved.size(axis) // tp_size()
                        saved = torch.narrow(
                            saved.data, axis, slice_size * tp_rank(), slice_size
                        ).contiguous()
                    else:
                        saved = saved.data
                    current.data.copy_(saved)
