# Standard Library
from distutils.version import LooseVersion
from functools import partial

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.comm import get_dp_process_group, get_rdp_process_group
from smdistributed.modelparallel.torch.core import (
    core,
    dp_rank,
    dp_size,
    rank,
    rdp_rank,
    rdp_size,
    tp_rank,
    tp_size,
)
from smdistributed.modelparallel.torch.ddp_model import ReducerType
from smdistributed.modelparallel.torch.nn.utils import get_local_channels, get_start_pos_for_slicing
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import collect_and_merge, get_distribution_axis, rmsg

logger = get_logger()


def _param_id_to_index(self):
    """Map the parameter id into the global index defined in nn.optimizer.state_dict"""
    param_mappings = {}
    start_index = 0

    # Adopt from https://github.com/pytorch/pytorch/blob/v1.6.0/torch/optim/optimizer.py#L87
    def update_group_params_index(group):
        nonlocal start_index
        param_mappings.update(
            {
                id(p): i
                for i, p in enumerate(group["params"], start_index)
                if id(p) not in param_mappings
            }
        )
        start_index += len(group["params"])

    for g in self.param_groups:
        update_group_params_index(g)
    return param_mappings


def _param_index_to_param_local(self):
    """Map the parameter index to the parameter,
    dictionary will only contain local parameters"""
    param_id_to_index = self._param_id_to_index()
    param_index_to_param = {}

    if not state.model:
        return param_index_to_param

    if self.redefined_params:
        param_gen = state.model.virtual_named_parameters()
    else:
        param_gen = state.model.named_parameters()

    for _, param in param_gen:
        param_id = id(param)
        if param_id in param_id_to_index:
            param_index_to_param[param_id_to_index[param_id]] = param

    return param_index_to_param


def param_name_to_index(self):
    """Map the parameter name into the global index"""
    param_id_to_index = self._param_id_to_index()
    name_to_index = {}
    if self.redefined_params:
        param_gen = state.model.virtual_named_parameters()
    else:
        param_gen = state.model.named_parameters()
    for name, param in param_gen:
        param_id = id(param)
        if param_id in param_id_to_index:
            name_to_index[name] = param_id_to_index[param_id]
        else:
            logger.warning(
                f"parameter {name} is missing when loading optimizer's state_dict, skip."
            )
    return name_to_index


def param_index_to_name(self):
    n2i = self.param_name_to_index()
    index_to_name = {}
    for n, i in n2i.items():
        index_to_name[i] = n
    return index_to_name


def local_state_dict(self, cast_to_cpu=True, gather_if_shard=True):
    """
    Return the state_dict that only contains the parameter states belong to this device
    Args:
        cast_to_cpu: bool, set to True so that all state will be stored on CPU
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

    return full_local_dict


def state_dict(self, cast_to_cpu=True):
    """
    Return the global state_dict that contains parameter states from all devices
    Args:
        cast_to_cpu: bool, set to True so that all state will be stored on CPU
    """
    local_dict = self.local_state_dict(cast_to_cpu=cast_to_cpu)

    if state.cfg.shard_optimizer_state:
        # doing this, because for a large (~200B) model, gathering the full optimizer states at cpu will
        # consume the entire CPU memory, even on a p4d
        logger.warning(
            "Since shard_optimizer_state is enabled, only returning local state_dict from state_dict() call on smp.DistributedOptimizer."
        )
        return local_dict

    if tp_size() > 1:
        from smdistributed.modelparallel.torch.comm import CommGroup, broadcast, recv_from, RankType

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
    return {"state": full_state, "param_groups": local_dict["param_groups"]}


def _actual_load_state_dict(self, state_dict):
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
            if not isinstance(param_state, torch.Tensor):
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


def load_state_dict(self, state_dict):
    """
    Load the state_dict. This function first check if the state_dict contains full states,
    if no, will allgather the states to create a full state_dict.
    This function does not load the state_dict into the optimizer. The real loading happens
    after the partition is finished, so each rank knows which parameters' states to load.
    """

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

    if state.model.partitioned:
        logger.debug(rmsg(f"Model already partitioned, loading state dict"))
        self._actual_load_state_dict(state_dict)
    else:

        def _actual_load_callable(model, optimizer):
            logger.debug(rmsg(f"Loading state dict after partitioning"))
            optimizer._actual_load_state_dict(state_dict)

        state.model.register_post_partition_hook(_actual_load_callable)


def validate_sharding_config():
    if not state.cfg.ddp:
        raise ValueError("Optimizer state sharding is only supported when DDP is enabled.")

    if not state.model.overlapping_allreduce:
        raise ValueError(
            "Optimizer state sharding is currently only supported for overlapping allreduce."
        )


def zero_grad(self, *args, **kwargs):
    self.orig_zero_grad(*args, **kwargs)

    if state.cfg.shard_optimizer_state:
        validate_sharding_config()
        for reducer, _ in state.model.module._iterate_reducer_and_pg():
            reducer.zero_grad_buffer()


def step(self):
    if state.cfg.shard_optimizer_state:
        validate_sharding_config()

    step_out = self.orig_step()

    if state.cfg.shard_optimizer_state:
        # use allgather_base for better memory efficiency and performance
        use_allgather_base = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

        for red_type in state.model.iterate_reducer_types():
            rank = dp_rank() if red_type == ReducerType.DEFAULT else rdp_rank()
            size = dp_size() if red_type == ReducerType.DEFAULT else rdp_size()
            group = (
                get_dp_process_group()
                if red_type == ReducerType.DEFAULT
                else get_rdp_process_group()
            )

            group_offset = state.model.group_offset[red_type]
            slice_size = state.model.group_size[red_type] // size

            buf = state.model.param_buffer
            if use_allgather_base:
                group_buf = buf.narrow(0, group_offset, slice_size * size)
            group_slices = [
                buf.narrow(0, group_offset + slice_size * r, slice_size) for r in range(size)
            ]

            with torch.no_grad():
                if use_allgather_base:
                    torch.distributed._all_gather_base(group_buf, group_slices[rank], group=group)
                else:
                    torch.distributed.all_gather(group_slices, group_slices[rank], group=group)

    return step_out


def _patch_method(optimizer, method_name, function):
    func = partial(function, optimizer)
    setattr(func, "__self__", optimizer)
    setattr(func, "__func__", function)
    setattr(optimizer, method_name, func)


def DistributedOptimizer(opt):
    """
    The DistributedOptimizer wrapper. This function will return the original optimizer
    with some attributes overriden.
    """
    if not state.model:
        raise RuntimeError(
            "smp.DistributedModel call must take place before smp.DistributedOptimizer."
        )

    model_params = {p for p in state.model.parameters()}
    for param_group in opt.param_groups:
        for param in param_group["params"]:
            if param not in model_params:
                raise RuntimeError(
                    f"Found parameter in optimizer that is not part of smp.DistributedModel, which might lead to incorrect results. This might be because the optimizer is initialized with the model parameters *before* calling the smp.DistributedModel wrapper. Please make sure to initialize the optimizer with the parameters of the smp.DistributedModel object."
                )

    opt.orig_load_state_dict = opt.load_state_dict
    opt.orig_state_dict = opt.state_dict
    opt.orig_step = opt.step
    opt.orig_zero_grad = opt.zero_grad

    opt._param_id_to_index = partial(_param_id_to_index, opt)
    opt.param_name_to_index = partial(param_name_to_index, opt)
    opt.param_index_to_name = partial(param_index_to_name, opt)

    opt.local_state_dict = partial(local_state_dict, opt)
    opt.state_dict = partial(state_dict, opt)

    opt._actual_load_state_dict = partial(_actual_load_state_dict, opt)
    opt.load_state_dict = partial(load_state_dict, opt)
    opt._param_index_to_param_local = partial(_param_index_to_param_local, opt)
    opt.zero_grad = partial(zero_grad, opt)
    _patch_method(opt, "step", step)

    state.optimizer = opt
    opt.redefined_params = False
    opt.index_to_name = opt.param_index_to_name()
    return opt
