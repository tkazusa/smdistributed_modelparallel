# Standard Library
import copy
import inspect
import os
from distutils.version import LooseVersion
from functools import partial

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.comm import (
    get_dp_process_group,
    get_rdp_process_group,
    recv_from,
    send,
)
from smdistributed.modelparallel.torch.core import (
    core,
    dp_rank,
    dp_size,
    param_shard_rank,
    param_shard_size,
    pp_size,
    rank,
    rdp_rank,
    rdp_size,
    tp_rank,
    tp_size,
)
from smdistributed.modelparallel.torch.ddp_model import ReducerType
from smdistributed.modelparallel.torch.exceptions import SMPCheckpointError, SMPValidationError
from smdistributed.modelparallel.torch.fp16.fp16 import Bit16_Optimizer
from smdistributed.modelparallel.torch.fp16.fp16util import register_optimizer_hooks
from smdistributed.modelparallel.torch.optimizers.backcompat_opt import (
    _actual_load_state_dict,
    _load_state_dict,
    _local_state_dict,
    _state_dict,
)
from smdistributed.modelparallel.torch.patches.execution import distributed_backward
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import rmsg

try:
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
except:
    pass

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


def local_state_dict(self, cast_to_cpu=True, gather_if_shard=True, fp32_states_only=False):
    """
    Deprecated! Only used for back compatible saving.
    Return the state_dict that only contains the parameter states belong to this device, also used for getting final states for saving optimizer
    Args:
        cast_to_cpu: bool, set to True so that all state will be stored on CPU
        fp32_states_only: bool, set to True to only get the parameter states for fp32 states, or we only
        want to get optimizer_state_dict for fp16 optimizer rather than save checkpoint.
    (WARNING) Be aware of that parameter id might change before/after partition when torchdistx is used for delayed param init
    since before partition all parameters are fake tensors
    """
    logger.warning(
        f"DistributedOptimizer.local_state_dict() will be deprecated soon. \
        For smp >=1.10 please use smp.save_checkpoint to save a checkpoint and smp.resume_from_checkpoint to load a checkpoint. \
        You can also use DistributedOptimizer.local_optimizer_state_dict to get the local optimizer states \
        and DistributedOptimizer.local_fp16_state_dict to get the local fp16 states and master parameters"
    )
    return _local_state_dict(
        self,
        cast_to_cpu=cast_to_cpu,
        gather_if_shard=gather_if_shard,
        fp32_states_only=fp32_states_only,
    )


def local_optimizer_state_dict(self, cast_to_cpu=True):
    """
    Return the state_dict that only contains the parameter states belong to this device
    Args:
        cast_to_cpu: bool, set to True so that all state will be stored on CPU
    (WARNING) Be aware of that parameter id might change before/after partition when torchdistx is used for delayed param init
    since before partition all parameters are fake tensors
    """

    local_dict = self.orig_state_dict()
    if cast_to_cpu:
        for param_idx, opt_state in local_dict["state"].items():
            cpu_opt_state = {}
            for name, tensor in opt_state.items():
                if isinstance(tensor, torch.Tensor) and tensor.shape != torch.Size([]):
                    cpu_opt_state[name] = tensor.cpu()
                else:
                    cpu_opt_state[name] = tensor
            local_dict["state"][param_idx] = cpu_opt_state

    local_dict["_smp_is_partial"] = True

    return local_dict


def local_fp16_state_dict(self):
    """
    Return the loss scaler and master parameters belong to this device when fp16 is enabled
    """
    if not state.cfg.fp16:
        raise SMPValidationError(
            "opt.local_fp16_state_dict can only be called when fp16 is enabled"
        )
    logger.info(rmsg(f"saving fp16 states locally on rank {rank()}"))
    fp16_states = {}
    loss_scaler = self.loss_scaler
    _model = loss_scaler.model
    loss_scaler.model = None
    _loss_scaler = copy.deepcopy(loss_scaler)
    loss_scaler.model = _model
    fp16_states["loss_scaler"] = _loss_scaler
    fp16_states["dynamic_loss_scale"] = self.dynamic_loss_scale
    fp16_states["overflow"] = self.overflow
    fp16_states["first_closure_call_this_step"] = self.first_closure_call_this_step
    cpu_fp32_from_fp16_groups = [
        [param.cpu() for param in group] for group in self.fp32_from_fp16_groups
    ]
    fp16_states["fp32_from_fp16"] = cpu_fp32_from_fp16_groups
    if self.master_params_created:
        register_optimizer_hooks(state.model)
    return fp16_states


def state_dict(self, cast_to_cpu=True, fp32_states_only=False):
    """
    Deprecated! SMP will no longer support to save full optimizer states
    Return the global state_dict that contains parameter states from all devices
    Args:
        cast_to_cpu: bool, set to True so that all state will be stored on CPU
        fp32_states_only: bool, set to True to only get the parameter states for fp32 states, or we only
        want to get optimizer_state_dict for fp16 optimizer but not save checkpoint.
    (WARNING) Be aware of that parameter id might change before/after partition when torchdistx is used for delayed param init
    since before partition all parameters are fake tensors
    """
    logger.warning(
        f"DistributedOptimizer.state_dict() will be deprecated soon. \
        Please use DistributedOptimizer.local_optimizer_state_dict() if you want to get optimizer's state_dict \
        and use DistributedOptimizer.local_fp16_state_dict() if you want to get fp16's states, which includes loss scaler and master parameters."
    )

    return _state_dict(self, cast_to_cpu=cast_to_cpu, fp32_states_only=fp32_states_only)


def check_valid_state_dict(state_dict):
    is_partial = "_smp_is_partial" in state_dict and state_dict["_smp_is_partial"]
    if not is_partial:
        raise SMPCheckpointError(
            f"DistributedOptimizer.load_state_dict does not support loading non-partial optimizer states, this is deprecated and dangerous. If you still want to load full checkpoint, try DistributedOptimizer.load_state_dict_backcompat"
        )

    if "_smp_is_partial" in state_dict:
        del state_dict["_smp_is_partial"]


def load_optimizer_checkpoint_zero2d(self, path, tag, load_sharded_optimizer_state):
    file = os.path.join(path, tag, f"optimizer_{param_shard_rank()}.pt")
    optimizer_state = torch.load(file, map_location="cpu")
    if "_smp_zero2d" not in optimizer_state or not optimizer_state["_smp_zero2d"]:
        raise SMPValidationError(
            "Error: trying to load a non-sharded checkpoint into sharded optimizer!"
        )
    del optimizer_state["_smp_zero2d"]
    # zero-2d's load_state_dict takes a list as input
    zero_sd_list = [None] * param_shard_size()
    zero_sd_list[param_shard_rank()] = optimizer_state
    self.orig_load_state_dict(zero_sd_list, load_optimizer_states=load_sharded_optimizer_state)


def load_optimizer_checkpoint(self, path, tag):
    """
    Load the optimizer checkpoint from file and load into optimizer
    """
    from smdistributed.modelparallel.torch.checkpoint import load

    file = os.path.join(path, tag, "optimizer_states")
    optimizer_state_dict = load(file, partial=True)
    self.load_local_optimizer_state_dict(optimizer_state_dict)
    optimizer_state_dict = None
    if state.cfg.fp16:
        file = os.path.join(path, tag, "fp16_states")
        fp16_states = load(file, partial=True)
        self.load_local_fp16_state_dict(fp16_states)


def load_local_fp16_state_dict(self, fp16_states):
    """
    Load the fp16 related states, including
    - Loss scaler and overflow related information
    - Master parameters in fp32
    """
    self.loss_scaler = fp16_states["loss_scaler"]
    self.loss_scaler.model = state.model
    self.dynamic_loss_scale = fp16_states["dynamic_loss_scale"]
    self.overflow = fp16_states["overflow"]
    self.first_closure_call_this_step = fp16_states["first_closure_call_this_step"]

    def load_hook_fn(mod, opt):
        for current_group, saved_group in zip(
            self.fp32_from_fp16_groups, fp16_states["fp32_from_fp16"]
        ):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)
        log = rmsg("Loaded fp16 states")
        logger.info(log) if rank() == 0 else logger.debug(log)

    state.model.register_post_step_hook(load_hook_fn)


def load_local_optimizer_state_dict(self, state_dict):
    """
    Load the state_dict, this function always expect the state_dict is partial as we do not support full state_dict now.
    This function does not load the state_dict into the optimizer. The real loading happens
    after the partition is finished, so each rank knows which parameters' states to load.
    (WARNING) Be aware of that parameter id might change before/after partition when torchdistx is used for delayed param init
    since before partition all parameters are fake tensors
    """
    check_valid_state_dict(state_dict)

    def load_hook_fn(mod, opt):
        nonlocal state_dict
        if state.cfg.fp16:
            if rank() == 0:
                logger.info(rmsg("loading fp16 optimizer"))
            if self.master_params_created:
                register_optimizer_hooks(state.model)
        logger.debug(rmsg(f"Loading optimizer state dict..."))
        self.orig_load_state_dict(state_dict)
        log = rmsg("Loaded optimizer state_dict")
        logger.info(log) if rank() == 0 else logger.debug(log)

    state.model.register_post_step_hook(load_hook_fn)


def load_state_dict(self, state_dict, gather_if_shard=False):
    """
    Deprecated! Only used for back compatible loading.
    Load the state_dict. This function first check if the state_dict contains full states,
    if no, will allgather the states to create a full state_dict.
    This function does not load the state_dict into the optimizer. The real loading happens
    after the partition is finished, so each rank knows which parameters' states to load.
    (WARNING) Be aware of that parameter id might change before/after partition when torchdistx is used for delayed param init
    since before partition all parameters are fake tensors
    Args:
        state_dict: The state_dict to load
        gather_if_shard: Deprecated, only used with back_compat to load old checkpoint
    """
    logger.warning(
        f"DistributedOptimizer.load_state_dict() will be deprecated soon. \
        For smp >=1.10 please use smp.save_checkpoint to save a checkpoint and smp.resume_from_checkpoint to load a checkpoint"
    )
    _load_state_dict(self, state_dict, gather_if_shard=gather_if_shard)


def validate_sharding_config():
    if not state.cfg.ddp:
        raise ValueError("Optimizer state sharding is only supported when DDP is enabled.")

    if not state.model.overlapping_allreduce:
        raise ValueError(
            "Optimizer state sharding is currently only supported for overlapping allreduce."
        )


def zero_grad(self, *args, **kwargs):
    if not state.cfg.zero2d_enabled():
        self.orig_zero_grad(*args, **kwargs)
    else:
        self.orig_zero_grad()

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


def update_fake_param(self, fake_to_materialized_param):
    """
    When initialized with deferred_init, params in param_groups are fake ones. Replace them with the materalized real tensors.
    """
    for param_group in self.param_groups:
        params = param_group["params"]
        for idx, param in enumerate(params):
            if param in fake_to_materialized_param:
                params[idx] = fake_to_materialized_param[param]


def _patch_method(optimizer, method_name, function):
    func = partial(function, optimizer)
    setattr(func, "__self__", optimizer)
    setattr(func, "__func__", function)
    setattr(optimizer, method_name, func)


def extract_zero_kwargs(config_dict, static_loss_scale, dynamic_loss_scale, dynamic_loss_args):
    kwargs = {}
    arguments = {
        p.name
        for p in inspect.signature(DeepSpeedZeroOptimizer_Stage3.__init__).parameters.values()
    }
    for k, v in config_dict["zero_optimization"].items():
        if k in arguments:
            kwargs[k] = v

    # Carry over loss scaling arguments coming from smp.DistributedOptimizer
    kwargs["dynamic_loss_scale"] = dynamic_loss_scale
    kwargs["static_loss_scale"] = 0 if dynamic_loss_scale else static_loss_scale
    kwargs["dynamic_loss_args"] = dynamic_loss_args

    return kwargs


def clip_master_grads(self, grad_clip):
    if not state.cfg.fp16 or state.cfg.zero2d_enabled():
        return

    self.orig_clip_master_grads(grad_clip)


def DistributedOptimizer(
    opt, static_loss_scale=1.0, dynamic_loss_scale=False, dynamic_loss_args=None
):
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

    opt.redefined_params = False
    opt._param_id_to_index = partial(_param_id_to_index, opt)
    opt.param_name_to_index = partial(param_name_to_index, opt)
    opt.param_index_to_name = partial(param_index_to_name, opt)
    opt.index_to_name = opt.param_index_to_name()

    index_mappings = (
        opt._param_id_to_index,
        opt.param_name_to_index,
        opt.param_index_to_name,
        opt.index_to_name,
    )

    if state.cfg.zero2d_enabled():
        zero_kwargs = extract_zero_kwargs(
            state.cfg.zero2d_config_dict(), static_loss_scale, dynamic_loss_scale, dynamic_loss_args
        )

        from deepspeed.runtime.fp16 import loss_scaler

        def zero_backward(loss_scaler, loss, retain_graph=False):
            distributed_backward(state.model, loss * loss_scaler.loss_scale, None)

        # override the inner backward implementation of ZeRO-2D with smp implementation
        loss_scaler.LossScalerBase.backward = zero_backward

        opt = DeepSpeedZeroOptimizer_Stage3(
            state.model,
            opt,
            None,
            ds_config=state.cfg.zero2d_config_dict(),
            mpu=None,
            **zero_kwargs,
        )

    elif state.cfg.fp16:
        # if zero2d is being used, fp16 is handled by zero optimizer
        opt = Bit16_Optimizer(
            state.model,
            opt,
            static_loss_scale=static_loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            dynamic_loss_args=dynamic_loss_args,
        )
        logger.info(rmsg("Bit16_Optimizer initialized with dtype torch.float16"))
    elif state.cfg.bf16:
        # bfloat16 does not need loss scale
        opt = Bit16_Optimizer(state.model, opt)
        logger.info(rmsg("Bit16_Optimizer initialized with dtype torch.bfloat16"))

    opt.redefined_params = False
    opt._param_id_to_index = index_mappings[0]
    opt.param_name_to_index = index_mappings[1]
    opt.param_index_to_name = index_mappings[2]
    opt.index_to_name = index_mappings[3]

    opt.orig_load_state_dict = opt.load_state_dict
    opt.orig_state_dict = opt.state_dict
    opt.orig_step = opt.step
    opt.orig_zero_grad = opt.zero_grad

    if state.cfg.fp16 and not state.cfg.zero2d_enabled():
        opt.orig_clip_master_grads = opt.clip_master_grads

    opt.local_state_dict = partial(local_state_dict, opt)
    opt.state_dict = partial(state_dict, opt)

    opt._actual_load_state_dict = partial(_actual_load_state_dict, opt)
    opt.load_state_dict = partial(load_state_dict, opt)
    opt._param_index_to_param_local = partial(_param_index_to_param_local, opt)
    opt.zero_grad = partial(zero_grad, opt)
    opt.clip_master_grads = partial(clip_master_grads, opt)
    opt.update_fake_param = partial(update_fake_param, opt)
    opt.load_optimizer_checkpoint_zero2d = partial(load_optimizer_checkpoint_zero2d, opt)
    opt.local_optimizer_state_dict = partial(local_optimizer_state_dict, opt)
    opt.local_fp16_state_dict = partial(local_fp16_state_dict, opt)
    opt.load_local_optimizer_state_dict = partial(load_local_optimizer_state_dict, opt)
    opt.load_local_fp16_state_dict = partial(load_local_fp16_state_dict, opt)
    opt.load_optimizer_checkpoint = partial(load_optimizer_checkpoint, opt)
    _patch_method(opt, "step", step)

    state.optimizer = opt

    if not state.cfg.zero2d_enabled() and (state.cfg.fp16 or state.cfg.bf16):
        state.model.register_post_step_hook(lambda mod, opt: opt.init_master_params())

    # zero-2d load should be called after this, which is enforced inside smp.resume_from_checkpoint
    if state.loaded_optimizer_state is not None:
        opt.load_optimizer_checkpoint(*state.loaded_optimizer_state)
        state.loaded_optimizer_state = None

    return opt
