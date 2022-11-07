# Standard Library
import os
from contextlib import contextmanager

# Third Party
import torch
import torch.nn as nn

# First Party
from smdistributed.modelparallel.torch.exceptions import DelayedParamDeviceError

try:
    # Will only be available for PT >= 1.11
    from torchdistx.deferred_init import _C as deferred_init_lib
except ImportError:
    deferred_init_lib = None


use_torchdistx = deferred_init_lib != None and int(os.getenv("SMP_TORCHDISTX_DEFERRED_INIT", 0)) > 0
use_fp32_init = int(os.getenv("SMP_USE_FLOAT32_INIT", 0)) > 0


@contextmanager
def zero_sharded_init(enabled=True):
    if enabled:
        from smdistributed.modelparallel.torch.state_mod import state
        import deepspeed

        with deepspeed.zero.Init(config_dict_or_path=state.cfg.zero2d_config_dict()):
            yield
    else:
        yield


@contextmanager
def delay_param_initialization(enabled=True):
    from smdistributed.modelparallel.torch.state_mod import state

    if enabled and state.cfg.zero2d_enabled():
        with zero_sharded_init():
            yield

        return

    if not use_torchdistx and (enabled or use_fp32_init):
        org_new = nn.Parameter.__new__
        org_uniform = nn.Parameter.uniform_
        org_normal = nn.Parameter.normal_
        org_fill = nn.Parameter.fill_
        org_zero = nn.Parameter.zero_
        org_init_uniform = nn.init.uniform_
        org_init_normal = nn.init.normal_

        if enabled:
            nn.Parameter.__new__ = get_new_method(org_new)
        nn.Parameter.uniform_ = get_smp_init_method(org_uniform, enable_delay_init=enabled)
        nn.Parameter.normal_ = get_smp_init_method(org_normal, enable_delay_init=enabled)
        nn.Parameter.fill_ = get_smp_init_method(org_fill, enable_delay_init=enabled)
        nn.Parameter.zero_ = get_smp_init_method(org_zero, enable_delay_init=enabled)
        nn.init.uniform_ = get_smp_init_method(org_init_uniform, enable_delay_init=enabled)
        nn.init.normal_ = get_smp_init_method(org_init_normal, enable_delay_init=enabled)
    elif enabled:
        deferred_init_lib.enable_deferred_init(True)
    try:
        yield
    finally:
        if not use_torchdistx and (enabled or use_fp32_init):
            nn.Parameter.__new__ = org_new
            nn.Parameter.uniform_ = org_uniform
            nn.Parameter.normal_ = org_normal
            nn.Parameter.fill_ = org_fill
            nn.Parameter.zero_ = org_zero
            nn.init.normal_ = org_init_normal
            nn.init.uniform_ = org_init_uniform
        elif enabled:
            deferred_init_lib.enable_deferred_init(False)


def get_new_method(org_new):
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])

        if data.device.type != "cpu":
            raise DelayedParamDeviceError

        obj = org_new(cls, data, requires_grad)

        # TODO: Please see: https://issues.amazon.com/issues/P52569861
        # Even with patching init methods, param.data.uniform_ seems to allocate
        # and initialize memory. This requires user script to skip calls on param.data.<rand_fn>_
        # patch_init_methods(obj)

        return obj

    return __new__


def get_smp_init_method(org_initializer, enable_delay_init=True, param=None):
    from smdistributed.modelparallel.torch.state_mod import state

    def smp_initializer(self, *args, **kwargs):
        key = param or self
        if enable_delay_init:
            state.param_initializers[key] = lambda x: org_initializer(x, *args, **kwargs)
        else:
            if key.dtype == torch.float16 and use_fp32_init:
                # Initialize in fp32 on CPU and cast back to fp16 will be much faster
                with torch.no_grad():
                    key_ = key.to(torch.float32)
                    org_initializer(key_, *args, **kwargs)
                    key.copy_(key_)
            else:
                org_initializer(key, *args, **kwargs)

    return smp_initializer


def patch_init_methods(param):
    param.data.uniform_ = get_delayed_init_method(param.data.uniform_, param)
    param.data.normal_ = get_delayed_init_method(param.data.normal_, param)
    param.data.fill_ = get_delayed_init_method(param.data.fill_, param)
    param.data.zero_ = get_delayed_init_method(param.data.zero_, param)
