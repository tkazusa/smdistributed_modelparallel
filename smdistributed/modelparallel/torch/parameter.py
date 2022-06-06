# Standard Library
from contextlib import contextmanager

# Third Party
import torch
import torch.nn as nn


@contextmanager
def delay_param_initialization(enabled=True):
    if enabled:
        org_new = nn.Parameter.__new__
        org_uniform = nn.Parameter.uniform_
        org_normal = nn.Parameter.normal_
        org_fill = nn.Parameter.fill_
        org_zero = nn.Parameter.zero_
        org_init_uniform = nn.init.uniform_
        org_init_normal = nn.init.normal_


        nn.Parameter.__new__ = get_new_method(org_new)
        nn.Parameter.uniform_ = get_delayed_init_method(org_uniform)
        nn.Parameter.normal_ = get_delayed_init_method(org_normal)
        nn.Parameter.fill_ = get_delayed_init_method(org_fill)
        nn.Parameter.zero_ = get_delayed_init_method(org_zero)
        nn.init.uniform_ = get_delayed_init_method(org_init_uniform)
        nn.init.normal_ = get_delayed_init_method(org_init_normal)

    try:
        yield
    finally:
        if enabled:
            nn.Parameter.__new__ = org_new
            nn.Parameter.uniform_ = org_uniform
            nn.Parameter.normal_ = org_normal
            nn.Parameter.fill_ = org_fill
            nn.Parameter.zero_ = org_zero
            nn.init.normal_ = org_init_normal
            nn.init.uniform_ = org_init_uniform


def get_new_method(org_new):
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])

        if data.device.type != "cpu":
            raise ValueError("Parameter must be on CPU for delayed initialization.")

        obj = org_new(cls, data, requires_grad)

        # TODO: Please see: https://issues.amazon.com/issues/P52569861
        # Even with patching init methods, param.data.uniform_ seems to allocate
        # and initialize memory. This requires user script to skip calls on param.data.<rand_fn>_
        #patch_init_methods(obj)

        return obj

    return __new__


def get_delayed_init_method(org_initializer, param=None):
    from smdistributed.modelparallel.torch.state_mod import state

    def delayed_initializer(self, *args, **kwargs):
        key = param or self
        state.param_initializers[key] = lambda x: org_initializer(x, *args, **kwargs)

    return delayed_initializer


def patch_init_methods(param):
    param.data.uniform_ = get_delayed_init_method(param.data.uniform_, param)
    param.data.normal_ = get_delayed_init_method(param.data.normal_, param)
    param.data.fill_ = get_delayed_init_method(param.data.fill_, param)
    param.data.zero_ = get_delayed_init_method(param.data.zero_, param)
