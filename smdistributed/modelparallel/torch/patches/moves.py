# Standard Library
from contextlib import contextmanager
from typing import Any

# Third Party
import torch
from torch.nn import Module

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import local_rank, pp_rank
from smdistributed.modelparallel.torch.state_mod import state

logger = get_logger()


@contextmanager
def patch_no_children(m: Module):
    # before calling original to method we trick pytorch into thinking this module has no children
    # this is because the to method uses a function defined within a function and
    # the structure is such that we can't control param assignment only for the current module
    # without affecting children

    cls = m.__class__
    orig_children_method = cls.children
    cls.children = no_children_method
    try:
        yield
    finally:
        # reset children method back to original
        cls.children = orig_children_method


def no_children_method(m: Module):
    """
    Defining an empty iterator
    """
    return
    yield


def _modify_device_in_args_kwargs(module_device, *args, **kwargs):
    # create new args and kwargs which override the device object if it was passed as one of the args
    # part of why this is needed is because to method does lot more than device placement only
    new_args = []
    new_kwargs = kwargs.copy()
    for arg in args:
        if isinstance(arg, torch.device):
            new_args.append(module_device)
        elif torch.is_tensor(arg):
            new_args.append(module_device)
            new_args.append(tensor.dtype)
        else:
            new_args.append(arg)
    if "device" in new_kwargs:
        new_kwargs["device"] = module_device
    if "tensor" in new_kwargs:
        new_kwargs["device"] = module_device
        new_kwargs["dtype"] = tensor.dtype
        del new_kwargs["tensor"]
    new_args = tuple(new_args)
    return new_args, new_kwargs


def _remove_device_from_args_kwargs(*args, **kwargs):
    new_args = []
    new_kwargs = kwargs.copy()
    for arg in args:
        if isinstance(arg, torch.device):
            continue
        elif torch.is_tensor(arg):
            new_args.append(tensor.dtype)
        else:
            new_args.append(arg)
    if "device" in new_kwargs:
        del new_kwargs["device"]
    if "tensor" in new_kwargs:
        new_kwargs["dtype"] = tensor.dtype
        del new_kwargs["tensor"]
    new_args = tuple(new_args)
    return new_args, new_kwargs


def _get_device_from_args_kwargs(*args, **kwargs):
    for arg in args:
        if isinstance(arg, torch.device):
            return arg
        elif isinstance(arg, torch.Tensor):
            # it uses tensor's device
            return arg.device
    if "device" in kwargs or "tensor" in kwargs:
        return kwargs["device"]
    return None


def original_to(m: Module, device: torch.device):
    # ignores any checks that are in distributed_to as we want to move the device in this case
    state.patch_manager.get_original_method("to", Module)(m, device)


def _get_device(m: Module):
    partition_id = state.module_manager.get_partition(m)
    if partition_id is not None and pp_rank() != partition_id:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", local_rank())
    return device


def distributed_to(m: Module, *args: Any, **kwargs: Any):
    """
    Assigns parameters to gpu only for modules assigned to a particular process
    """
    if not state.model.partitioned and _get_device_from_args_kwargs(*args, **kwargs) != None:
        logger.warning(
            "Model has not been partitioned yet, ignoring move of params and buffers to devices"
        )
        args, kwargs = _remove_device_from_args_kwargs(*args, **kwargs)
    elif state.model.partitioned:
        module_device = _get_device(m)
        args, kwargs = _modify_device_in_args_kwargs(module_device, *args, **kwargs)

    # call new to method for each child separately
    for child in m.children():
        distributed_to(child, *args, **kwargs)

    with patch_no_children(m):
        state.patch_manager.get_original_method("to", Module)(m, *args, **kwargs)
    # similar to how apply returns self in torch.nn.Module
    return m
