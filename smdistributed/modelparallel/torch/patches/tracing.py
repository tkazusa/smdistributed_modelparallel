# First Party
# Standard Library
from typing import Any

# Third Party
import torch
import torch.nn as nn
from torch.nn import Module

from smdistributed.modelparallel.torch.core import local_rank
from smdistributed.modelparallel.torch.exceptions import SMPRuntimeError
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import (
    check_requires_grad,
    convert_args_to_device,
    flatten_structure,
    get_size_tensors_in_obj,
    map_structure,
    rmsg,
    unflatten_structure,
)


class TracingEnd(Exception):
    pass


def is_retriable_on_gpu(module, exception):
    # https://github.com/NVIDIA/apex/blob/e1b7997a63babb73c3279ccecdc2bd3f61b8e462/csrc/type_shim.h#L33
    # if we falsely retry an exception, it'll just crash next when being retried on GPU
    # if we miss them here, they'll crash without trying on GPU
    if isinstance(module, nn.Embedding) and module.weight.dtype == torch.float16:
        return True

    return any(
        x in str(exception)
        for x in ["not implemented for 'Half'", "not supported on CPUType for Half"]
    ) or isinstance(exception, NotImplementedError)


def trace_forward(self: Module, *args: Any, **kwargs: Any):
    """
    Keeps track of the following during tracing
    1. shapes of inputs and outputs of modules as an int summed over all outputs
    2. order of execution of modules
    3. execution time of modules
    """
    # TODO: if gpu does not have enough memory for model inference, then tracing will fail
    # we will need to do the trace in parts
    state.module_manager.record_execution_order(self)
    original_forward = state.patch_manager.get_original_method("forward", self.__class__)

    try:
        with state.module_manager.record_execution_time(self, state.model.trace_device):
            with state.module_manager.record_memory_usage(self, state.model.trace_device):
                output = original_forward(self, *args, **kwargs)
    except Exception as e:
        if state.model.trace_device == "cpu" and is_retriable_on_gpu(self, e):
            gpu_device = torch.device("cuda", local_rank())
            cpu_device = torch.device("cpu")

            # move module, args to gpu
            args, kwargs = convert_args_to_device(args, kwargs, device=gpu_device)
            state.patch_manager.get_original_method("to", Module)(self, gpu_device)
            state.model.current_trace_device = "gpu"

            output = original_forward(self, *args, **kwargs)

            # move module, args, output to cpu
            output = map_structure(lambda x: x.to(cpu_device), output)
            args, kwargs = convert_args_to_device(args, kwargs, device=cpu_device)
            state.patch_manager.get_original_method("to", Module)(self, cpu_device)
            state.model.current_trace_device = "cpu"
        else:
            raise e

    state.module_manager.save_input_size(self, get_size_tensors_in_obj((args, kwargs)))
    state.module_manager.save_output_size(self, get_size_tensors_in_obj(output))
    if state.module_manager.is_main_module(self):
        if check_requires_grad(output):
            raise SMPRuntimeError("output should not require grad")
        raise TracingEnd()
        # cast to gpu so code outside module can work on it
        # return output.to(torch.device("cuda", local_rank()))

    return output


def trace_forward_seq(self: Module, *args: Any, **kwargs: Any):
    if "checkpoint_activations" in kwargs:
        del kwargs["checkpoint_activations"]
    return trace_forward(self, *args, **kwargs)
