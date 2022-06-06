# Third Party
# Standard Library
import collections
import itertools
import os
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import core, tp_rank, tp_size
from smdistributed.modelparallel.torch.nn.utils import get_local_channels, get_start_pos_for_slicing

logger = get_logger()


@contextmanager
def flattened(structure):
    flat_structure, structure_id = flatten_structure(structure)
    yield flat_structure
    structure = unflatten_structure(flat_structure, structure_id)


def flatten_structure(structure):
    """ Return a flat list containing all elements in a possibly nested structure. structure_id
    can be used to reconstruct the original structure. """

    from smdistributed.modelparallel.torch.state_mod import state

    flat_structure, phantom_structure, _ = _flatten_structure_internal(structure, 0)

    structure_id = len(state.phantom_structures)
    state.phantom_structures[structure_id] = phantom_structure

    return flat_structure, structure_id


def unflatten_structure(flat_structure, structure_id):
    """ Reconstruct a flattened structure using its structure_id. The original structure is deleted
    from memory at every step, so this function can only reconstruct the structures that were flattened
    in the current step. """

    from smdistributed.modelparallel.torch.state_mod import state

    phantom_structure = state.phantom_structures[structure_id]
    structure = _unflatten_structure_internal(flat_structure, phantom_structure)

    return structure


def _unflatten_structure_internal(flat_structure, phantom_structure):
    if isinstance(phantom_structure, (list, tuple, set)):
        structure = []
        for item in phantom_structure:
            structure.append(_unflatten_structure_internal(flat_structure, item))
        structure_type = type(phantom_structure)
        if is_instance_namedtuple(phantom_structure):
            structure = structure_type(*structure)
        else:
            structure = structure_type(structure)
    elif isinstance(phantom_structure, dict):
        phantom_structure_type = type(phantom_structure)
        if phantom_structure_type == collections.defaultdict:
            structure = collections.defaultdict(phantom_structure.default_factory)
        else:
            structure = phantom_structure_type()
        for k in phantom_structure:
            structure[k] = _unflatten_structure_internal(flat_structure, phantom_structure[k])
    else:
        assert isinstance(phantom_structure, int), "Invalid phantom structure"
        structure = flat_structure[phantom_structure]
    return structure


def is_instance_namedtuple(iterable):
    return (
        isinstance(iterable, tuple)
        and iterable.__class__.__base__ == tuple
        and hasattr(iterable, "_fields")
    )


def _flatten_structure_internal(structure, counter):
    flat_structure = []

    if isinstance(structure, (list, tuple, set)):
        phantom_structure = []
        for item in structure:
            flat, phantom, counter = _flatten_structure_internal(item, counter)
            flat_structure.extend(flat)
            phantom_structure.append(phantom)
        structure_type = type(structure)
        if is_instance_namedtuple(structure):
            phantom_structure = structure_type(*phantom_structure)
        else:
            phantom_structure = structure_type(phantom_structure)
    elif isinstance(structure, dict):
        structure_type = type(structure)
        if structure_type == collections.defaultdict:
            phantom_structure = collections.defaultdict(structure.default_factory)
        else:
            phantom_structure = structure_type()
        # Iteration order is deterministic only on/after python3.7 / cpython3.6
        # This should be fine for the library on SM
        for k, v in structure.items():
            flat, phantom, counter = _flatten_structure_internal(v, counter)
            flat_structure.extend(flat)
            phantom_structure[k] = phantom
    else:
        flat_structure = [structure]
        phantom_structure = counter
        counter += 1

    return flat_structure, phantom_structure, counter


def map_structure(fn, structure):
    """ Apply the callable `fn` to every element of the possibly nested structure `structure`,
    and return the result."""
    if isinstance(structure, (list, tuple, set)):
        mapped_structure = []
        for item in structure:
            mapped_structure.append(map_structure(fn, item))
        structure_type = type(structure)
        if is_instance_namedtuple(structure):
            mapped_structure = structure_type(*mapped_structure)
        else:
            mapped_structure = structure_type(mapped_structure)
        return mapped_structure
    elif isinstance(structure, dict):
        structure_type = type(structure)
        if structure_type == collections.defaultdict:
            mapped_structure = collections.defaultdict(structure.default_factory)
        else:
            mapped_structure = structure_type()
        for k, v in structure.items():
            mapped_structure[k] = map_structure(fn, v)
        return mapped_structure
    return fn(structure)


def debug_print_input(name, index, k, inp):
    if torch.is_tensor(inp):
        inp_print = f"{inp.shape}"
    elif isinstance(inp, torch.nn.Module):
        inp_print = f"module"  # {list(inp.named_modules())}'
    else:
        inp_print = f"{inp}"
    logger.debug(
        f"{core.rank()} when processing subgraph {name} has input {index} as" f"{k}:{inp_print}"
    )


def debug_print_outputs(name, outputs):
    s = f"{core.rank()} name: {name}, outputs:"
    if torch.is_tensor(outputs):
        list_of_outputs = [outputs]
    else:
        list_of_outputs = outputs
    for out in list_of_outputs:
        s += f"{id(out)} {out.shape} "
    logger.debug(s)


def _convert_to_device(arg, device, only_convert_gpu_tensors):
    if torch.is_tensor(arg) and (not only_convert_gpu_tensors or arg.is_cuda):
        return arg.to(device)
    else:
        return arg


def convert_args_to_device(
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    device: torch.device,
    only_convert_gpu_tensors: bool = False,
):
    convert = lambda x: _convert_to_device(x, device, only_convert_gpu_tensors)
    return map_structure(convert, (args, kwargs))


def rmsg(msg):
    return f"[{core.rank()}] {msg}"


def get_tensor_size(obj):
    return obj.numel() * dtype_size(obj.dtype)


def get_size_tensors_in_obj(obj):
    from smdistributed.modelparallel.torch.state_mod import state

    flat_obj, _ = flatten_structure(obj)
    total = 0
    for x in flat_obj:
        if torch.is_tensor(x) and not state.no_grad_context:
            total += get_tensor_size(x)
    return total


def get_devices_tensors_in_obj(obj, device_set):
    if isinstance(obj, (tuple, list, set)):
        for out in obj:
            get_devices_tensors_in_obj(out, device_set)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            get_devices_tensors_in_obj(v, device_set)
    elif torch.is_tensor(obj):
        device_set.add(obj.device)
    return device_set


def dtype_size(dtype):
    if dtype == torch.float16:
        return 2
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 8
    elif dtype == torch.complex64:
        return 8
    elif dtype == torch.complex128:
        return 16
    elif dtype == torch.bfloat16:
        return 2
    else:
        # default to 4 bytes
        return 4


def product(x):
    prod = 1
    for i in x:
        prod *= i
    return prod


def check_requires_grad(args):
    """Returns True if at least one of the inputs
    has requires_grad = True, otherwise returns False"""
    flat_args, _ = flatten_structure(args)
    return any([torch.is_tensor(x) and x.requires_grad for x in flat_args])


def pack_kwargs(args, kwargs):
    """Packs the kwargs which are tensors into a tuple along
    with args. Returns (args, pos_to_key_dtype) where pos_to_key_dtype
    is a dict which stores a mapping of position of tensor in
    the returned tuple to the (keyword, tensor_dtype)"""
    args, pos_to_key_dtype = list(args), {}
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            pos_to_key_dtype[len(args)] = (key, val.dtype)
            args.append(val)
    return tuple(args), pos_to_key_dtype


def unpack_kwargs(args, kwargs, pos_to_key_dtype):
    """Unpacks the args into args, kwargs (removes from args)
    and adds to kwargs, given the current kwargs and
    pos_to_key_dtype dict containing args to (keyword, dtype) mapping
    """
    return_args = []
    for pos, arg in enumerate(args):
        if pos not in pos_to_key_dtype:
            return_args.append(arg)
        else:
            key, dtype = pos_to_key_dtype[pos]
            # below will still copy if it cannot be safely cast
            kwargs[key] = arg.to(dtype, copy=False)
    return tuple(return_args), kwargs


def check_supported(module_name, args, kwargs, skip_tensor_check=False):
    """If the requires_grad is set to True, check if the inputs
    are floating or complex type, raise error if not"""

    from smdistributed.modelparallel.torch.state_mod import state

    def _validate_arg(x):
        if (
            isinstance(x, torch.Tensor)
            and not (x.dtype.is_floating_point or x.dtype.is_complex)
            and x.requires_grad
        ):
            raise TypeError(
                f"Module: {module_name} has an argument with requires_grad=True "
                "with a non-float type. "
                "Please change the type of the tensor to a float type "
                "or set requires_grad=False"
            )

    def _validate_params(x):
        if isinstance(x, torch.nn.Parameter):
            if state.is_tracing():
                state.module_manager.push_additional_parameters(module_name, x)
            # Check for unsupported parameters passed in forward
            try:
                state.model.get_param_name(x)
            except KeyError:
                raise RuntimeError(
                    f"Unsupported use case for module: '{module_name}'. Parameter passing in forward is not supported for parameters not owned by the DistributedModel."
                )

    if not skip_tensor_check:
        map_structure(_validate_arg, (args, kwargs))
    map_structure(_validate_params, (args, kwargs))

    return True


def get_distribution_axis(obj):
    """ Returns None if the object is not a distributed tensor"""

    from smdistributed.modelparallel.torch.state_mod import state

    axis = state.module_manager.get_parameter_distribution_axis(obj)
    if axis is None:
        axis = state.module_manager.get_buffer_distribution_axis(obj)
    return axis


def get_root_module_for_param(model, state_param_name):
    """Parameters in Pytorch are named according to the
    module hierarchy. Searches in the provided DistributedModel's module hierarchy,
    and returns the module whose default named_parameters() would have generated
    the parameter name as provided in state_param_name.

    Example:

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.modulec = Module2()

        class Module2(nn.Module):
            def __init__(self):
                super(Module2, self).__init__()
                self.child1 = nn.Linear(10, 10)
                self.child2 = nn.Linear(10, 10)

        model = Model()
        model = DistributedModel(model)
        ret_model = get_root_module_for_param(model, "child1.weight")
        assert ret_model == model.module.modulec
    """
    from smdistributed.modelparallel.torch.state_mod import state

    if not state_param_name:
        return model
    for name, param in itertools.chain(model.named_parameters(), model.named_buffers()):
        if state_param_name == name:
            return model.module.module if state.cfg.ddp else model.module
        # makes some assumptions about PyTorch param naming
        elif state_param_name in name:
            mod_names = name.split(".")[:-1]
            state_param_first_mod = state_param_name.split(".")[0]
            mod = model.module if not state.cfg.ddp else model.module.module
            for mod_name in mod_names:
                if state_param_first_mod != mod_name:
                    mod = getattr(mod, mod_name)
                else:
                    break
            return mod
    return model


def slice_tp_tensors_in_state_dict(state_dict):
    """Slices tp tensors in state dict
    """
    from smdistributed.modelparallel.torch.state_mod import state

    def _on_only_one_rank(param):
        return state.module_manager.is_one_rank_parameter(
            param
        ) or state.module_manager.is_one_rank_buffer(param)

    if len(state_dict) == 0:
        return state_dict

    # find root module for a state param name.
    state_param_name = list(state_dict.keys())[0]
    model = get_root_module_for_param(state.model, state_param_name)

    model_state_dict = model.state_dict()
    model_params = dict(named_parameters_nonunique(model))
    model_buffers = dict(named_buffers_nonunique(model))

    sliced_state_dict = {}
    for name, item in state_dict.items():
        param = model_params.get(name) if name in model_params else model_buffers.get(name, None)
        axis = get_distribution_axis(param)
        if axis is None:
            if _on_only_one_rank(param):
                if tp_rank() == 0:
                    sliced_state_dict[name] = item
            elif name in model_state_dict:
                sliced_state_dict[name] = item
        else:
            # item must be a torch.Tensor (specifically, nn.Parameter)
            assert isinstance(
                item, torch.Tensor
            ), f"Non-tensor distributed object found {type(item)}"

            # slice the tensor
            weight_split_shapes = state.module_manager.weight_split_shapes.get(param, None)
            slice_size = (
                get_local_channels(item.size(axis))
                if weight_split_shapes == None
                else weight_split_shapes[tp_rank()]
            )
            start = (
                get_start_pos_for_slicing(item.size(axis))
                if weight_split_shapes == None
                else sum(weight_split_shapes[: tp_rank()])
            )
            sliced_state_dict[name] = item.narrow(axis, start, slice_size).contiguous()
    return sliced_state_dict


def collect_and_merge(
    input_dict, strict=False, optimizer=False, model_partitioned=True, collect_fn=None
):
    from smdistributed.modelparallel.torch.comm import CommGroup, allgather

    if collect_fn == None:
        collect_fn = allgather

    if tp_size() > 1:
        tp_group_dict = collect_and_merge_in_group(
            input_dict, strict, CommGroup.TP_GROUP, optimizer=optimizer, collect_fn=collect_fn
        )
        if not model_partitioned:
            # no need of PP allgather here as PP params would already be gathered/inited at the beginning on CPU
            return tp_group_dict

        return collect_and_merge_in_group(
            tp_group_dict, strict, CommGroup.PP_GROUP, optimizer=optimizer, collect_fn=collect_fn
        )
    elif model_partitioned:
        return collect_and_merge_in_group(
            input_dict, strict, CommGroup.PP_GROUP, optimizer=optimizer, collect_fn=collect_fn
        )
    else:
        return input_dict


def named_buffers_nonunique(model, prefix="", recurse=True):
    gen = _named_members_nonunique(
        model, lambda module: module._buffers.items(), prefix=prefix, recurse=recurse
    )
    for elem in gen:
        yield elem


def named_parameters_nonunique(model, prefix="", recurse=True):
    gen = _named_members_nonunique(
        model, lambda module: module._parameters.items(), prefix=prefix, recurse=recurse
    )
    for elem in gen:
        yield elem


def _named_members_nonunique(model, get_members_fn, prefix="", recurse=True):
    modules = model.named_modules(prefix=prefix) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
        members = get_members_fn(module)
        for k, v in members:
            if v is None:
                continue
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v


def collect_and_merge_in_group(
    input_dict, strict=False, group=None, optimizer=False, collect_fn=None
):
    """
    Allgather a dictionary from all ranks and merge into a single dictionary.
    Supports optimizer and model state dictionaries currently.
        input_dict: the dict to be allgathered and merged
        strict: if True, will raise an error if there are duplicate entries in the dicts across different ranks
        group: The CommGroup to allgather over
    """
    from smdistributed.modelparallel.torch.comm import CommGroup, allgather
    from smdistributed.modelparallel.torch.state_mod import state

    if group is None:
        group = CommGroup.PP_GROUP

    if group not in {CommGroup.TP_GROUP, CommGroup.PP_GROUP}:
        raise ValueError("Only TP_GROUP and PP_GROUP are supported for allgather and merge.")

    def _duplicated_params(dict1, dict2):
        same_key = set(dict1.keys()).intersection(set(dict2.keys()))
        duplicated_params = []
        for name in same_key:
            if isinstance(dict1[name], torch.Tensor):
                duplicated_params.append(name)
        return duplicated_params

    if collect_fn is None:
        collect_fn = allgather

    all_input_dict = collect_fn(input_dict, group)

    full_input_dict = {}

    if group == CommGroup.PP_GROUP:
        for item in all_input_dict:
            duplicated_params = _duplicated_params(item, full_input_dict)
            if len(duplicated_params) > 0:
                if strict:
                    raise ValueError(
                        f"{core.rank()} Duplicate parameters {duplicated_params} found across ranks while combining state_dicts."
                    )
                else:
                    logger.warning(
                        f"{core.rank()} Loading duplicated parameters {duplicated_params}"
                    )
            full_input_dict.update(item)
    else:
        # group == CommGroup.TP_GROUP
        if not optimizer:
            # state_dict() called on the DistributedModel

            # Assumes that tp_rank 0 has maximum params.
            # if a param is not distributed but present only
            # on one rank, below code assumes it is on tp_rank() == 0
            if len(all_input_dict) == 0:
                return full_input_dict
            max_params_index = 0
            max_params_obj = all_input_dict[max_params_index]
            example_param_name = list(max_params_obj.keys())[0] if len(max_params_obj) > 0 else None
            model = get_root_module_for_param(state.model, example_param_name)
            model_params = dict(named_parameters_nonunique(model))
            model_buffers = dict(named_buffers_nonunique(model))

            for param_name, param in max_params_obj.items():
                param_or_buffer = (
                    model_params.get(param_name)
                    if param_name in model_params
                    else model_buffers.get(param_name, None)
                )
                axis = get_distribution_axis(param_or_buffer)

                if axis is None:
                    # param is not distributed: could be an only one rank param or a param replicated on all ranks.
                    # In both cases, we need to replicate params, so that state_dict contains the param for all tp_ranks.
                    full_input_dict[param_name] = param
                else:
                    # param is distributed, concatenate across axis for the tp_group
                    full_input_dict[param_name] = torch.cat(
                        [all_input_dict[r][param_name] for r in range(tp_size())], axis
                    )
        else:
            # state_dict() called on the DistributedOptimizer
            param_index_to_param = state.optimizer._param_index_to_param_local()
            # Assumes that tp_rank 0 has maximum params.
            # if a param is not distributed but present only
            # on one rank (one_rank_parameters), below code assumes
            # it is on tp_rank == 0
            max_params_index = 0
            max_params_obj = all_input_dict[max_params_index]

            param_name_to_index = state.optimizer.param_name_to_index()
            param_name_to_index_tp_group = state.param_name_to_index_tp_group
            param_index_to_name_tp_group = state.param_index_to_name_tp_group

            for param_index, param_states in max_params_obj.items():
                # Below line assumes that non distributed params are
                # present on tp_rank() == 0, if present at all
                param_name = param_index_to_name_tp_group[max_params_index][param_index]
                param_index_local = param_name_to_index.get(param_name, None)

                axis = get_distribution_axis(param_index_to_param.get(param_index_local, None))

                result_param_states = {}
                for param_state in param_states:
                    if isinstance(param_states[param_state], torch.Tensor):
                        if axis is not None:
                            # param state is distributed, concatenate along the axis
                            outputs = []
                            for r in range(tp_size()):
                                rank_param_index = param_name_to_index_tp_group[r][param_name]
                                outputs.append(all_input_dict[r][rank_param_index][param_state])
                            result_param_states[param_state] = torch.cat(outputs, axis)
                        else:
                            # assumption that non distributed param state are present on tp_rank() == 0
                            rank_param_index = param_name_to_index_tp_group[max_params_index][
                                param_name
                            ]
                            result_param_states[param_state] = max_params_obj[rank_param_index][
                                param_state
                            ]
                    else:
                        result_param_states[param_state] = param_states[param_state]
                full_input_dict[param_index] = result_param_states

    return full_input_dict


def raise_ddp_overlapping_exception(method_name):
    raise NotImplementedError(
        f"{method_name} is only supported when SMP is configured to use DDP and overlapping_allreduce is set to True. "
        "Please set ddp=True in SMP config, and ensure overlapping_allreduce parameter in smp.DistributedModel wrapper is set to True."
    )


def add_weight_split_shapes(weight, shapes):
    from smdistributed.modelparallel.torch.state_mod import state

    state.module_manager.weight_split_shapes[weight] = shapes


def check_env_var_truthy(env_var_name, default):
    return os.getenv(env_var_name, default).lower() in ["1", "true"]
