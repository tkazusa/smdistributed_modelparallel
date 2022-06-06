# Standard Library
import copy
from contextlib import contextmanager
from distutils.version import LooseVersion

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import init

# First Party
from smdistributed.modelparallel.torch.comm import CommGroup
from smdistributed.modelparallel.torch.core import local_rank, tp_rank, tp_size


def cat_cuda_dim0(tensor_list):
    if len(tensor_list) == 0:
        return []

    t0 = tensor_list[0]
    shp = list(t0.shape)
    shp[0] = sum([t.shape[0] for t in tensor_list])

    out = torch.empty(*shp, dtype=t0.dtype, device=torch.device("cuda", local_rank()))
    cur = 0
    for t in tensor_list:
        size = t.shape[0]
        out_slice = out.narrow(0, cur, size)
        out_slice.copy_(t)
        cur += size

    return out


def shard_sequence(*args, shift=0, bwd_allgather=True):
    seq_length = args[0].shape[1]

    if seq_length % tp_size() != 0:
        raise ValueError(
            "Sequence length must be divisible by the tensor parallelism degree when prescaled_batch is True."
        )

    seq_per_tp = seq_length // tp_size()

    transposed = [(arg.transpose(0, 1).contiguous() if arg is not None else None) for arg in args]
    return tuple(
        (narrow_for_tp(arg, 0, shift, bwd_allgather=bwd_allgather) if arg is not None else None)
        for arg in transposed
    )


def unshard_sequence(seq_length, *args):
    if seq_length % tp_size() != 0:
        raise ValueError(
            "Sequence length must be divisible by the tensor parallelism degree when prescaled_batch is True."
        )

    seq_per_tp = seq_length // tp_size()
    allgathered = [(allgather_for_tp(arg, 0) if arg is not None else None) for arg in args]
    return tuple(
        (arg.transpose(0, 1).contiguous() if arg is not None else None) for arg in allgathered
    )


def update_copy_dict(d, **kwargs):
    d_copy = copy.copy(d)
    for k, v in kwargs.items():
        d_copy[k] = v
    return d_copy


def parse_args(args, kwargs, keys, module_obj=None):
    keys_items = list(keys.items())
    parsed_kwargs = {}

    def _set_attr(k, v):
        parsed_kwargs[k] = v
        if module_obj is not None:
            setattr(module_obj, k, v)

    for i, arg in enumerate(args):
        key = keys_items[i][0]
        _set_attr(key, arg)

    for key, value in kwargs.items():
        _set_attr(key, value)

    for key, default in keys_items:
        if key not in parsed_kwargs:
            _set_attr(key, default)

    return parsed_kwargs


def override_params_with_normal(module, initializer_range):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=initializer_range)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=initializer_range)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()


@contextmanager
def parameter_creation_scope(
    module, scaled_batch=True, dtype=None, use_normal=False, initializer_range=0.02
):
    """
    If an nn.Parameter that is part of a DistributedModule acts on the full batch (instead of just the local batch)
    it must be created in this context, so that it is used with the correct reducer. WARNING: this context operates
    on the parameters of the submodule too, so it should not be nested to avoid duplicate hooks.
    """
    from smdistributed.modelparallel.torch.state_mod import state

    if dtype is None:
        dtype = torch.get_default_dtype()

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    existing_params = {p for p in module.parameters()}
    existing_buffers = {b for b in module.buffers()}

    yield

    for p in module.parameters():
        if scaled_batch and p not in existing_params:
            state.module_manager.add_scaled_batch_parameter(p)

    for b in module.buffers():
        if scaled_batch and b not in existing_buffers:
            state.module_manager.add_scaled_batch_buffer(b)

    if use_normal:
        override_params_with_normal(module, initializer_range)

    torch.set_default_dtype(default_dtype)


@contextmanager
def initialize_with_input_partition(
    module, fan_in_scale=None, fan_out_scale=None, exclude_from_dist=None, axis=1
):
    """ Adjust parameter initialization logic to take into account the entire size of the weight distributed across tp_group,
        when input channels are partitioned. """
    from smdistributed.modelparallel.torch.state_mod import state

    fan_in_scale = fan_in_scale or 1
    fan_out_scale = fan_out_scale or 1
    if not exclude_from_dist:
        exclude_from_dist = []

    existing_params = {p for p in module.parameters()}
    existing_buffers = {b for b in module.buffers()}
    with _patch_fan_in_and_fan_out("input", fan_in_scale, fan_out_scale):
        yield

    for n, p in module.named_parameters():
        if p not in existing_params and n not in exclude_from_dist:
            state.module_manager.add_distributed_parameter(p, axis)
        elif n in exclude_from_dist:
            state.module_manager.add_one_rank_parameter(p)

    for n, b in module.named_buffers():
        if b not in existing_buffers and n not in exclude_from_dist:
            state.module_manager.add_distributed_buffer(b, axis)
        elif n in exclude_from_dist:
            state.module_manager.add_one_rank_buffer(b)


@contextmanager
def initialize_with_output_partition(
    module, fan_in_scale=None, fan_out_scale=None, exclude_from_dist=None, axis=0
):
    """ Adjust parameter initialization logic to take into account the entire size of the weight distributed across tp_group,
        when output channels are partitioned. """
    from smdistributed.modelparallel.torch.state_mod import state

    fan_in_scale = fan_in_scale or 1
    fan_out_scale = fan_out_scale or 1
    if not exclude_from_dist:
        exclude_from_dist = []

    existing_params = {p for p in module.parameters()}
    existing_buffers = {b for b in module.buffers()}
    with _patch_fan_in_and_fan_out("output", fan_in_scale, fan_out_scale):
        yield

    for n, p in module.named_parameters():
        if p not in existing_params and n not in exclude_from_dist:
            state.module_manager.add_distributed_parameter(p, axis)
        elif n in exclude_from_dist:
            state.module_manager.add_one_rank_parameter(p)

    for n, b in module.named_buffers():
        if b not in existing_buffers and n not in exclude_from_dist:
            state.module_manager.add_distributed_buffer(b, axis)
        elif n in exclude_from_dist:
            state.module_manager.add_one_rank_buffer(b)


@contextmanager
def _patch_fan_in_and_fan_out(partition, fan_in_scale, fan_out_scale):
    """
    When PyTorch initializes a weight (m-by-n matrix), it initializes each element randomly,
    with a variance that depends on the dimensions m and n. When we distribute the weight into
    d devices, we create a local weight with dimensions m/d-by-n, so the variance of the local
    elements are computed based on the reduced dimensions. By patching this function, we are
    making PyTorch effectively aware of the full dimension of the weight, so the initialization
    variance is computed as if the whole matrix is being initialized.
    """

    assert partition in [None, "input", "output"]

    org_fan_in_and_fan_out = init._calculate_fan_in_and_fan_out

    def adjusted_fan_in_and_fan_out(tensor):
        fan_in, fan_out = org_fan_in_and_fan_out(tensor)
        fan_in *= fan_in_scale
        fan_out *= fan_out_scale

        if partition == "input":
            return tp_size() * fan_in, fan_out
        elif partition == "output":
            return fan_in, tp_size() * fan_out
        else:
            raise ValueError(f"Unsupported partition type {partition}.")

    init._calculate_fan_in_and_fan_out = adjusted_fan_in_and_fan_out
    yield
    init._calculate_fan_in_and_fan_out = org_fan_in_and_fan_out


def get_local_channels(num_channels, rank=None):
    """Given the number of channel in a certain dimension, get the number of local num_channels
    for a certain rank in case of tensor parallel.
    IMPORTANT: This function needs to align with get_start_pos_for_slicing()!!!"""
    if rank == None:
        rank = tp_rank()
    div = num_channels // tp_size()
    rem = num_channels % tp_size()
    return div + 1 if rank < rem else div


def get_start_pos_for_slicing(num_channels, rank=None):
    """Given the number of channel in a certain dimension, get the start position
    for slicing for a certain rank in case of tensor parallel.
    IMPORTANT: This function needs to align with get_local_channels()!!!"""
    if rank == None:
        rank = tp_rank()
    rem = num_channels % tp_size()
    local_in_features = get_local_channels(num_channels, rank=rank)
    start = local_in_features * rank + (0 if rank < rem else rem)
    return start


def narrow_and_scale_for_tracing(tensor, narrow_dim, scale_dim):

    # support negative indexing
    if scale_dim < 0 and scale_dim >= -tensor.dim():
        scale_dim += tensor.dim()
    if narrow_dim < 0 and narrow_dim >= -tensor.dim():
        narrow_dim += tensor.dim()

    num_channels = tensor.size(narrow_dim)
    local_channels = get_local_channels(num_channels)
    tensor_slice = tensor.narrow(narrow_dim, 0, local_channels)

    return scale_batch_for_tracing(tensor_slice, scale_dim)


def scale_batch_for_tracing(tensor, batch_dim, merge_shapes=None):
    if merge_shapes is not None:
        return torch.cat(
            [tensor.narrow(batch_dim, 0, merge_shapes[i]) for i in range(tp_size())], dim=batch_dim
        )
    return torch.cat([tensor for _ in range(tp_size())], dim=batch_dim)


def narrow_batch_for_tracing(tensor, batch_dim):
    batch_size = tensor.size(batch_dim)
    local_batch_size = get_local_channels(batch_size)
    return tensor.narrow(batch_dim, 0, local_batch_size)


def batch_collective(collective, tensor_list, batch_dim):

    concat_dim = None
    # TODO implement fast method to find concat_dim

    if concat_dim is None:
        # if we cannot concatenate, call collectives individually
        return [collective(tensor, batch_dim) for tensor in tensor_list]
    else:
        # if we found it, make single collective call
        cat_result = collective(cat_tensors, batch_dim)
        cat_sizes = [tensor.shape[concat_dim] for tensor in tensor_list]
        return torch.split(cat_result, cat_sizes, dim=concat_dim)


def reduce_scatter_for_tp(tensor, batch_dim, split_shapes=None, shift=0, transpose=None):
    """
    Reduce (sum) and split the result across the batch dimension. Becomes allgather in backward.
    """

    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing():
        return ReduceScatterForTP.apply(tensor, batch_dim, split_shapes, shift, transpose)
    else:
        if transpose:
            tensor.transpose_(*transpose)
        narrowed = narrow_batch_for_tracing(tensor, batch_dim)
        if shift < 0:
            narrowed = narrowed.narrow(
                batch_dim, 0, narrowed.size()[batch_dim] + shift
            ).contiguous()
        return narrowed


def fused_allgather_for_tp(tensor, batch_dim, merge_shapes=None):
    """
    Allgather and concatenate tensors along the batch dimension. In contrast to the allgather call,
    this performs a reduction in backward. Equivalent to an allgather call followed by a bwd_allreduce
    call.
    """

    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing():
        return FusedAllgatherForTP.apply(tensor, batch_dim, merge_shapes)
    else:
        return scale_batch_for_tracing(tensor, batch_dim, merge_shapes=merge_shapes)


def narrow_for_tp(tensor, batch_dim, shift=0, bwd_allgather=True):
    """
    Narrow the tensor along the batch dimension, and take the slice based on the current tp_rank. In backward,
    allgathers the gradients for all slices from other tp_ranks.
    """

    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing():
        return NarrowForTP.apply(tensor, batch_dim, shift, bwd_allgather)
    else:
        narrowed = narrow_batch_for_tracing(tensor, batch_dim)
        if shift < 0:
            return narrowed.narrow(batch_dim, 0, narrowed.size()[batch_dim] + shift).contiguous()
        else:
            return narrowed


def allgather_for_tp(tensor, batch_dim, merge_shapes=None):
    """
    Allgather and concatenate tensors along the batch dimension. In contrast to the fused_allgather call,
    this DOES NOT perform a reduction in backward. WARNING: Should only be used when a bwd_allreduce call
    follows it, or else it would lead to incorrect results. allgather + bwd_allreduce combination is
    useful when the intermediate value (allgather output) is needed (e.g., in transformer layer), and
    the combination of the two is functionally equivalent to a single fused_allgather call.
    """

    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing():
        return AllgatherForTP.apply(tensor, batch_dim, merge_shapes)
    else:
        return scale_batch_for_tracing(tensor, batch_dim, merge_shapes=merge_shapes)


def scatter_and_merge_for_tp(tensor, split_dim, merge_dim, merge_shapes=None):
    """
    Slice the tensor along split_dim, and concatenate along merge_dim. Does the opposite in backward.
    """

    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing():
        return ScatterAndMergeForTP.apply(tensor, split_dim, merge_dim, merge_shapes)
    else:
        return narrow_and_scale_for_tracing(tensor, split_dim, merge_dim)


def fwd_allreduce_for_tp(tensor):
    """
    Allreduce tensors in forward, no-op in backward. Can be replaced with fused_allgather + reduce_scatter
    combination, when they appear back-to-back, for efficiency.
    """

    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing():
        return ForwardAllreduceForTP.apply(tensor)
    else:
        return tensor


def bwd_allreduce_for_tp(tensor):
    """
    No-op in forward, allreduce in backward.
    """

    from smdistributed.modelparallel.torch.state_mod import state

    if not state.is_tracing():
        return BackwardAllreduceForTP.apply(tensor)
    else:
        return tensor


def get_merge_shapes(input_tensor_or_size, split_dim=None):
    """
    Get the merge shapes for all tp ranks. The merge shape is calculated based on the way we general local channels in get_local_channels function
    Used in 2 cases:
    1. The merge dimension in backward call is the split dimension in the forward call.
       if the forward split demension is not divisible by the tp_size(), the backward merge dimension
       will be different for each tp rank .
    2. In back-to-back matrix multiplication when optimize==memory, we have 2 scatter_and_merge_for_tp calls.
       The second call is to merge the dimension that splited in the first call. If it is an uneven split we need to know the shapes.
    """
    if isinstance(input_tensor_or_size, torch.Tensor):
        split_size = input_tensor_or_size.size()[split_dim]
    else:
        split_size = input_tensor_or_size
    return [get_local_channels(split_size, rank=r) for r in range(tp_size())]


class NarrowForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, batch_dim, shift, bwd_allgather):
        ctx.batch_dim = batch_dim
        ctx.shift = shift
        ctx.split_shapes = [
            get_local_channels(input_tensor.size()[batch_dim], rank=r) for r in range(tp_size())
        ]
        ctx.bwd_allgather = bwd_allgather

        if shift > ctx.split_shapes[tp_rank()]:
            raise ValueError(
                f"Shift {shift} must be less than or equal to the local size {ctx.split_shapes[tp_rank()]}."
            )

        if shift < 0:
            ctx.split_shapes[0] += shift
        elif shift > 0:
            ctx.split_shapes[tp_size() - 1] -= shift

        # eg. start position for tp_rank 2 is ctx.split_shapes[0] + ctx.split_shapes[1]
        return input_tensor.narrow(
            batch_dim, sum(ctx.split_shapes[: tp_rank()]), ctx.split_shapes[tp_rank()]
        ).contiguous()

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        if not ctx.bwd_allgather:
            return grad

        allgathered = _allgather_impl(grad, ctx.batch_dim, split_shapes=ctx.split_shapes)

        if ctx.shift == 0:
            return allgathered, None, None, None
        elif ctx.shift < 0:
            shape = list(allgathered.shape)
            shape[ctx.batch_dim] += 1
            allgathered.resize_(*shape)
            allgathered.narrow(ctx.batch_dim, shape[ctx.batch_dim] - 1, 1).zero_()
            return allgathered, None, None, None
        else:
            zeros_shape = list(allgathered.shape)
            zeros_shape[ctx.batch_dim] = ctx.shift
            device = torch.device("cuda", local_rank())
            zeros = torch.zeros(*zeros_shape, dtype=allgathered.dtype, device=device)
            return torch.cat((zeros, allgathered), dim=ctx.batch_dim), None, None, None


class AllgatherForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, batch_dim, merge_shapes):
        ctx.batch_dim = batch_dim
        return _allgather_impl(input_tensor, batch_dim, split_shapes=merge_shapes)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        local_bs = get_local_channels(grad.size(ctx.batch_dim))
        return grad.narrow(ctx.batch_dim, tp_rank() * local_bs, local_bs), None, None


class FusedAllgatherForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, batch_dim, merge_shapes):
        ctx.batch_dim = batch_dim
        return _allgather_impl(input_tensor, batch_dim, split_shapes=merge_shapes)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        return _reduce_scatter_impl(grad, ctx.batch_dim), None, None


class ForwardAllreduceForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, tensor):
        from smdistributed.modelparallel.torch.comm import get_tp_process_group
        from smdistributed.modelparallel.torch.state_mod import state

        with state.nccl_throttler.throttle():
            torch.distributed.all_reduce(tensor, group=get_tp_process_group())
        return tensor

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        return grad


class BackwardAllreduceForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        from smdistributed.modelparallel.torch.comm import get_tp_process_group
        from smdistributed.modelparallel.torch.state_mod import state

        with state.nccl_throttler.throttle():
            torch.distributed.all_reduce(grad, group=get_tp_process_group())
        return grad


class ReduceScatterForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, batch_dim, split_shapes, shift, transpose):
        ctx.batch_dim = batch_dim
        ctx.shift = shift
        ctx.transpose = transpose
        if transpose:
            input_tensor.transpose_(*transpose)

        if split_shapes == None:
            # either the shape is divisible by tp_size or we use the get_local_channels() to generate the local channels
            ctx.split_shapes = [
                get_local_channels(input_tensor.size()[batch_dim], rank=r) for r in range(tp_size())
            ]
        else:
            ctx.split_shapes = split_shapes
        return _reduce_scatter_impl(input_tensor, batch_dim, split_shapes=split_shapes, shift=shift)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        if ctx.shift != 0:
            if ctx.shift < 0:
                ctx.split_shapes[0] += ctx.shift
            else:
                ctx.split_shapes[tp_size() - 1] -= ctx.shift

        allgathered = _allgather_impl(grad, ctx.batch_dim, split_shapes=ctx.split_shapes)
        if ctx.shift == 0:
            if ctx.transpose:
                allgathered.transpose_(*reversed(ctx.transpose))
            return allgathered, None, None, None, None
        elif ctx.shift < 0:
            shape = list(allgathered.shape)
            shape[ctx.batch_dim] += 1
            allgathered.resize_(*shape)
            allgathered.narrow(ctx.batch_dim, shape[ctx.batch_dim] - 1, 1).zero_()
            if ctx.transpose:
                allgathered.transpose_(*reversed(ctx.transpose))
            return allgathered, None, None, None, None
        else:
            zeros_shape = list(allgathered.shape)
            zeros_shape[ctx.batch_dim] = ctx.shift
            device = torch.device("cuda", local_rank())
            zeros = torch.zeros(*zeros_shape, dtype=allgathered.dtype, device=device)
            cat = torch.cat((zeros, allgathered), dim=ctx.batch_dim)
            if ctx.transpose:
                cat.transpose_(*reversed(ctx.transpose))
            return cat, None, None, None, None


class ScatterAndMergeForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, split_dim, merge_dim, merge_shapes):
        from smdistributed.modelparallel.torch.state_mod import state

        ctx.split_dim = split_dim
        ctx.merge_dim = merge_dim
        ctx.bwd_merge_shapes = get_merge_shapes(input_tensor, split_dim=split_dim)

        with state.nccl_throttler.throttle():
            return state.comm.scatter_and_merge_tensor(
                input_tensor,
                split_axis=split_dim,
                merge_axis=merge_dim,
                group=CommGroup.TP_GROUP,
                merge_shapes=merge_shapes,
            )

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        from smdistributed.modelparallel.torch.state_mod import state

        with state.nccl_throttler.throttle():
            return (
                state.comm.scatter_and_merge_tensor(
                    grad,
                    split_axis=ctx.merge_dim,
                    merge_axis=ctx.split_dim,
                    group=CommGroup.TP_GROUP,
                    merge_shapes=ctx.bwd_merge_shapes,
                ),
                None,
                None,
                None,
            )


def _maybe_pad_tensor_for_tp_slicing(tensor, pad_size, dim):
    """Pad zeros at the end if this ranks have fewer local channels at the corresponding dim."""
    dim = len(tensor.size()) + dim if dim < 0 else dim
    orig_size = tensor.size()[dim]
    if orig_size > pad_size:
        raise ValueError(
            f"When padding the tensor, orig_size {orig_size} must be smaller or equal to pad_size {pad_size}"
        )

    if orig_size == pad_size:
        return tensor
    else:
        tensor_size = tensor.size()
        padded_shape = [0 for _ in range(len(tensor_size) * 2)]
        # padded_shape is starting from the last dimension
        pad_dim = len(tensor_size) - 1 - dim
        # pad an addition zero at end
        padded_shape[2 * pad_dim + 1] = pad_size - orig_size
        return F.pad(input=tensor, pad=tuple(padded_shape), mode="constant", value=0).contiguous()


def _maybe_unpad_tensor(tensor, orig_size, dim):
    """Remove the padded zero from tensor at dim"""
    return tensor.narrow(dim, 0, orig_size).contiguous()


def _needs_pad(split_shapes, batch_dim_shape):
    return any([shape != batch_dim_shape for shape in split_shapes])


# Gerneral guidance for using _base collectives
# _reduce_scatter_base https://github.com/pytorch/pytorch/blob/v1.10.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1591
# _all_gather_base https://github.com/pytorch/pytorch/blob/v1.10.1/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1933
# 1. _all_gather_base needs PT >= 1.9 and _reduce_scatter_base needs PT >= 1.10;
# 2. Only use it when batch_dim == 0, otherwise the underlaying nccl call won't work correctly
# since it will only check the the total elements equal and read the tensor as data pointers;
# 3. For _reduce_scatter_base, do not use it when padding is required as after padding
# we need to concatenate again, which is same as the original reduce_scatter
def _reduce_scatter_impl(input_tensor, batch_dim, split_shapes=None, shift=0):
    from smdistributed.modelparallel.torch.comm import get_tp_process_group
    from smdistributed.modelparallel.torch.state_mod import state

    bs_size = input_tensor.size(batch_dim)
    if split_shapes != None:
        if sum(split_shapes) != bs_size:
            raise ValueError(f"Dimension size {bs_size} can not be splitted by {split_shapes}")
        if len(split_shapes) != tp_size():
            raise ValueError(
                f"Split shapes {split_shapes} must have same length of tp_size {tp_size()}"
            )
    else:
        split_shapes = [get_local_channels(bs_size, rank=r) for r in range(tp_size())]

    batch_dim_shape = max(split_shapes)

    use_reduce_scatter_base = (
        LooseVersion(torch.__version__) >= LooseVersion("1.10.0")
        and batch_dim == 0
        and not _needs_pad(split_shapes, batch_dim_shape)
    )

    # Split the input tensor among batch_dim
    # Directly use input_tensor when use _reduce_scatter_base
    if not use_reduce_scatter_base:
        input_tensor_split = []
        start = 0
        for r in range(tp_size()):
            if shift < 0 and r == 0:
                split_shapes[r] = split_shapes[r] + shift
            elif shift > 0 and r == tp_size() - 1:
                split_shapes[r] = split_shapes[r] - shift

            input_tensor_split.append(
                input_tensor.narrow(batch_dim, start, split_shapes[r]).contiguous()
            )
            start += split_shapes[r]

        input_tensor_split = [
            _maybe_pad_tensor_for_tp_slicing(t, batch_dim_shape, batch_dim)
            for r, t in enumerate(input_tensor_split)
        ]

    output_shape = list(input_tensor.shape)
    output_shape[batch_dim] = max(split_shapes)

    output_tensor = torch.empty(
        *output_shape, dtype=input_tensor.dtype, device=torch.device("cuda", local_rank())
    )

    with state.nccl_throttler.throttle():
        if use_reduce_scatter_base:
            torch.distributed._reduce_scatter_base(
                output_tensor, input_tensor, group=get_tp_process_group()
            )
        else:
            torch.distributed.reduce_scatter(
                output_tensor, input_tensor_split, group=get_tp_process_group()
            )

    return _maybe_unpad_tensor(output_tensor, split_shapes[tp_rank()], batch_dim)


def _allgather_impl(input_tensor, batch_dim, split_shapes=None):
    from smdistributed.modelparallel.torch.comm import get_tp_process_group
    from smdistributed.modelparallel.torch.state_mod import state

    use_allgather_base = LooseVersion(torch.__version__) >= LooseVersion("1.9.0") and batch_dim == 0

    if split_shapes != None:
        if len(split_shapes) != tp_size():
            raise ValueError(
                f"Split shapes {split_shapes} must have same length of tp_size {tp_size()}"
            )
    else:
        # If there is not split shapes, all tp ranks should share the same shape in batch_dim
        split_shapes = [input_tensor.size()[batch_dim] for _ in range(tp_size())]

    recv_shape = list(input_tensor.size())
    batch_dim_shape = max(split_shapes)
    recv_shape[batch_dim] = batch_dim_shape

    if batch_dim == 0:
        # Create the receive tensors as the views of output_tensor
        # Only works when batch_dim==0 othervise the narrowed tensors will be non-contiguous
        # Will help saving memory usage when padding is not required
        recv_shape[batch_dim] = batch_dim_shape * tp_size()
        output_tensor = torch.empty(
            *recv_shape, dtype=input_tensor.dtype, device=torch.device("cuda", local_rank())
        ).contiguous()
        output_list = [
            output_tensor.narrow(batch_dim, batch_dim_shape * r, batch_dim_shape)
            for r in range(tp_size())
        ]
    else:
        output_list = [
            torch.empty(
                *recv_shape, dtype=input_tensor.dtype, device=torch.device("cuda", local_rank())
            )
            for r in range(tp_size())
        ]

    input_tensor = _maybe_pad_tensor_for_tp_slicing(input_tensor, batch_dim_shape, batch_dim)

    with state.nccl_throttler.throttle():
        if use_allgather_base:
            torch.distributed._all_gather_base(output_tensor, input_tensor, get_tp_process_group())
        else:
            torch.distributed.all_gather(output_list, input_tensor, get_tp_process_group())

    if batch_dim == 0:
        # If there is no padding, the output_tensor contains no extra data, should be the real output
        # This could avoid the concatenate which will create a copy
        if not _needs_pad(split_shapes, batch_dim_shape):
            return output_tensor

    output_list = [
        _maybe_unpad_tensor(t, split_shapes[r], batch_dim) for r, t in enumerate(output_list)
    ]

    if batch_dim == 0:
        return cat_cuda_dim0(output_list)
    else:
        return torch.cat(output_list, batch_dim)


# def _allgather_impl(input_tensor, batch_dim):
#    from smdistributed.modelparallel.torch.comm import get_tp_process_group
#    from smdistributed.modelparallel.torch.state_mod import state
#
#    local_bs = input_tensor.size(batch_dim)
#    output_shape = list(input_tensor.shape)
#    output_shape[0], output_shape[batch_dim] = output_shape[batch_dim], output_shape[0]
#    output_shape[0] = local_bs * tp_size()
#
#    output_tensor = torch.empty(
#        *output_shape, dtype=input_tensor.dtype, device=torch.device("cuda", local_rank())
#    )
#    output_list = [output_tensor.narrow(0, r * local_bs, local_bs) for r in range(tp_size())]
#
#    transposed_input = input_tensor.transpose(0, batch_dim).contiguous()
#
#    # output_tensor gets populated here because it shares the same underlying memory with output_list
#    with state.nccl_throttler.throttle():
#        torch.distributed.all_gather(output_list, transposed_input, get_tp_process_group())
#
#    return output_tensor.transpose(0, batch_dim).contiguous()
