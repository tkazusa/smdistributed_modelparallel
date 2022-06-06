# Third Party
# Standard Library
from distutils.version import LooseVersion

import torch
import torch.distributed as dist

# File downloaded as is from https://raw.githubusercontent.com/pytorch/pytorch/1.7/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py
# This is not packaged with the binary for some reason

_pt_19_or_newer = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")
_pt_110_or_newer = LooseVersion(torch.__version__) >= LooseVersion("1.10.0")


def allreduce_hook(
    process_group: object, bucket: dist._GradBucket if not _pt_19_or_newer else dist.GradBucket
) -> torch.futures.Future:
    """
        This DDP communication hook just calls ``allreduce`` using ``GradBucket``
        tensors. Once gradient tensors are aggregated across all workers, its ``then``
        callback takes the mean and returns the result. If user registers this hook,
        DDP results is expected to be same as the case where no hook was registered.
        Hence, this won't change behavior of DDP and user can use this as a reference
        or modify this hook to log useful information or any other purposes while
        unaffecting DDP behavior.

        Example::
            >>> ddp_model._register_comm_hook(process_group, allreduce_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = process_group.size() if process_group is not None else dist.get_world_size()

    if _pt_110_or_newer:
        tensor = bucket.buffer()
    elif _pt_19_or_newer:
        tensor = bucket.get_tensor()
    else:
        tensor = bucket.get_tensors()[0]
    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()

    def then_callback(fut):
        return [fut.value()[0].div_(world_size)]

    return fut.then(then_callback)


def fp16_compress_hook(
    process_group: object, bucket: dist._GradBucket if not _pt_19_or_newer else dist.GradBucket
):
    """
        This DDP communication hook implements a simple gradient compression
        approach that converts ``GradBucket`` tensors whose type is assumed to be
        ``torch.float32`` to half-precision floating point format (``torch.float16``).
        It allreduces those ``float16`` gradient tensors. Once compressed gradient
        tensors are allreduced, its then callback called ``decompress`` converts the
        aggregated result back to ``float32`` and takes the mean.

        Example::
            >>> ddp_model._register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = process_group.size() if process_group is not None else dist.get_world_size()

    if _pt_110_or_newer:
        compressed_tensor = bucket.buffer().to(torch.float16)
    elif _pt_19_or_newer:
        compressed_tensor = bucket.get_tensor().to(torch.float16)
    else:
        compressed_tensor = bucket.get_tensors()[0].to(torch.float16)

    fut = dist.all_reduce(compressed_tensor, group=group_to_use, async_op=True).get_future()

    def decompress(fut):
        return [fut.value()[0].to(torch.float32).div_(world_size)]

    return fut.then(decompress)


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        torch.zeros_like(
            all_gather_in_list, device=all_gather_in_list.device, dtype=all_gather_in_list.dtype
        )
        for _ in range(world_size)
    ]
    return out_list


def _allgather_then_aggregate_hook(
    process_group: object, bucket: dist._GradBucket if not _pt_19_or_newer else dist.GradBucket
) -> torch.futures.Future:
    """
        Similar to ``allreduce_hook``, this hook first gathers ``GradBucket`` tensors
        and its ``then`` callback aggregates the gathered gradient tensors and takes
        mean. Instead of ``allreduce`` this hook uses ``allgather``. Note that with
        W workers, both the computation and communication time scale as O(W) for
        allgather compared to O(logW) for allreduce. Therefore, this hook is expected
        to be much slower than ``allreduce_hook`` although both essentially do the
        same thing with the gradients.

        .. warning ::
            This is for test and experiments. User is suggested to use a faster
            alternative called ``allreduce_hook``  that uses ``allreduce`` protocol
            instead of ``allgather`` protocol.

        Example::
            >>> ddp_model._register_comm_hook(process_group, allreduce_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = process_group.size() if process_group is not None else dist.get_world_size()

    if _pt_110_or_newer:
        tensor = bucket.buffer()
    elif _pt_19_or_newer:
        tensor = bucket.get_tensor()
    else:
        tensor = bucket.get_tensors()[0]

    fut = dist.all_gather(
        _get_allgather_out_list(tensor, world_size), tensor, group=group_to_use, async_op=True
    ).get_future()

    def aggregate(fut):
        all_ranks_tensor = fut.value()[0]
        if _pt_110_or_newer:
            tensor = bucket.buffer()
        elif _pt_19_or_newer:
            tensor = bucket.get_tensor()
        else:
            tensor = bucket.get_tensors()[0]
        for r, gathered_tensor in enumerate(all_ranks_tensor):
            if r != rank:
                tensor += gathered_tensor

        return [tensor.div_(world_size)]

    return fut.then(aggregate)
