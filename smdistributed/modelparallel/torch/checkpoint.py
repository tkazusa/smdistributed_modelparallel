# Standard Library
import glob
import re

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import (
    mp_rank,
    mp_size,
    pp_rank,
    pp_size,
    rank,
    rdp_rank,
    rdp_size,
    size,
    tp_rank,
    tp_size,
)

logger = get_logger()


def is_3part_ckpt_file(prefix):
    # checkpoint format (v3): {ckpt_name}_{pp_rank}_{tp_rank}_{rdp_rank}
    return ckpt_file_matches(prefix, f"{prefix}_((\d)+)_((\d)+)_((\d)+)$")


def is_2part_ckpt_file(prefix):
    # checkpoint format (v2): {ckpt_name}_{pp_rank}_{tp_rank}
    return ckpt_file_matches(prefix, f"{prefix}_((\d)+)_((\d)+)$")


def ckpt_file_matches(prefix, regex):
    matching_files = glob.glob(f"{prefix}_*")
    # sagemaker creates temp files which sometimes show up in S3, ignore those
    matching_files = [x for x in matching_files if ".sagemaker-" not in x]
    num_parts = len(matching_files)
    if num_parts == 0:
        raise RuntimeError(f"Could not find partial checkpoint files with prefix {prefix}")
    filename = matching_files[0]
    matches = re.search(regex, filename)
    return True if matches else False


def _validate_num_parts_checkpoint(prefix):
    matching_files = glob.glob(f"{prefix}_*")
    # sagemaker creates temp files which sometimes show up in S3, ignore those
    matching_files = [x for x in matching_files if ".sagemaker-" not in x]
    num_parts = len(matching_files)

    if num_parts == 0:
        raise RuntimeError(f"Could not find partial checkpoint files with prefix {prefix}")

    for filename in matching_files:
        ckpt_file_v3 = re.search(f"{prefix}_((\d)+)_((\d)+)_((\d)+)$", filename)
        ckpt_file_v2 = re.search(f"{prefix}_((\d)+)_((\d)+)$", filename)
        ckpt_file_v1 = re.search(f"{prefix}_((\d)+)$", filename)
        if ckpt_file_v1:
            # For backward compatibility, with checkpoint files of format ckpt_{pp_rank}
            num_tp_parts = 1
            num_pp_parts = num_parts
            rank = int(ckpt_file_v1.group(1))
            if rank >= num_parts:
                raise RuntimeError(
                    f"The rank of a checkpoint file ({rank}) exceeds the total number of checkpoint files ({num_parts}) matching given prefix ({prefix}). Please ensure all checkpoint parts are available for loading at the given location. "
                )
        elif ckpt_file_v2:
            num_tp_parts = num_parts // pp_size()
            num_pp_parts = num_parts // tp_size()
            pp_rank = int(ckpt_file_v2.group(1))
            tp_rank = int(ckpt_file_v2.group(3))
            if pp_rank >= num_pp_parts:
                raise RuntimeError(
                    f"The pp_rank of a checkpoint file ({pp_rank}) exceeds the total number of unique pp_rank checkpoint files ({num_pp_parts}) matching given prefix ({prefix}). Please ensure all checkpoint parts are available for loading at the given location."
                )
            if tp_rank >= num_tp_parts:
                raise RuntimeError(
                    f"The tp_rank of a checkpoint file ({tp_rank}) exceeds the total number of unique tp_rank checkpoint files ({num_tp_parts}) matching given prefix ({prefix}). Please ensure all checkpoint parts are available for loading at the given location."
                )
        elif ckpt_file_v3:
            num_tp_parts = num_parts // (pp_size() * rdp_size())
            num_pp_parts = num_parts // (tp_size() * rdp_size())
            num_rdp_parts = num_parts // (tp_size() * pp_size())
            pp_rank = int(ckpt_file_v3.group(1))
            tp_rank = int(ckpt_file_v3.group(3))
            rdp_rank = int(ckpt_file_v3.group(5))
            if pp_rank >= num_pp_parts:
                raise RuntimeError(
                    f"The pp_rank of a checkpoint file ({pp_rank}) exceeds the total number of unique pp_rank checkpoint files ({num_pp_parts}) matching given prefix ({prefix}). Please ensure all checkpoint parts are available for loading at the given location."
                )
            if tp_rank >= num_tp_parts:
                raise RuntimeError(
                    f"The tp_rank of a checkpoint file ({tp_rank}) exceeds the total number of unique tp_rank checkpoint files ({num_tp_parts}) matching given prefix ({prefix}). Please ensure all checkpoint parts are available for loading at the given location."
                )
            if rdp_rank >= num_rdp_parts:
                raise RuntimeError(
                    f"The rdp_rank of a checkpoint file ({rdp_rank}) exceeds the total number of unique rdp_rank checkpoint files ({num_rdp_parts}) matching given prefix ({prefix}). Please ensure all checkpoint parts are available for loading at the given location."
                )

    expected_num_parts = size() if ckpt_file_v3 else mp_size()
    if (
        (ckpt_file_v3 and num_rdp_parts != rdp_size())
        or num_pp_parts != pp_size()
        or num_tp_parts != tp_size()
        or num_parts != expected_num_parts
    ):
        msg = f"{pp_size()} (pipeline) partitions, {tp_size()} (tensor parallel) partitions, and {rdp_size()} reduced-data-parallel ranks"
        raise RuntimeError(
            f"Can not load a {num_parts} parts checkpoint to a model with {msg}. Please save the full model and load that instead when changing the number of partitions across runs. Matching files: {matching_files}"
        )


def save(obj, f, partial=True, v3=False, **kwargs):
    """
    SMP version of torch.save
    Args:
        obj: dict, the dictionary to save
        f: str, file path to save the checkpoint
        partial: bool, set True so that each mp rank will save a separate file
    """
    if partial:
        if v3:
            # v3 saving format, current use case is to skip the gather of opt states when sharding is enabled
            f = f"{f}_{pp_rank()}_{tp_rank()}_{rdp_rank()}"
        else:
            # v2 saving format, designed to only save on rdp rank 0
            f = f"{f}_{pp_rank()}_{tp_rank()}"
    if rank() == 0 or partial:
        torch.save(obj, f, **kwargs)


def load(f, partial=True, **kwargs):
    """
    SMP version of torch.load
    Args:
        f: str, file path to load the checkpoint
        partial: bool, set True so that each mp rank will load a separate file
    """
    if partial:
        _validate_num_parts_checkpoint(f)
        if is_3part_ckpt_file(f):
            f = f"{f}_{pp_rank()}_{tp_rank()}_{rdp_rank()}"
        elif is_2part_ckpt_file(f):
            f = f"{f}_{pp_rank()}_{tp_rank()}"
        else:
            f = f"{f}_{pp_rank()}"
    obj = torch.load(f, **kwargs)
    return obj
