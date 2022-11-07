# Standard Library
import glob
import os
import re
import shutil
from distutils.version import LooseVersion

# Third Party
import torch

# First Party
from smdistributed.modelparallel import __version__ as smp_version
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.comm import WORLD, barrier, gather
from smdistributed.modelparallel.torch.core import (
    mp_rank,
    mp_size,
    param_shard_rank,
    pp_rank,
    pp_size,
    rank,
    rdp_rank,
    rdp_size,
    size,
    tp_rank,
    tp_size,
)
from smdistributed.modelparallel.torch.exceptions import (
    IncompatibleCheckpointFoundError,
    IncompatibleCheckpointRankFoundError,
    MissingCheckpointFilesError,
    SMPCheckpointError,
    SMPValidationError,
)
from smdistributed.modelparallel.torch.model import DistributedModel
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import check_env_var_truthy, rmsg

logger = get_logger()


def is_3part_ckpt_file(prefix, back_compat=False):
    # checkpoint format (v3): {ckpt_name}_{pp_rank}_{tp_rank}_{rdp_rank}
    v3_str = "_((\d)+)_((\d)+)_((\d)+)$" if back_compat else "_((\d)+)_((\d)+)_((\d)+)\.pt$"
    return ckpt_file_matches(prefix, f"{prefix}{v3_str}")


def is_2part_ckpt_file(prefix, back_compat=False):
    # checkpoint format (v2): {ckpt_name}_{pp_rank}_{tp_rank}
    v2_str = "_((\d)+)_((\d)+)$" if back_compat else "_((\d)+)_((\d)+)\.pt$"
    return ckpt_file_matches(prefix, f"{prefix}{v2_str}")


def ckpt_file_matches(prefix, regex):
    matching_files = glob.glob(f"{prefix}_*")
    # sagemaker creates temp files which sometimes show up in S3, ignore those
    matching_files = [x for x in matching_files if ".sagemaker-" not in x]
    num_parts = len(matching_files)
    if num_parts == 0:
        raise MissingCheckpointFilesError(prefix)
    filename = matching_files[0]
    matches = re.search(regex, filename)
    return True if matches else False


def _validate_num_parts_checkpoint(prefix, back_compat=False):
    matching_files = glob.glob(f"{prefix}_*")
    # sagemaker creates temp files which sometimes show up in S3, ignore those
    matching_files = [x for x in matching_files if ".sagemaker-" not in x]
    num_parts = len(matching_files)

    if num_parts == 0:
        raise MissingCheckpointFilesError(prefix)

    v3_str = "_((\d)+)_((\d)+)_((\d)+)$" if back_compat else "_((\d)+)_((\d)+)_((\d)+)\.pt$"
    v2_str = "_((\d)+)_((\d)+)$" if back_compat else "_((\d)+)_((\d)+)\.pt$"
    v1_str = "_((\d)+)$" if back_compat else "_((\d)+)\.pt$"
    for filename in matching_files:
        ckpt_file_v3 = re.search(f"{prefix}{v3_str}", filename)
        ckpt_file_v2 = re.search(f"{prefix}{v2_str}", filename)
        ckpt_file_v1 = re.search(f"{prefix}{v1_str}", filename)
        if ckpt_file_v1:
            # For backward compatibility, with checkpoint files of format ckpt_{pp_rank}
            num_tp_parts = 1
            num_pp_parts = num_parts
            rank = int(ckpt_file_v1.group(1))
            if rank >= num_parts:
                raise IncompatibleCheckpointRankFoundError("rank", rank, num_parts, prefix)
        elif ckpt_file_v2:
            num_tp_parts = num_parts // pp_size()
            num_pp_parts = num_parts // tp_size()
            pp_rank = int(ckpt_file_v2.group(1))
            tp_rank = int(ckpt_file_v2.group(3))
            if pp_rank >= num_pp_parts:
                raise IncompatibleCheckpointRankFoundError("pp_rank", pp_rank, num_pp_parts, prefix)
            if tp_rank >= num_tp_parts:
                raise IncompatibleCheckpointRankFoundError("tp_rank", tp_rank, num_tp_parts, prefix)
        elif ckpt_file_v3:
            num_tp_parts = num_parts // (pp_size() * rdp_size())
            num_pp_parts = num_parts // (tp_size() * rdp_size())
            num_rdp_parts = num_parts // (tp_size() * pp_size())
            pp_rank = int(ckpt_file_v3.group(1))
            tp_rank = int(ckpt_file_v3.group(3))
            rdp_rank = int(ckpt_file_v3.group(5))
            if pp_rank >= num_pp_parts:
                raise IncompatibleCheckpointRankFoundError("pp_rank", pp_rank, num_pp_parts, prefix)
            if tp_rank >= num_tp_parts:
                raise IncompatibleCheckpointRankFoundError("tp_rank", tp_rank, num_tp_parts, prefix)
            if rdp_rank >= num_rdp_parts:
                raise IncompatibleCheckpointRankFoundError(
                    "rdp_rank", rdp_rank, num_rdp_parts, prefix
                )

    expected_num_parts = size() if ckpt_file_v3 else mp_size()
    if (
        (ckpt_file_v3 and num_rdp_parts != rdp_size())
        or num_pp_parts != pp_size()
        or num_tp_parts != tp_size()
        or num_parts != expected_num_parts
    ):
        raise IncompatibleCheckpointFoundError(num_parts, matching_files)


def save(obj, f, partial=True, v3=False, **kwargs):
    """
    SMP version of torch.save
    Args:
        obj: dict, the dictionary to save
        f: str, file path to save the checkpoint
        partial: bool, set True so that each rank will save a separate file, which rank to save is controlled by user
    """
    if partial:
        if v3:
            # v3 saving format, current use case is to skip the gather of opt states when sharding is enabled
            f = f"{f}_{pp_rank()}_{tp_rank()}_{rdp_rank()}.pt"
        else:
            # v2 saving format, designed to only save on rdp rank 0
            f = f"{f}_{pp_rank()}_{tp_rank()}.pt"
    if rank() == 0 or partial:
        torch.save(obj, f, **kwargs)


def load(f, partial=True, back_compat=False, **kwargs):
    """
    SMP version of torch.load
    Checkpoint name will contain the rank information, it will have the following format:
    - Old: name_rankinfo, name is defined by user
    - New: name_rankinfo.pt
    Args:
        f: str, file path to load the checkpoint
        partial: bool, set True so that each mp rank will load a separate file
        back_compat: load the old version checkpoint, which has format name.pt_rankinfo
    """
    if partial:
        _validate_num_parts_checkpoint(f, back_compat=back_compat)
        if is_3part_ckpt_file(f, back_compat=back_compat):
            f = f"{f}_{pp_rank()}_{tp_rank()}_{rdp_rank()}"
        elif is_2part_ckpt_file(f, back_compat=back_compat):
            f = f"{f}_{pp_rank()}_{tp_rank()}"
        else:
            f = f"{f}_{pp_rank()}"
    if partial and not back_compat:
        f = f"{f}.pt"
    obj = torch.load(f, **kwargs)
    return obj


def _check_valid_tag(tag):
    if not isinstance(tag, str):
        raise SMPCheckpointError(f"checkpoint tag can only be str, getting {type(tag)}")
    all_tags = gather(tag, WORLD)
    if rank() == 0:
        for tag in all_tags:
            if tag != all_tags[0]:
                raise SMPCheckpointError(
                    f"checkpoint tag should be same across ranks, getting {all_tags}"
                )


def save_checkpoint(
    path,
    tag,
    partial=True,
    model=None,
    optimizer=None,
    user_content=None,
    translate_if_full=True,
    num_kept_partial_checkpoints=None,
):
    """
    Save a checkpoint. Model and optimizer checkpoints will be saved as separate files. Checkpoint folder will have the following format:
    - path
        - tag_partial (folder)
            - model_rankinfo.pt
            - optimizer_rankinfo.pt
            - fp16_states_rankinfo.pt
            - user_content.pt
        - tag (checkpoint file for full checkpoint)
        - tag_user_content.pt (user_content file for full checkpoint)
        - newest (a file that indicates the newest checkpoint)
    Args:
        path: Path to save the checkpoint. Smp will create directories if it does not exist
        tag: A tag for the current checkpoint, ususally the train steps. Note: tag needs to be same across ranks
        partial: Whether to save the parial checkpoint
        model: The model to save. It needs to be smp.DistributedModel
        optimizer: The optimizer to save. It needs to be smp.DistributedOptimizer
        user_content: User defined content to save
        translate_if_full: Whether to translate the full state_dict to HF state_dict if possible
        num_kept_partial_checkpoints: The max number of partial checkpoint to keep on disk
    """
    if model is None and optimizer is None:
        logger.warn("Both model and optimizer are None, skipping saving checkpoint...")
        return

    if num_kept_partial_checkpoints is not None:
        if num_kept_partial_checkpoints <= 0:
            logger.warn(
                f"Invalid num_kept_partial_checkpoints number {num_kept_partial_checkpoints}, ignoring..."
            )
            num_kept_partial_checkpoints = None

    if state.cfg.zero2d_enabled():
        if not partial:
            logger.warn(
                f"Sharded data parallelism does not support to save full checkpoint, saving partial instead..."
            )
        if state.optimizer is None:
            raise SMPValidationError(
                "When sharded data parallelism is enabled, smp.save_checkpoint call be called only after smp.DistributeOptimizer is created"
            )
        # Prepare for checkpoint save by ensuring all parameters are partitioned
        state.optimizer.checkpoint_event_prologue()

    ckpt_type = "partial" if partial else "full"
    logger.info(rmsg(f"Saving {ckpt_type} checkpoint with tag {tag} to {path}"))

    _check_valid_tag(tag)
    if partial:
        tag = f"{tag}_{ckpt_type}"
        # Ensure save_dir directory exists
        os.makedirs(os.path.join(path, tag), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

    barrier()

    if model is not None:
        if state.cfg.zero2d_enabled():
            save_model_checkpoint_zero2d(model, path, tag=tag)
        else:
            save_model_checkpoint(
                model, path, tag=tag, partial=partial, translate_if_full=translate_if_full
            )

    if optimizer is not None:
        if state.cfg.zero2d_enabled():
            save_optimizer_checkpoint_zero2d(optimizer, path, tag=tag)
        else:
            if partial:
                save_optimizer_checkpoint(optimizer, path, tag=tag)
            else:
                logger.warn("Saving optimizer for full checkpoint is not supported, skipping...")

    if state.cfg.zero2d_enabled():
        state.optimizer.checkpoint_event_epilogue()

    if user_content is not None:
        if partial:
            file = os.path.join(path, tag, "user_content.pt")
        else:
            file = os.path.join(path, f"user_content_{tag}")
        save(user_content, file, partial=False)

    # Save smp config
    if partial:
        smp_config = state.cfg.get_config_dict()
        smp_config["smp_version"] = smp_version
        file = os.path.join(path, tag, "smp_config.pt")
        save(smp_config, file, partial=False)

    if rank() == 0 and partial:
        newest_file = os.path.join(path, "newest")
        if os.path.isfile(newest_file):
            with open(newest_file, "r") as fd:
                existing_files = fd.readlines()
        else:
            existing_files = []
        with open(newest_file, "w") as fd:
            # Remove the oldest checkpoints until existing_files meets num_kept_partial_checkpoints-1
            if num_kept_partial_checkpoints is not None:
                while len(existing_files) >= num_kept_partial_checkpoints:
                    oldest_dir = os.path.join(path, existing_files[0][:-1])
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                    existing_files = existing_files[1:]
            existing_files.append(f"{tag}\n")
            fd.writelines(existing_files)


def save_model_checkpoint_zero2d(model, path, tag=""):
    # Only the first parameter parallel group needs to store the optimizer state checkpoints for zero-2d
    if rank() == param_shard_rank():
        sd = state.model.module.state_dict()
        model_state = dict(
            module=sd, shard_size=state.cfg.sharded_data_parallel_degree, _smp_zero2d=True
        )
        file = os.path.join(path, tag, f"model_{param_shard_rank()}.pt")
        torch.save(model_state, file)
    barrier()


def save_optimizer_checkpoint_zero2d(optimizer, path, tag=""):
    # Only the first parameter parallel group needs to store the optimizer state checkpoints for zero-2d
    if rank() == param_shard_rank():
        zero_sd = optimizer.orig_state_dict()
        zero_sd["_smp_zero2d"] = True
        file = os.path.join(path, tag, f"optimizer_{param_shard_rank()}.pt")
        torch.save(zero_sd, file)
    barrier()


def save_model_checkpoint(
    model, path, tag="", partial=True, translate_if_full=True, translate_function=None
):
    if not isinstance(model, DistributedModel):
        raise SMPValidationError(
            "smp.save_checkpoint can only take smp.DistributedModel as input model"
        )
    if rdp_rank() == 0:
        if partial:
            model_checkpoint = model.local_state_dict()
        else:
            model_checkpoint = model.state_dict()
            if translate_if_full:
                if rank() == 0:
                    if translate_function is None:
                        for smp_to_hf, _ in state.module_manager.translate_functions:
                            model_checkpoint = smp_to_hf(model_checkpoint)
                    else:
                        model_checkpoint = translate_function(model_checkpoint)
            else:
                model_checkpoint["_smp_is_partial"] = False
        if partial:
            file = os.path.join(path, tag, "model")
        else:
            file = os.path.join(path, tag)
        save(model_checkpoint, file, partial=partial)
        logger.info(rmsg("model checkpoint saved."))
    barrier()


def save_optimizer_checkpoint(optimizer, path, tag=None):
    if not hasattr(optimizer, "orig_load_state_dict"):
        raise SMPValidationError(
            "smp.save_checkpoint can only take smp.DistributeOptimizer as input optimizer"
        )
    if state.cfg.shard_optimizer_state or rdp_rank() == 0:
        optimizer_state_dict = optimizer.local_optimizer_state_dict()
        file = os.path.join(path, tag, "optimizer_states")
        save(optimizer_state_dict, file, v3=state.cfg.shard_optimizer_state)
        optimizer_state_dict = None
        if state.cfg.fp16:
            fp16_states = optimizer.local_fp16_state_dict()
            file = os.path.join(path, tag, "fp16_states")
            save(fp16_states, file, v3=state.cfg.shard_optimizer_state)
        logger.info(rmsg("optimizer checkpoint saved."))
    barrier()


def validate_zero2d_loading(partial):
    if state.optimizer is None:
        raise SMPValidationError(
            "When sharded data parallelism is enabled, smp.resume_from_checkpoint should be called after optimizer is wrapped with smp.DistributeOptimizer"
        )
    if not partial:
        raise SMPValidationError(
            "When sharded data parallelism is enabled, smp.resume_from_checkpoint only support to load partial checkpoint. If you want to load a full model, you can load it into the orginal torch model using model.load_state_dict before smp.DistributedModel wrapper"
        )


def resume_from_checkpoint(
    path,
    tag=None,
    partial=True,
    strict=True,
    load_optimizer=True,
    load_sharded_optimizer_state=True,
    translate_function=None,
):
    """
    Resume from a checkpoint.
    Args:
        path: Path to load the checkpoint.
        tag: Tag of the checkpoint to resume. If not provided, smp will try to locate the newest checkpoint from the saved newest file
        partial: Whether to load the partial checkpoint
        strict: Load with strict load, no extra key or missing key is allowed
        load_optimizer: Whether to load optimizer
        load_sharded_optimizer_state: Only used for sharded data parallelism. When this is False, zero-2d will only load the fp16 states but not the optimizer states
        translate_function: function to translate the full checkpoint into smp format. For supported models this is not required.
    """
    if state.cfg.zero2d_enabled():
        validate_zero2d_loading(partial)

    ckpt_type = "partial" if partial else "full"
    if tag is None:
        if not partial:
            raise SMPValidationError("tag is required to load a full checkpoint!")
        newest_path = os.path.join(path, "newest")
        if os.path.isfile(newest_path):
            with open(newest_path, "r") as fd:
                tag = fd.readlines()[-1][:-1]
        else:
            raise SMPValidationError(
                f"Unable to find newest file at {newest_path}, if trying to load newest "
                "checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint."
            )
        user_tag = tag[: -(len(ckpt_type) + 1)]
    else:
        user_tag = tag
        if partial:
            tag = user_tag + "_" + ckpt_type

    if partial and ckpt_type not in tag:
        raise SMPValidationError(
            f"The newest checkpoint file is not saved with `{ckpt_type}` postfix, this might be a checkpoint saved with smp <= 1.10, please load with the back compatible APIs."
        )

    logger.info(f"Resuming from {ckpt_type} checkpoint with tag {user_tag} from {path}")

    checkpoint_path = os.path.join(path, tag)
    if not os.path.exists(checkpoint_path):
        raise SMPValidationError(f"Checkpoint {checkpoint_path} does not exist!")

    def _should_verify_config():
        return check_env_var_truthy("SMP_VERIFY_CHECKPOINT_CONFIG", "1")

    # Verify smp config
    if partial and state.initialized and rank() == 0 and _should_verify_config():
        file = os.path.join(path, tag, "smp_config.pt")
        saved_smp_config = load(file, partial=False)
        verify_smp_config(saved_smp_config, partial=partial, load_optimizer=load_optimizer)
    elif not state.initialized and rank() == 0:
        logger.warning(
            "smp.resume_from_checkpoint is called before smp.init, skipping smp config validation..."
        )

    if state.cfg.zero2d_enabled():
        # Prepare for checkpoint load by ensuring all parameters are partitioned
        state.optimizer.checkpoint_event_prologue()

    if state.model is None:
        state.loaded_model_state = (path, tag, partial, strict, translate_function)
    else:
        if state.cfg.zero2d_enabled():
            state.model.load_model_checkpoint_zero2d(path, tag, strict)
        else:
            state.model.load_model_checkpoint(path, tag, partial, strict, translate_function)

    if load_optimizer:
        if partial is False:
            logger.warning(f"Skipping loading optimizer for full checkpoint loading...")
        else:
            if state.optimizer is None:
                state.loaded_optimizer_state = (path, tag)
            else:
                # Load with zero optimizer will always reach this if condition
                if state.cfg.zero2d_enabled():
                    state.optimizer.load_optimizer_checkpoint_zero2d(
                        path, tag, load_sharded_optimizer_state
                    )
                else:
                    state.optimizer.load_optimizer_checkpoint(path, tag)

    if state.cfg.zero2d_enabled():
        state.optimizer.checkpoint_event_epilogue()

    user_content = None
    if partial:
        user_content_path = os.path.join(path, tag, "user_content.pt")
    else:
        user_content_path = os.path.join(path, f"user_content_{tag}")
    if os.path.isfile(user_content_path):
        user_content = load(user_content_path, partial=False)
    return user_content


def verify_smp_config(saved_smp_config, partial=True, load_optimizer=True):
    mismatch = {}
    smp_config = state.cfg.get_config_dict()

    if LooseVersion(saved_smp_config["smp_version"]) < LooseVersion("1.10.0"):
        raise SMPValidationError(
            f"Checkpoint was saved with smp version {saved_smp_config['smp_version']} < 1.10.0, which is not supported with this checkpointing API."
        )
    elif saved_smp_config["smp_version"] != smp_version:
        logger.warning(
            f"WARNING: Checkpoint was saved with a different version of smp, this might potentially cause issues. Checkpoint version: {saved_smp_config['smp_version']}, Current smp version: {smp_version}."
        )
    del saved_smp_config["smp_version"]

    for key, val in saved_smp_config.items():
        if saved_smp_config[key] != smp_config[key]:
            mismatch[key] = (saved_smp_config[key], smp_config[key])

    # zero-2d does not support elastic_checkpoint now, need to enforce same sharded_data_parallel_degree
    if (load_optimizer and partial) or state.cfg.zero2d_enabled():
        violation_features = set(
            [
                "pipeline_parallel_degree",
                "tensor_parallel_degree",
                "shard_optimizer_state",
                "sharded_data_parallel_degree",
            ]
        )
        intersection = violation_features.intersection(set(mismatch.keys()))
        if len(intersection) > 0:
            violation_features_str = "".join(
                [
                    f"feature {f}, loaded {mismatch[f][0]}, current {mismatch[f][1]}\n"
                    for f in intersection
                ]
            )
            raise SMPValidationError(
                f"The following features changes are not allowed for loading partial: \n{violation_features_str}"
            )

    if len(mismatch) > 0:
        mismatch_str = "".join(
            [
                f"feature {f}, loaded {mismatch[f][0]}, current {mismatch[f][1]}\n"
                for f in mismatch.keys()
            ]
        )
        logger.warning(
            f"Warning: Found mismatched features between save and load: \n{mismatch_str}"
        )
