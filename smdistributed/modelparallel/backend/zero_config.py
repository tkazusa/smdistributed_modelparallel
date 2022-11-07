# Standard Library
import collections.abc
import copy
import json
import os


def parse_json(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)


def construct_zero2d_config_dict(smp_config, smp_core):
    if not smp_config.zero2d_enabled():
        return {}

    default_config = construct_default_zero2d_config_dict(
        smp_config.sharded_data_parallel_degree, smp_config, smp_core
    )

    # update the defaults with the config params with sdp_ prefix
    updated_config = update_zero2d_config_dict_with_sdp_params(default_config, smp_config)

    # parse zero2d config dict if specified
    if smp_config._sharded_data_parallelism_config is not None:
        custom_config = parse_json(smp_config._sharded_data_parallelism_config)
        zero_config = recursive_update(default_config, custom_config)
    else:
        zero_config = default_config

    validate_zero2d_config(zero_config, smp_config, smp_core)
    return zero_config


def update_zero2d_config_dict_with_sdp_params(default_config, smp_config):
    updated_config = copy.deepcopy(default_config)
    updated_config["zero_optimization"]["reduce_bucket_size"] = smp_config.sdp_reduce_bucket_size
    updated_config["zero_optimization"][
        "stage3_param_persistence_threshold"
    ] = smp_config.sdp_param_persistence_threshold
    updated_config["zero_optimization"][
        "stage3_max_live_parameters"
    ] = smp_config.sdp_max_live_parameters
    updated_config["zero_optimization"][
        "zero2d_hierarchy_allgather"
    ] = smp_config.sdp_hierarchical_allgather
    updated_config["gradient_clipping"] = smp_config.sdp_gradient_clipping

    return updated_config


def construct_default_zero2d_config_dict(sharding_degree, smp_config, smp_core):
    dirname = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirname, "ds_config_defaults.json")
    default_config = parse_json(path)

    default_config["zero_optimization"]["zero2d_shard_size"] = sharding_degree

    default_config["zero_optimization"]["zero2d_hierarchy_allgather"] = (
        sharding_degree > smp_core.lib.smp_device_count()
    )

    # handle batch size - the batch sizes are not really used but they still need to be consistent with dp degree, to keep DS config validation happy
    default_config["train_micro_batch_size_per_gpu"] = 1
    default_config["train_batch_size"] = smp_core.dp_size()

    # if smp fp16 is enabled, auto-enable fp16 in zero config
    # loss_scale will get overriden by smp.DistributedOptimizer later
    if smp_config.fp16:
        default_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 20,
            "loss_scale_window": 1000,
        }

    if smp_config.bf16:
        default_config["bf16"] = {"enabled": True}

    return default_config


def recursive_update(old_dict, new_dict):
    for k, v in new_dict.items():
        if isinstance(v, collections.abc.Mapping):
            old_dict[k] = recursive_update(old_dict.get(k, {}), v)
        else:
            old_dict[k] = v
    return old_dict


def validate_zero2d_config(input_config, smp_config, smp_core):
    from smdistributed.modelparallel.backend.exceptions import SMPInvalidArgumentError

    try:
        import deepspeed  # noqa isort:skip
    except ImportError:
        raise SMPInvalidArgumentError(
            "Sharded data parallelism is enabled, but DeepSpeed is not found in environment!"
        )

    shard_size = input_config["zero_optimization"]["zero2d_shard_size"]

    if input_config["zero_optimization"]["contiguous_gradients"]:
        raise SMPInvalidArgumentError(
            "contiguous_gradients must be set to false in _sharded_data_parallelism_config."
        )

    if input_config["zero_optimization"]["cpu_offload"]:
        raise SMPInvalidArgumentError(
            "cpu_offload must be set to false in _sharded_data_parallelism_config."
        )

    if input_config["zero_optimization"]["stage"] != 3:
        raise SMPInvalidArgumentError("Only stage 3 is supported in sharded data parallelism.")

    # validate sharding degree
    if shard_size > smp_core.dp_size():
        raise SMPInvalidArgumentError(
            f"ZeRO-2D sharding degree ({shard_size}) cannot be larger than the data parallelism degree ({smp_core.dp_size()})."
        )

    # if smp fp16 is disabled but zero fp16 is enabled, raise error
    if not smp_config.fp16 and input_config.get("fp16", {}).get("enabled", False):
        raise SMPInvalidArgumentError(
            "If fp16 is enabled in _sharded_data_parallelism_config, it must also be enabled in smp config."
        )

    # ZeRO-2D complains if hierarchical allgather is enabled when sharding within node
    if shard_size <= smp_core.lib.smp_device_count():
        input_config["zero_optimization"]["zero2d_hierarchy_allgather"] = False
