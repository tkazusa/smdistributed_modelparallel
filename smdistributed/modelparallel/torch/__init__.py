# Standard Library
import functools
import os
from typing import Dict, Union

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend import init_internal
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch import amp, nn, smplib
from smdistributed.modelparallel.torch.checkpoint import (
    load,
    resume_from_checkpoint,
    save,
    save_checkpoint,
)
from smdistributed.modelparallel.torch.comm import *
from smdistributed.modelparallel.torch.core import *
from smdistributed.modelparallel.torch.model import DistributedModel, model_creation
from smdistributed.modelparallel.torch.optimizers.optimizer import DistributedOptimizer
from smdistributed.modelparallel.torch.parameter import delay_param_initialization
from smdistributed.modelparallel.torch.patches.checkpoint import checkpoint
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.step import step

logger = get_logger()


__all__ = [
    "init",
    "step",
    "nn",
    "is_initialized",
    "is_tracing",
    "DistributedModel",
    "DistributedOptimizer",
    "partition",
    "set_partition",
    "tensor_parallelism",
    "set_tensor_parallelism",
    "num_microbatches",
    "rank",
    "size",
    "local_rank",
    "local_size",
    "dp_rank",
    "dp_size",
    "mp_rank",
    "mp_size",
    "pp_rank",
    "pp_size",
    "tp_rank",
    "tp_size",
    "rdp_rank",
    "rdp_size",
    "get_dp_process_group",
    "get_mp_process_group",
    "get_pp_process_group",
    "get_tp_process_group",
    "get_rdp_process_group",
    "get_world_process_group",
    "mp_barrier",
    "pp_barrier",
    "dp_barrier",
    "tp_barrier",
    "rdp_barrier",
    "broadcast",
    "send",
    "recv_from",
    "allgather",
    "barrier",
    "amp",
    "load",
    "save",
    "tp_register",
    "tp_register_with_module",
]


# Make sure we are inside SM and the mpi job has started already
# To avoid init for the case that there is any import in the training toolkit etc that happens before training
# Assumes that OpenMPI is used as the underlying MPI implementation
sagemaker_env = "SM_HP_MP_PARAMETERS" in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ


def init(config: Dict[str, Union[str, int, bool]] = None):
    """
    Initialize SMP.
    Inputs:
        config : a dict which specifies the user-defined configuration.
        config accepts the following keys:
            microbatches (optional) [defaults to 1]:
                An integer specifying the number of microbatches. The batch size must be divisible by this number.
            pipeline (optional) [defaults to 'simple']:
                A str specifying the pipeline type. Can be either 'simple' or 'interleaved'
            horovod (optional) [defaults to False]:
                A bool specifying whether SMP is used with Horovod. If Horovod is used, must be set to True.
            optimize (optional) [defaults to 'speed']:
                A str specifying whether SMP should optimize speed or memory. Must be one of 'speed' or 'memory'.
                Currently not functional.
            partitions: int
                Number of partitions
            auto_partition: bool
                default to True
            memory_weight: float, between 0.0 and 1.0 [default: 0.2]
                The weight of memory-use balancing in auto-partitioning objective, as opposed to balancing computation.
            tolerance: float, larger than 1.0 [default: 1.2]
                The tolerance for load imbalance. Larger values indicate higher tolerance. Too low tolerance might lead to
                increased communication.
    """
    if state.initialized and sagemaker_env:
        logger.warning(
            "SageMaker model parallelism is initialized already. Ignoring the smp.init() call."
        )
        return

    from smdistributed.modelparallel.backend.config import ModelParallelConfig

    if config is None:
        config = {}
    config = ModelParallelConfig(config)
    init_internal(
        os.path.dirname(__file__),
        config,
        core,
        state,
        true_if_pt_else_tf=True,
        start_pipeline_threads=False,
    )
    smplib.smp_torch_launch_listener()
    state.init_model_specific_state()
    torch._C._cuda_setStream(state.stream._cdata)
    state.patch_manager.save_original_methods()
    # this needs to be patched before distribute_model as we need patch during init
    state.patch_manager.patch_constructor()
    torch.cuda.set_device(local_rank())


def is_initialized():
    return state.initialized


def is_tracing() -> bool:
    return state.is_tracing()


def reset():
    state.init_model_specific_state()


num_microbatches = state.num_microbatches

set_activation_checkpointing = functools.partial(state.module_manager.set_activation_checkpointing)
partition = functools.partial(state.module_manager.partition)
set_partition = functools.partial(state.module_manager.set_partition)
tensor_parallelism = functools.partial(state.module_manager.tensor_parallelism)
set_tensor_parallelism = functools.partial(state.module_manager.set_tensor_parallelism)
tp_register = functools.partial(state.tp_registry.register)
tp_register_with_module = functools.partial(state.tp_registry.register_with_module)
delay_param_initialization = functools.partial(delay_param_initialization)
model_creation = functools.partial(model_creation)


""" If running through SageMaker, init() is hidden and automatically called with import
    by default.
    Manual call of smp.init() is still an option. To enable this option, user needs to
    explicitly set an environment variable SMP_MANUAL_INIT through Python when using
    SageMaker estimator API.
"""
if sagemaker_env:  # running through SageMaker
    manual_init = os.environ.get("SMP_MANUAL_INIT", default="0")
    manual_init = int(manual_init)
    if manual_init == 0:
        init()
