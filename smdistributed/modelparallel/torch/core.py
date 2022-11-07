# Standard Library
from functools import lru_cache

# Third Party
import torch.distributed as dist

# First Party
from smdistributed.modelparallel.backend.core import ModelParallelCore

core = ModelParallelCore()

# expose the core API when smp.torch is imported
# these are moved to this file to prevent some circular imports

shutdown = core.shutdown
rank = core.rank
size = core.size
local_rank = core.local_rank
local_size = core.local_size
dp_rank = core.dp_rank
dp_size = core.dp_size
pp_rank = core.pp_rank  # device id
mp_rank = core.mp_rank
tp_rank = core.tp_rank
tp_size = core.tp_size
rdp_rank = core.rdp_rank
rdp_size = core.rdp_size
pp_size = core.pp_size
mp_size = core.mp_size
get_dp_group = core.get_dp_group
get_mp_group = core.get_mp_group
get_pp_group = core.get_pp_group
get_tp_group = core.get_tp_group
get_rdp_group = core.get_rdp_group


@lru_cache(maxsize=1)
def param_shard_rank():
    from smdistributed.modelparallel.torch.state_mod import state

    if not state.cfg.zero2d_enabled():
        return 0
    if state.optimizer == None:
        raise SMPValidationError(
            f"core.param_shard_rank can only be called after smp.DistributeOptimizer is created"
        )
    return dist.get_rank(state.optimizer.ds_param_shard_group)


@lru_cache(maxsize=1)
def param_shard_size():
    from smdistributed.modelparallel.torch.state_mod import state

    if not state.cfg.zero2d_enabled():
        return 1
    if state.optimizer == None:
        raise SMPValidationError(
            f"core.param_shard_size can only be called after smp.DistributeOptimizer is created"
        )
    return dist.get_world_size(state.optimizer.ds_param_shard_group)
