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
