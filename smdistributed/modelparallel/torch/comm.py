# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup, RankType  # noqa: skip
from smdistributed.modelparallel.torch.core import core
from smdistributed.modelparallel.torch.state_mod import state

MP_GROUP = CommGroup.MP_GROUP
PP_GROUP = CommGroup.PP_GROUP
TP_GROUP = CommGroup.TP_GROUP
RDP_GROUP = CommGroup.RDP_GROUP
DP_GROUP = CommGroup.DP_GROUP
WORLD = CommGroup.WORLD

# Ranktypes haven't been made part of the namespace directly so as to not mislead users into thinking enum value is the actual rank
# now they have to use it as smp.RankType.WORLD_RANK which makes it clear that this is not rank but rankType

# public comm APIs
# they are redefined here as during import state.comm isn't initialized


def broadcast(obj, group):
    return state.comm.broadcast_large(obj, group)


def send(obj, dest_rank, rank_type):
    return state.comm.send_large(obj, dest_rank, rank_type)


def recv_from(src_rank, rank_type):
    return state.comm.recv_from_large(src_rank, rank_type)


def allgather(obj, group):
    return state.comm.allgather_large(obj, group)

def gather(obj, group, rank=0):
    return state.comm.gather_large(obj, group, rank=rank)


def dp_barrier():
    return state.comm.dp_barrier()


def rdp_barrier():
    return state.comm.rdp_barrier()


def tp_barrier():
    return state.comm.tp_barrier()


def pp_barrier():
    return state.comm.pp_barrier()


def mp_barrier():
    return state.comm.mp_barrier()


def barrier(group=CommGroup.WORLD):
    if group == CommGroup.WORLD:
        core.barrier()
    elif group == CommGroup.PP_GROUP:
        pp_barrier()
    elif group == CommGroup.MP_GROUP:
        mp_barrier()
    elif group == CommGroup.DP_GROUP:
        dp_barrier()
    elif group == CommGroup.TP_GROUP:
        tp_barrier()
    elif group == CommGroup.RDP_GROUP:
        rdp_barrier()

    else:
        raise ValueError(
            "Invalid group passed, it needs to be one of CommGroup.WORLD, CommGroup.PP_GROUP, CommGroup.DP_GROUP"
        )


def get_dp_process_group():
    if state.cfg.ddp:
        return state.dp_process_group
    else:
        raise RuntimeError(
            "Torch.dist is not initialized. Please enable DDP in config to use torch.dist"
        )


def get_pp_process_group():
    if state.cfg.ddp:
        return state.pp_process_group
    else:
        raise RuntimeError(
            "Torch.dist is not initialized. Please enable DDP in config to use torch.dist"
        )


def get_tp_process_group():
    if state.cfg.ddp:
        return state.tp_process_group
    else:
        raise RuntimeError(
            "Torch.dist is not initialized. Please enable DDP in config to use torch.dist"
        )


def get_rdp_process_group():
    if state.cfg.ddp:
        return state.rdp_process_group
    else:
        raise RuntimeError(
            "Torch.dist is not initialized. Please enable DDP in config to use torch.dist"
        )


def get_mp_process_group():
    return state.mp_process_group


def get_world_process_group():
    if state.cfg.ddp:
        return state.world_process_group
    else:
        raise RuntimeError(
            "Torch.dist is not initialized. Please enable DDP in config to use torch.dist"
        )
