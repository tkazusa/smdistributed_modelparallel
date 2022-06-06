# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup, RankType  # noqa: skip
from smdistributed.modelparallel.tensorflow.core_mod import core
from smdistributed.modelparallel.tensorflow.state_mod import state

PP_GROUP = CommGroup.PP_GROUP
MP_GROUP = CommGroup.MP_GROUP
DP_GROUP = CommGroup.DP_GROUP
TP_GROUP = CommGroup.TP_GROUP
RDP_GROUP = CommGroup.RDP_GROUP
WORLD = CommGroup.WORLD

# Ranktypes haven't been made part of the namespace directly so as to not mislead users into thinking enum value is the actual rank
# now they have to use it as smp.RankType.WORLD_RANK which makes it clear that this is not rank but rankType

# public comm APIs
# they are redefined here as during import state.comm isn't initialized


def broadcast(obj, group):
    return state.comm.broadcast(obj, group)


def send(obj, dest_rank, rank_type):
    return state.comm.send(obj, dest_rank, rank_type)


def recv_from(src_rank, rank_type):
    return state.comm.recv_from(src_rank, rank_type)


def allgather(obj, group):
    return state.comm.allgather(obj, group)


def dp_barrier():
    return state.comm.dp_barrier()


def pp_barrier():
    return state.comm.pp_barrier()


def rdp_barrier():
    return state.comm.rdp_barrier()


def tp_barrier():
    return state.comm.tp_barrier()


def mp_barrier():
    return state.comm.mp_barrier()


def barrier(group=CommGroup.WORLD):
    if group == CommGroup.WORLD:
        core.barrier()
    elif group == CommGroup.PP_GROUP:
        pp_barrier()
    elif group == CommGroup.DP_GROUP:
        dp_barrier()
    elif group == CommGroup.MP_GROUP:
        mp_barrier()
    elif group == CommGroup.TP_GROUP:
        tp_barrier()
    elif group == CommGroup.RDP_GROUP:
        rdp_barrier()
    else:
        raise ValueError("Invalid group passed, it needs to be a CommGroup type.")
