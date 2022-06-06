# Standard Library
import copy
import ctypes
import pickle
from enum import Enum
from typing import NamedTuple


class CommGroup(Enum):
    WORLD = 0
    PP_GROUP = 1
    TP_GROUP = 2
    RDP_GROUP = 3
    DP_GROUP = 4
    MP_GROUP = 5

    def get_rank_type(self):
        if self == CommGroup.WORLD:
            return RankType.WORLD_RANK
        elif self == CommGroup.PP_GROUP:
            return RankType.PP_RANK
        elif self == CommGroup.TP_GROUP:
            return RankType.TP_RANK
        elif self == CommGroup.RDP_GROUP:
            return RankType.RDP_RANK
        elif self == CommGroup.DP_GROUP:
            return RankType.DP_RANK
        elif self == CommGroup.MP_GROUP:
            return RankType.MP_RANK


class RankType(Enum):
    WORLD_RANK = 0
    PP_RANK = 1
    TP_RANK = 2
    RDP_RANK = 3
    DP_RANK = 4
    MP_RANK = 5

    def get_comm_group(self):
        if self == RankType.WORLD_RANK:
            return CommGroup.WORLD
        elif self == RankType.PP_RANK:
            return CommGroup.PP_GROUP
        elif self == RankType.TP_RANK:
            return CommGroup.TP_GROUP
        elif self == RankType.RDP_RANK:
            return CommGroup.RDP_GROUP
        elif self == RankType.DP_RANK:
            return CommGroup.DP_GROUP
        elif self == RankType.MP_RANK:
            return CommGroup.MP_GROUP


class TransactionIdentifier(NamedTuple):
    id: int
    is_user_api: bool

    def get(self):
        return 2 * self.id + int(self.is_user_api)


class CollectiveCommunicator:
    """ A communicator object which exposes a intra-pp_group, intra-dp_group, and intra-world communication primitives for Python objects."""

    def __init__(self, core):
        self.core = core
        self.lib = self.core.lib

        # these are global ranks of ppgroup and dpgroup
        self.pp_group = self.core.get_pp_group()
        self.tp_group = self.core.get_tp_group()
        self.rdp_group = self.core.get_rdp_group()
        self.dp_group = self.core.get_dp_group()
        self.mp_group = self.core.get_mp_group()
        self.world = list(range(self.core.size()))

        self.tracked_transactions = set()

    """
    User visible functions will use default transaction id of 0, and be synchronous for now.

    They should not have transaction_id and server arguments.
    """

    def _get_global_ranks(self, comm_group, given_rank=None):
        """
        Returns global rank of sender and the group with which to communicate
        """
        if not isinstance(comm_group, CommGroup):
            raise ValueError(
                "`group` is a required argument and has to be one of WORLD, PP_GROUP or DP_GROUP"
            )

        if comm_group == CommGroup.WORLD:
            if given_rank is None:
                given_rank = self.core.rank()
            rank = given_rank
            group = self.world
        elif comm_group == CommGroup.PP_GROUP:
            if given_rank is None:
                given_rank = self.core.pp_rank()
            rank = self.core.pp_rank_to_rank(given_rank)
            group = self.pp_group
        elif comm_group == CommGroup.TP_GROUP:
            if given_rank is None:
                given_rank = self.core.tp_rank()
            rank = self.core.tp_rank_to_rank(given_rank)
            group = self.tp_group
        elif comm_group == CommGroup.RDP_GROUP:
            if given_rank is None:
                given_rank = self.core.rdp_rank()
            rank = self.core.rdp_rank_to_rank(given_rank)
            group = self.rdp_group
        elif comm_group == CommGroup.DP_GROUP:
            if given_rank is None:
                given_rank = self.core.dp_rank()
            rank = self.core.dp_rank_to_rank(given_rank)
            group = self.dp_group
        elif comm_group == CommGroup.MP_GROUP:
            if given_rank is None:
                given_rank = self.core.mp_rank()
            rank = self.core.mp_rank_to_rank(given_rank)
            group = self.mp_group

        return rank, group

    def _get_group_size(self, group):
        if group == CommGroup.WORLD:
            return self.core.size()
        elif group == CommGroup.PP_GROUP:
            return self.core.pp_size()
        elif group == CommGroup.DP_GROUP:
            return self.core.dp_size()
        elif group == CommGroup.MP_GROUP:
            return self.core.mp_size()
        elif group == CommGroup.TP_GROUP:
            return self.core.tp_size()
        elif group == CommGroup.RDP_GROUP:
            return self.core.rdp_size()
        else:
            raise ValueError(f"Unsupported CommGroup type {group}.")

    def broadcast(self, obj, group):
        """
        If group is CommGroup.WORLD, expects src_rank to be rank of src.
        If group is CommGroup.PP_GROUP, expects src_rank to be pp_rank of src.
        If group is CommGroup.MP_GROUP, expects src_rank to be mp_rank of src.
        If group is CommGroup.DP_GROUP, expects src_rank to be dp_rank of src.
        If group is CommGroup.TP_GROUP, expects src_rank to be tp_rank of src.
        If group is CommGroup.RDP_GROUP, expects src_rank to be rdp_rank of src.

        Given object will be picked before sending and depickled when receiving before returning.
        """
        rank, group = self._get_global_ranks(group, None)
        group_size = len(group)
        transaction_id = TransactionIdentifier(rank, True)
        self.async_bcast_global(obj, group, transaction_id=transaction_id)
        self.wait_transaction(transaction_id)
        return obj

    def send(self, obj, dest_rank, rank_type):
        """
        Given object will be picked before sending and depickled when receiving before returning.
        """
        rank, _ = self._get_global_ranks(rank_type.get_comm_group(), dest_rank)
        transaction_id = TransactionIdentifier(self.core.rank(), True)
        server = False

        self.async_send_global(obj, rank, transaction_id, do_pickle=True, server=False)
        self.wait_transaction(transaction_id)

    def recv_from(self, src_rank, rank_type):
        rank, _ = self._get_global_ranks(rank_type.get_comm_group(), src_rank)
        server = False
        transaction_id = TransactionIdentifier(rank, True)
        return self.recv_from_global(
            rank, transaction_id=transaction_id, depickle=True, server=server
        )

    def allgather(self, obj, group):
        curr_rank, peers = self._get_global_ranks(group, None)
        num_ranks = len(peers)
        if num_ranks == 1:
            return [obj]

        # Passing current rank as transaction id since transaction id needs to be unique for every transaction
        send_trans = TransactionIdentifier(curr_rank, True)
        self.async_bcast_global(obj, peers, transaction_id=send_trans)

        allgather_obj = []
        for peer in peers:
            if peer != curr_rank:
                # Passing sender's rank as transaction id to indicate unique transaction to receive from
                allgather_obj.append(self.recv_from_global(peer, TransactionIdentifier(peer, True)))
            else:
                allgather_obj.append(obj)
        self.wait_transaction(send_trans)
        return allgather_obj

    def pp_barrier(self):
        # TODO: introduce less expensive barriers in the backend
        self.allgather(1, group=CommGroup.PP_GROUP)

    def dp_barrier(self):
        # TODO: introduce less expensive barriers in the backend
        self.allgather(1, group=CommGroup.DP_GROUP)

    def tp_barrier(self):
        # TODO: introduce less expensive barriers in the backend
        self.allgather(1, group=CommGroup.TP_GROUP)

    def rdp_barrier(self):
        # TODO: introduce less expensive barriers in the backend
        self.allgather(1, group=CommGroup.RDP_GROUP)

    def mp_barrier(self):
        # TODO: introduce less expensive barriers in the backend
        self.allgather(1, group=CommGroup.MP_GROUP)

    """
    Above are Public APIs (except methods which start with `_` of course).
    Below are to be used internally by SMP only.
    `_global` suffix to method name indicates that they expect rank as global rank.
    """

    def async_bcast(self, obj, transaction_id, group, server=False):
        _, dest_ranks = self._get_global_ranks(group, None)
        self.async_bcast_global(obj, dest_ranks, transaction_id, server=server)

    def async_bcast_global(self, obj, dest_ranks, transaction_id, server=False):
        c_str, count = self._serialize_object(obj)
        num_dest_ranks = len(dest_ranks)
        self.lib.smp_async_bcast(
            c_str,
            ctypes.c_int(count),
            ctypes.c_int(num_dest_ranks),
            (ctypes.c_int * num_dest_ranks)(*dest_ranks),
            ctypes.c_int(transaction_id.get()),
            ctypes.c_bool(server),
        )
        self.tracked_transactions.add((transaction_id, server))

    def async_send(self, obj, rank, transaction_id, rank_type, do_pickle=True, server=False):
        dest_rank, _ = self._get_global_ranks(rank_type.get_comm_group(), rank)
        self.async_send_global(obj, dest_rank, transaction_id, do_pickle=do_pickle, server=server)

    def async_send_global(self, obj, rank, transaction_id, do_pickle=True, server=False):
        c_str, count = self._serialize_object(obj, do_pickle)
        self.lib.smp_async_send(
            c_str,
            ctypes.c_int(count),
            ctypes.c_int(rank),
            ctypes.c_int(transaction_id.get()),
            ctypes.c_bool(server),
        )
        self.tracked_transactions.add((transaction_id, server))

    def recv_from_global(self, rank, transaction_id, depickle=True, server=False):
        self.async_recv_from_global(rank, transaction_id, server=server)
        return self.wait_recv_from_global(rank, transaction_id, depickle=depickle, server=server)

    def async_recv_from(self, rank, transaction_id, rank_type, server=False):
        rank, _ = self._get_global_ranks(rank_type.get_comm_group(), rank)
        self.async_recv_from_global(rank, transaction_id, server=server)

    def async_recv_from_global(self, rank, transaction_id, server=False):
        self.lib.smp_async_recv(
            ctypes.c_int(rank), ctypes.c_int(transaction_id.get()), ctypes.c_bool(server)
        )
        self.tracked_transactions.add((transaction_id, server))

    def wait_recv(self, rank, transaction_id, rank_type, depickle=True, server=False):
        rank, _ = self._get_global_ranks(rank_type.get_comm_group(), rank)
        return self.wait_recv_from_global(rank, transaction_id, depickle=depickle, server=server)

    def wait_recv_from_global(self, rank, transaction_id, depickle=True, server=False):
        """ Wait for the reception with given transaction_id to complete, and return the object. """

        self._validate_transaction_id(transaction_id, server)
        count = self.lib.smp_wait_recv(
            ctypes.c_int(rank), ctypes.c_int(transaction_id.get()), ctypes.c_bool(server)
        )
        self.tracked_transactions.remove((transaction_id, server))

        self.lib.smp_retrieve_object.restype = ctypes.POINTER(ctypes.c_char * count)
        buf = self.lib.smp_retrieve_object(
            ctypes.c_int(rank), ctypes.c_int(transaction_id.get()), ctypes.c_bool(server)
        )

        python_obj = self._deserialize_object(buf, depickle)
        self.lib.smp_clean_recv_resources(
            ctypes.c_int(rank), ctypes.c_int(transaction_id.get()), ctypes.c_bool(server)
        )

        return python_obj

    def poll_recv_from_global(self, rank, transaction_id, server=False):
        self._validate_transaction_id(transaction_id, server)
        rval = self.lib.smp_poll_recv(
            ctypes.c_int(rank), ctypes.c_int(transaction_id.get()), ctypes.c_bool(server)
        )
        return rval

    def wait_all_transactions(self, server=False):
        """ Wait for all the pending send/broadcasts to complete."""
        transactions = copy.copy(self.tracked_transactions)
        for transaction in transactions:
            self.wait_transaction(*transaction)

    def wait_transaction(self, transaction_id, server=False):
        """ Wait for send/broadcast operations with the given transaction_id to complete."""
        self._validate_transaction_id(transaction_id, server)
        count = self.lib.smp_wait_transaction(
            ctypes.c_int(transaction_id.get()), ctypes.c_bool(server)
        )
        self.tracked_transactions.remove((transaction_id, server))
        return count

    def _validate_transaction_id(self, transaction_id, server):
        if (transaction_id, server) not in self.tracked_transactions:
            raise ValueError(f"Invalid transaction ID: {transaction_id}")

    def _deserialize_object(self, buf, depickle=True):
        s = buf.contents
        return pickle.loads(s) if depickle else s

    def _serialize_object(self, obj, do_pickle=True):
        if do_pickle:
            pickle_obj, count = self.pickle_obj(obj)
            return ctypes.create_string_buffer(pickle_obj, count), count
        else:
            count = len(obj)
            return ctypes.create_string_buffer(obj, count), count

    def pickle_obj(self, obj):
        """
        Used by PT ServerCommunicator as well.
        """
        pickle_obj = pickle.dumps(obj)
        count = len(pickle_obj)
        return pickle_obj, count
