# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.collectives import (  # noqa: skip
    CollectiveCommunicator,
    CommGroup,
    RankType,
    TransactionIdentifier,
)
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.core import core, rank
from smdistributed.modelparallel.torch.ops import recv as recv_tensor
from smdistributed.modelparallel.torch.ops import send as send_tensor
from smdistributed.modelparallel.torch.ops import synchronize, wait_and_clear
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import get_tensor_size


class PTCollectiveCommunicator(CollectiveCommunicator):
    """
    PT's collective communicator. The python object and the tensors will be sent separately.
    The python object will be deserialized and the tensors will be extracted. Then the python object(without tensor)
    will be send in using the backend's collective API and the tensors will be sent using torch's send/recv functions
    defined in ops.py
    """

    def async_send_batch_global(
        self, tensor_transmissions, stubbed_obj, global_dest_rank, transaction_id
    ):
        """
        Send a list of tensor and a python object async
        """
        handles = []
        self.async_send_global(stubbed_obj, global_dest_rank, transaction_id)
        for transmission in tensor_transmissions:
            # Set collective to True so that listener won't receive this message
            handles.append(
                send_tensor(
                    transmission.tensor,
                    global_dest_rank,
                    transmission.link_id,
                    server=False,
                    release_after_send=False,
                )
            )
        return handles

    def recv_batch_global(self, global_src_rank, transaction_id):
        """
        Receive a list of tensor and a python object async
        """
        stubbed_obj = self.recv_from_global(global_src_rank, transaction_id)

        handles = []
        output_tensors = []
        stubs = state.serialization_manager.extract_stubs(stubbed_obj)

        for stub in stubs:
            tensor = torch.empty((0,), device=torch.device("cuda", core.local_rank()))
            handles.append(recv_tensor(tensor, stub.src, stub.link_id, server=False))

        for handle in handles:
            output_tensors.append(synchronize(handle))
        return stubbed_obj, output_tensors

    def send_large(self, obj, dest_rank, rank_type):
        """
        Send a large python object with tensors sync
        """
        # Get tensors and stubbed_object
        dest_rank, _ = self._get_global_ranks(rank_type.get_comm_group(), dest_rank)
        stubbed_obj, tx_list = state.serialization_manager.serialize(obj, False, [dest_rank])

        # Get link ids
        transaction_id = TransactionIdentifier(core.rank(), True)
        handles = self.async_send_batch_global(tx_list, stubbed_obj, dest_rank, transaction_id)

        # synchronize
        for handle in handles:
            wait_and_clear(handle)
        self.wait_transaction(transaction_id)

    def recv_from_large(self, src_rank, rank_type):
        """
        Receive a large python object with tensors sync
        """
        src_rank, _ = self._get_global_ranks(rank_type.get_comm_group(), src_rank)
        transaction_id = TransactionIdentifier(src_rank, True)
        stubbed_obj, tensor_list = self.recv_batch_global(src_rank, transaction_id)

        return state.serialization_manager.deserialize(stubbed_obj, tensor_list)

    def async_recv_from_large(self, src_rank, rank_type):
        """
        Receive a large python object with tensors async
        """
        src_rank, _ = self._get_global_ranks(rank_type.get_comm_group(), src_rank)
        transaction_id = TransactionIdentifier(src_rank, True)

        stubbed_obj = self.recv_from_global(src_rank, transaction_id)

        handles = []
        output_tensors = []
        stubs = state.serialization_manager.extract_stubs(stubbed_obj)

        for stub in stubs:
            tensor = torch.empty((0,), device=torch.device("cuda", core.local_rank()))
            handles.append(recv_tensor(tensor, stub.src, stub.link_id, server=False))
        return handles, stubbed_obj

    def wait_recv_large(self, handles, stubbed_obj):
        """
        Wait the recv call to finish and get result
        """
        tensor_list = []
        for handle in handles:
            tensor_list.append(synchronize(handle))
        return state.serialization_manager.deserialize(stubbed_obj, tensor_list)

    def async_bcast_large_global(self, stubbed_obj, tx_list, dest_ranks, transaction_id):
        """
        Broadcast a list of tensor and a python object async
        """
        handles = []
        for dest_rank in dest_ranks:
            if dest_rank != core.rank():
                handles.extend(
                    self.async_send_batch_global(tx_list, stubbed_obj, dest_rank, transaction_id)
                )
        return handles

    def broadcast_large(self, obj, group):
        """
        Broadcast a large python object with tensors sync
        """
        _, dest_ranks = self._get_global_ranks(group, None)
        other_ranks = [r for r in dest_ranks if r != rank()]
        if len(other_ranks) == 0:
            return

        stubbed_obj, tx_list = state.serialization_manager.serialize(obj, False, other_ranks)

        # Get link ids
        transaction_id = TransactionIdentifier(core.rank(), True)

        handles = self.async_bcast_large_global(stubbed_obj, tx_list, dest_ranks, transaction_id)

        # synchronize
        for handle in handles:
            wait_and_clear(handle)
        self.wait_transaction(transaction_id)

    def allgather_large(self, obj, group):
        """
        Allgather a large python object with tensors sync
        """
        self_rank, dest_ranks = self._get_global_ranks(group, None)
        other_ranks = [r for r in dest_ranks if r != self_rank]
        if len(other_ranks) == 0:
            return [obj]

        stubbed_obj, tx_list = state.serialization_manager.serialize(obj, False, other_ranks)

        # Get link ids
        transaction_id = TransactionIdentifier(core.rank(), True)

        handles = self.async_bcast_large_global(stubbed_obj, tx_list, dest_ranks, transaction_id)
        allgather_obj = []
        for peer in dest_ranks:
            if peer != self_rank:
                allgather_obj.append(self.recv_from_large(peer, RankType.WORLD_RANK))
            else:
                allgather_obj.append(obj)

        # synchronize
        for handle in handles:
            wait_and_clear(handle)
        self.wait_transaction(transaction_id)

        return allgather_obj

    def gather_large(self, obj, group, rank=0):
        """
        Gather a large python object with tensors sync
        """
        self_rank, dest_ranks = self._get_global_ranks(group, None)
        rank, _ = self._get_global_ranks(group, rank)
        other_ranks = [r for r in dest_ranks if r != self_rank]
        if len(other_ranks) == 0:
            return [obj]

        gather_obj = []

        if self_rank == rank:
            stubobj_handles = {}
            # async recv calls
            for peer in dest_ranks:
                if peer != self_rank:
                    stubobj_handles[peer] = self.async_recv_from_large(peer, RankType.WORLD_RANK)
            # syncronize and get result
            for peer in dest_ranks:
                if peer != self_rank:
                    obj = self.wait_recv_large(*stubobj_handles[peer])
                    gather_obj.append(obj)
                else:
                    gather_obj.append(obj)
        else:
            # sync send for other ranks
            self.send_large(obj, rank, RankType.WORLD_RANK)

        return gather_obj

    def allgatherv_tensor(self, tensor, counts, group=CommGroup.DP_GROUP):
        _, dest_ranks = self._get_global_ranks(group, None)
        smplib.smp_torch_nccl_allgatherv(dest_ranks, tensor, counts, group.value)

    def scatter_and_merge_tensor(
        self, tensor, split_axis, merge_axis, group=CommGroup.TP_GROUP, merge_shapes=None
    ):
        """ Slice the input tensor along split_axis, apply all-to-all collective to slices, and
        concatenate the received slices across merge_axis. The resulting tensor at rank i is the
        concatenation of the i'th slices of every other rank along the merge_axis. """

        if merge_shapes == None:
            merge_shapes = []

        group_size = self._get_group_size(group)
        if group_size == 1:
            return tensor

        # support negative indexing
        if merge_axis < 0 and merge_axis >= -tensor.dim():
            merge_axis = merge_axis + tensor.dim()
        if split_axis < 0 and split_axis >= -tensor.dim():
            split_axis = split_axis + tensor.dim()

        if tensor.device.type == "cuda":
            _, dest_ranks = self._get_global_ranks(group, None)
            return smplib.smp_torch_nccl_scatter_and_merge(
                dest_ranks, tensor, split_axis, merge_axis, group.value, merge_shapes
            )
        else:
            # TODO: Add merge_shapes to support unbalanced partition
            return self._smp_scatter_and_merge_tensor(tensor, split_axis, merge_axis, group)

    def _smp_scatter_and_merge_tensor(self, tensor, split_axis, merge_axis, group):
        group_size = self._get_group_size(group)

        def _get_src_dest_id(src, dest):
            return src * group_size + dest

        # split tensor: the split dimension may not be divisible by the group size
        div = tensor.shape[split_axis] // group_size
        rem = tensor.shape[split_axis] % group_size
        split_sizes = [(div + 1 if r < rem else div) for r in range(group_size)]

        tensor_slices = torch.split(tensor, split_sizes, split_axis)

        self_rank, dest_ranks = self._get_global_ranks(group, None)
        other_ranks = [r for r in dest_ranks if r != self_rank]

        # send the link_ids for the tensor transmissions
        link_ids = {}
        for tensor_slice, dest_rank in zip(tensor_slices, dest_ranks):
            if dest_rank != self_rank:
                link_id = state.link_manager.get_link(get_tensor_size(tensor_slice))
                link_ids[dest_rank] = link_id
                transaction_id = TransactionIdentifier(
                    _get_src_dest_id(self_rank, dest_rank), False
                )
                self.async_send_global(link_id, dest_rank, transaction_id)

        # send the slices
        send_handles = []
        for tensor_slice, dest_rank in zip(tensor_slices, dest_ranks):
            if dest_rank != self_rank:
                link_id = link_ids[dest_rank]
                send_handles.append(
                    send_tensor(
                        tensor_slice, dest_rank, link_id, server=False, release_after_send=False
                    )
                )

        # receive slices
        recv_handles = {}
        recvd_tensors = [None for _ in range(group_size)]
        for global_rank in dest_ranks:
            if global_rank != self_rank:
                tensor = torch.empty((0,), device=torch.device("cuda"))
                transaction_id = TransactionIdentifier(
                    _get_src_dest_id(global_rank, self_rank), False
                )
                link_id = self.recv_from_global(global_rank, transaction_id)
                recv_handles[global_rank] = recv_tensor(tensor, global_rank, link_id, server=False)

        recvd_tensors = []
        for i, dest_rank in enumerate(dest_ranks):
            if dest_rank == self_rank:
                recvd_tensors.append(tensor_slices[i])
            else:
                recvd_tensors.append(synchronize(recv_handles[dest_rank]))

        # concatenate
        output_tensor = torch.cat(recvd_tensors, merge_axis)

        for dest_rank in dest_ranks:
            if self_rank != dest_rank:
                transaction_id = TransactionIdentifier(
                    _get_src_dest_id(self_rank, dest_rank), False
                )
                self.wait_transaction(transaction_id)

        # synchronize
        for handle in send_handles:
            wait_and_clear(handle)

        return output_tensor
