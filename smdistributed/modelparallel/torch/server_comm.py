# Standard Library
import collections
import copy
import pickle
import time
from contextlib import contextmanager
from typing import List, Optional, Union

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.collectives import TransactionIdentifier
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.core import core, local_rank, pp_rank, rank
from smdistributed.modelparallel.torch.messages import (
    ForwardExecutionResult,
    MicrobatchEndResult,
    StubbedMessage,
)
from smdistributed.modelparallel.torch.ops import send, synchronize
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import rmsg

logger = get_logger()


class ServerCommunicator:
    """
    A communicator for Module servers. Removes the need for explicit recv calls, and replaces
    it with a has_message()/get_next_message() API that automatically fetches the objects sent
    to the current rank. If the object has torch.Tensors as members, they will be removed and
    sent individually using the torch.Tensor communication API of the backend, whereas the
    Python object itself will use the collectives API. Upon reception, the original object will
    be reconstructed by inserting the torch.Tensors in appropriate locations, and returned to the caller.
    """

    def __init__(self):
        self.messages_awaiting_tensors = {}
        self.ready_messages = collections.deque()
        self._local_tensor_cache = {}
        self._stubs_for_mb: Dict[int, "TensorStub"] = collections.defaultdict(list)

        # used to cache link ids in static mode
        # mapping: step_fucntion->msg_meta->tensor_index_to_link_ids
        # Note: the msg_meta could be same for different messages, using a queue here to enforce sequencing
        self.msg_meta_to_link_id = collections.defaultdict(lambda: collections.defaultdict(list))
        self.record_msg_meta_to_link_id = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list))
        )
        # msg_meta_to_link_id for current minibatch
        self.minibatch_msg_meta_to_link_id = None
        # used to record link_id for each message
        self.tensor_index_to_link_id = {}

    def clear_minibatch_state(self):
        self._local_tensor_cache.clear()
        smplib.smp_torch_clear_tensor_receptions()
        self._stubs_for_mb.clear()
        self.minibatch_msg_meta_to_link_id = None

    def clear_microbatch_state(self, mb):
        # there's no reason to keep that tensor around. this helps
        # ensure that we don't accumulate any memory throughout the step.
        for stub in self._stubs_for_mb[mb]:
            if not stub.is_dummy:
                smplib.smp_torch_clear_tensor_reception(stub.src, stub.link_id)
        del self._stubs_for_mb[mb]

    def send(self, msg, peers: Optional[Union[int, List[int]]] = None, group=True):
        """
        Send the object `msg` to all the ranks in `peers`. If `msg` contains
        torch.Tensors, they will be extracted and sent individually, and the rest
        of the object will be serialized and sent through the collectives API.

        `peers` is rank within pp_group if group is True.
        `peers` need to be global ranks if group is False.

        If `peers` is None and group is True, sends to others in pp_group, if group is False
        sends to all other ranks.
        """
        # send internal only takes global ranks, so this function needs to convert to global rank
        if peers is None and group is True:
            peers = list(range(core.pp_size()))
        elif peers is None and group is False:
            peers = list(range(core.size()))
        elif isinstance(peers, int):
            peers = [peers]

        if group is True:
            peers = [core.pp_rank_to_rank(peer) for peer in peers]

        # by now peers has all global ranks

        try:
            # dont send to self msg.stubify() might re-introduce the current rank
            # in case of fast mode
            peers.remove(rank())
        except:
            pass

        if isinstance(msg, StubbedMessage):
            with self._maybe_record_and_update_link_ids(msg):
                msg, tx_list = msg.stubify(peers)
        else:
            tx_list = []

        if len(peers) > 0:
            self._send_internal(msg, tx_list, peers)

    @contextmanager
    def _maybe_record_and_update_link_ids(self, message):
        """For static mode:
           - On record step, record the link ids for each tensor/message
           - After record step, read the link ids for each message
           The link id will be deterministic for each tensor for every step
        """
        if state.skip_metadata_transmission():
            msg_meta = self._get_stubmsg_meta(message)
            assert (
                msg_meta in self.minibatch_msg_meta_to_link_id
                and len(self.minibatch_msg_meta_to_link_id[msg_meta]) > 0
            ), rmsg(f"message {message} is not recorded properly!")
            tensor_index_to_link_id = self.minibatch_msg_meta_to_link_id[msg_meta].pop()
            state.serialization_manager.tensor_index_to_link_id = tensor_index_to_link_id
            logger.debug(
                rmsg(
                    f"Static mode: sending message {message} with tensor_index_to_link_id {tensor_index_to_link_id}"
                )
            )
        try:
            yield
        finally:
            state.serialization_manager.tensor_index_to_link_id = None
            if state.should_record_metadata():
                # During record step, for each message, SerializationManager will
                # record the link id for each tensor in self.tensor_index_to_link_id
                # We need to reset self.tensor_index_to_link_id for each message here
                msg_meta = self._get_stubmsg_meta(message)
                logger.debug(
                    rmsg(
                        f"Static mode: recording message {message} with tensor_index_to_link_id {self.tensor_index_to_link_id}"
                    )
                )
                cur_step = state.exec_server.server_queue.step[state.current_step_func()]
                self.record_msg_meta_to_link_id[state.current_step_func()][cur_step][
                    msg_meta
                ].insert(0, self.tensor_index_to_link_id)
                self.tensor_index_to_link_id = {}

    def _get_stubmsg_meta(self, message):
        message_meta = copy.copy(message)
        if isinstance(message, (MicrobatchEndResult, ForwardExecutionResult)):
            message_meta.strip_outputs()
        else:
            message_meta.strip_tensors()
        return message_meta

    def has_message(self) -> bool:
        """ Return whether there is a message arrived for this rank. Returns True only when both
        the stubbed message and the associated tensors are ready. """
        # check if there are messages arrived
        if smplib.smp_torch_has_next_message():
            msg = self._get_message_internal()
            if isinstance(msg, StubbedMessage):
                stubs = state.serialization_manager.extract_stubs(msg)
            else:
                stubs = []
            self.messages_awaiting_tensors[id(msg)] = (msg, stubs)

        # check if all tensors have arrived for the messages awaiting tensors
        _ready = []
        for _, (msg, stubs) in self.messages_awaiting_tensors.items():
            if self._all_tensors_ready(stubs):
                _ready.append(id(msg))
        for k in _ready:
            self.ready_messages.append(self.messages_awaiting_tensors[k])
            del self.messages_awaiting_tensors[k]
        return len(self.ready_messages) > 0

    def get_next_message(self, block=False, destubify=True):
        """
        Reconstruct and return the next message arrived at this rank.
        Arguments:
            block     : If True, will block execution until there is a message available.
            destubify : If True, the arrived message will be reconstructed by inserting the received torch.Tensors.
        """
        if block:
            while not self.has_message():
                time.sleep(0.00005)
        elif not self.has_message():
            return None

        # at this point self.has_message() must have returned True, so self.ready_messages cannot be empty
        assert len(self.ready_messages) > 0, "has_message returned True, but no ready messages"

        stubbed_msg, stubs = self.ready_messages.popleft()
        if hasattr(stubbed_msg, "mb"):
            self._stubs_for_mb[stubbed_msg.mb].extend(stubs)
        tensors = [self._reconstruct_tensor(stub) for stub in stubs]

        if destubify:
            if isinstance(stubbed_msg, StubbedMessage):
                with state.serialization_manager.catch_and_raise_for_large_object(stubbed_msg):
                    stubbed_msg_copy = copy.deepcopy(stubbed_msg)
                destub_msg = stubbed_msg.destubify(stubbed_msg, tensors)
                if state.should_record_metadata():
                    state.exec_server.server_queue.set_task_artifact(stubbed_msg_copy, stubs)
                return destub_msg
            else:
                return stubbed_msg
        else:
            return stubbed_msg, tensors

    def reconstruct_destubmessage(self, stubbed_msg, stubs):
        """Replace the stubs in a message with tensors, used in static mode when the message and stubs are local.
           Should be aligned with has_message() and get_next_message()"""
        logger.debug(
            rmsg(f"Static mode: stubbed_msg {stubbed_msg} is waiting tensors for stubs {stubs}")
        )
        while not self._all_tensors_ready(stubs):
            time.sleep(0.00005)
        if hasattr(stubbed_msg, "mb"):
            self._stubs_for_mb[stubbed_msg.mb].extend(stubs)
        tensors = [self._reconstruct_tensor(stub) for stub in stubs]
        logger.debug(rmsg(f"Static mode: stubbed_msg {stubbed_msg} have all tensors ready"))
        return stubbed_msg.destubify(stubbed_msg, tensors)

    def _reconstruct_tensor(self, stub):
        if stub.is_dummy:
            # dummy tensor at the parent. this means that the true tensor
            # was sent directly into the next child. we will raise an
            # error if this dummy tensor is used at the parent. torch.empty
            # fills with uninitialized data.

            # we need to preserve the original shape for some edge cases where in backward pass, true
            # gradients are still sent to the parent for aggregation.
            # when this happens the forward activation is not really used, but the shapes
            # still need to be compatible to keep the autograd engine happy.
            tensor = torch.empty(
                *stub.shape,
                dtype=stub.dtype,
                requires_grad=stub.requires_grad,
                device=torch.device(stub.device, local_rank()),
            )
            tensor._smp_is_dummy = True
            state.serialization_manager.dummy_registry.put(tensor, stub)
        else:
            tensor = self._get_tensor(stub)
            tensor.requires_grad = stub.requires_grad

        if stub.module_info is not None:
            tensor._smp_module_info = stub.module_info
        return tensor

    def _send_internal(self, stubbed_msg, tx_list, peers):
        """
        Treats peers as global ranks in this function
        """

        # start with tensor transmissions since they will take longer
        handles = []
        for transmission in tx_list:
            for dest in transmission.dests:
                if dest == rank():
                    self.cache_tensor(transmission.link_id, transmission.tensor)
                else:
                    handles.append(
                        send(
                            transmission.tensor,
                            dest,
                            transmission.link_id,
                            needs_meta_transmission=(not state.skip_metadata_transmission()),
                        )
                    )
                    logger.debug(
                        rmsg(
                            f"Sending tensor with size {transmission.tensor.size()} to {dest} on link {transmission.link_id}"
                        )
                    )

        if not state.skip_metadata_transmission():
            # send the stubbed message
            for peer in peers:
                coll_message, coll_msg_len = state.comm.pickle_obj(stubbed_msg)
                coll_link = state.link_manager.get_link(coll_msg_len)
                state.comm.async_send_global(
                    coll_message,
                    peer,
                    TransactionIdentifier(coll_link, False),
                    do_pickle=False,
                    server=True,
                )
                logger.debug(rmsg(f"Sending obj to {peer} on link {coll_link}"))

            state.comm.wait_all_transactions(server=True)
        for handle in handles:
            synchronize(handle)

    def _get_message_internal(self):
        buf = smplib.smp_torch_get_next_message()
        smplib.smp_torch_clean_next_message()
        return pickle.loads(buf)

    def cache_tensor(self, link_id, tensor):
        """ Cache tensor if the next consumer is in the same rank. """

        assert hasattr(
            tensor, "_smp_module_info"
        ), "Attempt to locally cache tensor without module info attribute"

        module_info = tensor._smp_module_info
        self._local_tensor_cache[link_id] = tensor

        logger.debug(rmsg(f"Locally cached tensor at {pp_rank()} on link {link_id}"))

    def fetch_tensor(self, link_id):
        """ Get tensor that is locally cached. """
        assert link_id in self._local_tensor_cache, "Cannot find tensor in local cache."

        tensor = self._local_tensor_cache[link_id].detach()

        logger.debug(rmsg(f"Fetching local tensor at link {link_id}"))
        return tensor

    def _all_tensors_ready(self, stubs):
        for stub in stubs:
            if not stub.is_dummy and not self._check_tensor(stub):
                return False
        return True

    def _check_tensor(self, stub):
        if stub.src == rank():
            assert stub.link_id in self._local_tensor_cache, "Cannot find tensor in local cache."
            return True

        return smplib.smp_torch_check_tensor(stub.src, stub.link_id)

    def _get_tensor(self, stub):
        if stub.src == rank():
            return self.fetch_tensor(stub.link_id)

        t = smplib.smp_torch_get_tensor(stub.src, stub.link_id)

        logger.debug(rmsg(f"Picking up tensor with (src, link_id)=({stub.src}, {stub.link_id})"))
        assert t is not None, "Retrieved tensor is None."
        return t
