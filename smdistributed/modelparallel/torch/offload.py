# Standard Library
from collections import OrderedDict, defaultdict
from typing import Tuple

# Third Party
import torch

# First Party
from smdistributed.modelparallel.torch.core import core, local_rank, tp_rank
from smdistributed.modelparallel.torch.nn.dist_module import DistributedModule


def can_shard_activation_offloading(module):
    from smdistributed.modelparallel.torch.state_mod import state

    if not state.cfg._shard_offloaded_activations:
        return False

    if not isinstance(module, DistributedModule):
        return False

    return module.can_shard_activation_offloading()


class OffloadRegistry:
    """ Keeps track of wrapping objects and tensor attributes for the offloading inputs """

    def __init__(self):
        # task -> list(stubified objects)
        self.stubified = defaultdict(list)

        # (task, item_id) -> list(stubs)
        self.stubs = defaultdict(list)

    def put(self, task, obj):
        from smdistributed.modelparallel.torch.state_mod import state

        item_id = len(self.stubified[task])
        _id = (task, item_id)

        stubified, tx_list = state.serialization_manager.serialize(
            obj, False, [0], for_offload=True
        )
        stubs = state.serialization_manager.extract_stubs(stubified)

        self.stubified[task].append(stubified)
        self.stubs[(task, item_id)] = stubs

        return item_id, [t.tensor for t in tx_list]

    def reconstruct(self, task, item_id, tensors):
        from smdistributed.modelparallel.torch.state_mod import state

        for t, stub in zip(tensors, self.stubs[(task, item_id)]):
            t.requires_grad_(stub.requires_grad)

        return state.serialization_manager.deserialize(self.stubified[task][item_id], tensors)

    def get_tensor_specs(self, task, item_id):
        return self.stubs[(task, item_id)]

    def reset(self):
        self.stubified.clear()
        self.stubs.clear()


class TensorOffloader:
    """
    Exposes a save_for_backward / saved_tensors API, similar to torch ContextMixin, that can be used
    for saving/retrieving tensors in autograd functions. Unlike ContextMixin, it offloads the tensors
    at the CPU to save memory. load() API can be used to preemptively start loading tensors. H2D/D2H
    copies are done at separate streams, and the data transfer is fully overlapped with GPU execution.
    """

    def __init__(self):
        high_priority = -1
        self.d2h_stream = torch.cuda.Stream(device=local_rank(), priority=high_priority)
        self.h2d_stream = torch.cuda.Stream(device=local_rank(), priority=high_priority)

        # dict of OrderedDicts keyed by ServerTask. Each OrderedDict maps from item_id -> cpu tensors
        self.offloaded = defaultdict(OrderedDict)

        # (task, item_id) -> gpu tensors
        self.loaded = {}

        # dict mapping ServerTask to int. self.occupied[task] = k means that the first k elements
        # in self.offloaded[task] are occupied by actual offloaded tensors. the rest are empty buffers.
        self.occupied = defaultdict(int)

        # dict mapping ServerTask to int. self.num_offloaded[task] = k means that k elements have been
        # offloaded already for this task. May not always match self.occupied, because some tensors are
        # only offloaded by tp_rank 0. In those cases, at tp_rank != 0, num_offloaded gets incremented
        # but self.occupied does not.
        self.num_offloaded = defaultdict(int)

        # (task, item_id) -> CUDA event representing the completion of D2H copy
        self.offload_events = {}

        # (task, item_id) -> tuple(tensors) to keep reference of tensors till offload finishes
        # else the gpu memory can be reclaimed by torch memory manager for another tensor
        # causing corruption of the offload.
        self.ongoing_offload_tensors = {}

        # (task, item_id) -> CUDA event representing the completion of H2D copy
        self.load_events = {}

        # cache of currently unused CUDA events
        self.event_cache = []

        # registry to keep track of wrapping objects and tensor attributes
        self.registry = OffloadRegistry()

    def save_for_backward(self, shard_activation_offloading, *args):
        """ Save tensors at the CPU during forward pass """

        from smdistributed.modelparallel.torch.state_mod import state

        current_task = state.exec_server.current_task.task_metadata
        task_id = state.exec_server.server_queue.current_task
        task = (current_task, task_id)

        item_id, tensors = self.registry.put(task, args)
        _id = (task, item_id)

        self.offload(shard_activation_offloading, _id, tensors)

        return _id

    def saved_tensors(self, shard_activation_offloading, fwd_task_and_index, item_id):
        """ Retrieve the saved tensors during backward pass """

        from smdistributed.modelparallel.torch.state_mod import state
        from smdistributed.modelparallel.torch.comm import get_tp_process_group

        bwd_task = state.exec_server.current_task.task_metadata
        state.exec_server.server_queue.maybe_record_task_pair(fwd_task_and_index, bwd_task)

        if not shard_activation_offloading:
            tensors = self.get(fwd_task_and_index, item_id)
        else:
            if tp_rank() == 0:
                tensors = self.get(fwd_task_and_index, item_id)

                # the load happened on a separate stream and we made the default stream wait for that event to finish
                # ideally we need to do the same on stream used by nccl, but we can't access it
                # so we add a dummy operation on default stream, to make nccl wait on load stream indirectly
                tensors = tuple(t.add_(0.0) for t in tensors)
                for t in tensors:
                    torch.distributed.broadcast(t, core.tp_rank_to_rank(0), get_tp_process_group())
            else:
                tensors = []
                stubs = self.registry.get_tensor_specs(fwd_task_and_index, item_id)
                for i, stub in enumerate(stubs):
                    tensors.append(
                        torch.empty(
                            *stub.shape, dtype=stub.dtype, device=torch.device("cuda", local_rank())
                        )
                    )
                    torch.distributed.broadcast(
                        tensors[-1], core.tp_rank_to_rank(0), get_tp_process_group()
                    )
        return self.registry.reconstruct(fwd_task_and_index, item_id, tensors)

    def _record_event(self):
        """ Re-uses events to avoid event creation overhead """

        torch.cuda.set_device(local_rank())
        if len(self.event_cache) == 0:
            self.event_cache.append(torch.cuda.Event())
        event = self.event_cache.pop()

        return torch.cuda.current_stream().record_event(event)

    def _clear_tensors_for_finished_offloads(self):
        """
        Clears the tensors whose references were being kept in memory so offload happens correctly.
        They can be deleted once the cuda event marking the offload is ready.
        """
        del_keys = []
        for k, tensors in self.ongoing_offload_tensors.items():
            cuda_event = self.offload_events[k]
            if cuda_event.query():
                del_keys.append(k)

        for k in del_keys:
            self.ongoing_offload_tensors.pop(k)

    def offload(self, shard_offload, task_and_item_id, tensors: Tuple[torch.Tensor]):
        """
        Offload the given tensors to CPU, associate with the given task, and return the item_id
        to retrieve it later.
        """
        torch.cuda.set_device(local_rank())

        # we want to clear as often as possible, so called for each offload call
        self._clear_tensors_for_finished_offloads()
        # if offloading can be sharded, only tp_rank 0 offloads and loads activations. upon loading,
        # tp_rank 0 broadcasts the activations to the peer tp_ranks. otherwise, all tp_ranks offload
        # and load back simultaneously.

        task, item_id = task_and_item_id

        if tp_rank() == 0 or not shard_offload:
            if len(self.offloaded[task]) == self.occupied[task]:
                # create new cpu tensors in pinned memory
                cpu_tensors = []
                cpu_device = torch.device("cpu")
                cpu_tensors = tuple(
                    torch.empty(t.shape, dtype=t.dtype, device=cpu_device, pin_memory=True)
                    for t in tensors
                )
                self.offloaded[task][item_id] = cpu_tensors

            ready_event = self._record_event()
            with torch.cuda.stream(self.d2h_stream):
                # wait for the tensors to be ready on the PT compute stream
                torch.cuda.current_stream().wait_event(ready_event)

                for cpu_tensor, gpu_tensor in zip(self.offloaded[task][item_id], tensors):
                    cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                self.ongoing_offload_tensors[(task, item_id)] = tensors

                # TODO: investigate using separate events for separate tensors better so we can clear them earlier
                self.offload_events[(task, item_id)] = self._record_event()

            self.occupied[task] += 1

    def _load(self, task, item_id):
        # to clear as often as possible, we also call this during load
        self._clear_tensors_for_finished_offloads()

        torch.cuda.set_device(local_rank())
        torch.cuda.current_stream().wait_event(self.offload_events[(task, item_id)])
        tensors = self.offloaded[task][item_id]
        self.loaded[(task, item_id)] = tuple(t.cuda(non_blocking=True) for t in tensors)
        self.load_events[(task, item_id)] = self._record_event()

    def load(self, task):
        """
        Asynchronously start loading all the CPU tensors associated with the task in the reverse order
        they are offloaded (since they will be needed in reverse order).
        """
        torch.cuda.set_device(local_rank())

        with torch.no_grad():
            ready_event = self._record_event()
            with torch.cuda.stream(self.h2d_stream):
                torch.cuda.current_stream().wait_event(ready_event)
                for item_id, tensors in reversed(self.offloaded[task].items()):
                    self._load(task, item_id)

    def get(self, task, item_id: int):
        """
        Retrieve the tensors that are loaded back to GPU with the given task and item_id.
        """
        torch.cuda.set_device(local_rank())

        # if self.loaded[(task, item_id)] does not exist, it means load(task) has not been called yet.
        # this can happen in the first steps, where we do not know the backward -> forward task
        # mapping yet, so we cannot predictively load tensors. in this case, we load it during the get()
        # call. this will block the main computation and cause performance loss, but it is only for the
        # first few steps.

        if (task, item_id) not in self.loaded:
            ready_event = self._record_event()
            with torch.cuda.stream(self.h2d_stream):
                torch.cuda.current_stream().wait_event(ready_event)
                self._load(task, item_id)

        torch.cuda.current_stream().wait_event(self.load_events[(task, item_id)])
        tensors = self.loaded.pop((task, item_id))

        # to clear as often as possible, we also call this at the end of get
        self._clear_tensors_for_finished_offloads()

        return tensors

    def reset(self):
        """ Release all CUDA events. We do this jointly for all events at the end of the step because
        we do not have CPU synchronization wrt these events during the step. """

        for task in self.occupied:
            self.occupied[task] = 0
        for task in self.num_offloaded:
            self.num_offloaded[task] = 0
        self.event_cache += list(self.load_events.values()) + list(self.offload_events.values())
        self.load_events = {}
        self.offload_events = {}
        self.ongoing_offload_tensors = {}
        self.registry.reset()
