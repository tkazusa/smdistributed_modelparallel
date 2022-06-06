# Standard Library
import os
from collections import OrderedDict, defaultdict, deque, namedtuple
from contextlib import contextmanager
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

# Third Party
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import local_rank, pp_rank, pp_size, rank
from smdistributed.modelparallel.torch.module_partition import ModulePartitioner
from smdistributed.modelparallel.torch.patches.checkpoint import CheckpointConfig
from smdistributed.modelparallel.torch.utils import get_tensor_size, rmsg

logger = get_logger()

TensorModuleInfo = namedtuple(
    "TensorModuleInfo", "module_name count tensor_idx is_forward end_index"
)


class TraceResults(NamedTuple):
    mod_execution_order: List[nn.Module]
    traced_input_sizes: Dict[nn.Module, int]
    traced_output_sizes: Dict[nn.Module, int]
    mod_execution_times: Dict[nn.Module, float]
    mod_memory_usage: Dict[nn.Module, float]


class PartitioningAndTraceResults(NamedTuple):
    mod_partitions: Dict[str, int]
    mod_execution_order: List[str]
    traced_input_sizes: Dict[str, int]
    traced_output_sizes: Dict[str, int]
    mod_execution_times: Dict[str, float]
    mod_memory_usage: Dict[str, float]

    def __repr__(self):
        return "<PartTraceResults>"


class ModuleManager:
    def __init__(self):
        self.cfg = None
        self.reset()

    def set_config(self, config):
        self.cfg = config

    def reset(self):
        """
        This helps reset all state so we can run on a different model after reset
        """
        # maintain manual partition assignment
        self._module_partitions: Dict[nn.Module, int] = {}
        self._cur_partition: Optional[int] = None

        # set of activation_checkpoint configs of modules
        self._activation_checkpoint_modules_config = {}

        # flag representing whether tensor parallelism is currently enabled through the manual API.
        # tensor parallelism will be activated on a best-effort basis whenever this is True.
        self._tensor_parallelism_enabled = False

        # set of modules for which we will apply tensor parallelism
        self._tensor_parallelism_modules = set()

        # mapping from module id to dict containing optional config for distribution
        self._tensor_parallelism_config = {}

        # the configuration to be used for the tensor-parallel modules marked in the current context
        self._current_tp_config = {}

        # set of modules that are tensor-parallelized
        self._distributed_modules = set()

        # set of parameters that operate on the batch scaled by tp_size()
        self._scaled_batch_parameters = set()
        self._scaled_batch_buffers = set()

        # set of parameters which are present only on one tp_rank() (tp_rank() == 0)
        # and set to None on others
        self._one_rank_parameters = set()
        self._one_rank_buffers = set()

        # set of parameters that are distributed across tp_ranks. typically will
        # be the same as self._scaled_batch_parameters, but does not have to, depending
        # on how the parameter_creation_scope / initialize_with_input_partition contexts
        # are used in DistributedModule implementation
        self._distributed_parameters = {}
        self._distributed_buffers = {}

        # collect information from tracing
        self._traced_input_sizes: Dict[nn.Module, int] = {}
        self._traced_output_sizes: Dict[nn.Module, int] = {}
        self._module_execution_order: List[nn.Module] = []
        self._module_execution_times: Dict[nn.Module, float] = {}

        # CUDA events that mark the start and end of each module execution during tracing
        self._mod_execution_cuda_events: Dict[nn.Module, Tuple] = {}

        self._module_memory_usage: Dict[nn.Module, float] = {}

        # used for serialization and deserialization as the key to above dicts is a module
        # object local to a process. These dicts help convert them to a string so information
        # can be sent to other processes
        self._module_to_name: Dict[nn.Module, str] = {}
        self._name_to_module: Dict[str, nn.Module] = {}

        # mapping from child module names to parent module names
        self._parent_map: Dict[str, List[str]] = defaultdict(list)

        # sum of _to_recv counts of all children of a module (does not include the module itself)
        # (keyed by (microbatch, module_name))
        self._pending_bwd_counts: Dict[Tuple, int] = defaultdict(lambda: 0)

        # used to identify parent module
        # this is sent and received across ranks for each request
        self._module_execution_stack: List[nn.Module] = []

        # dict of dicts, with first key as microbatch, second key as module
        # and value as a deque data structure. This will be used a stack
        # to record outputs during forward for a microbatch and a module.
        # In the backward pass, outputs will be popped from stack and backward
        # called on them
        self._module_outputs: Dict[int, Dict[Tuple, Deque[Tuple[torch.Tensor]]]] = defaultdict(
            lambda: defaultdict(deque)
        )

        # records outputs and out_grads for a mb, parent_module and module
        # this data structure is used to wait on backward requests from child
        # and club multiple backward requests into a single request
        self._bwd_tensors: Dict[int, Dict[Tuple, Deque[Tuple[torch.Tensor]]]] = defaultdict(
            lambda: defaultdict(deque)
        )

        # stores the number of real responses yet to receive for backward requests. this will be incremented
        # when we have an additional backward request that we haven't received response for
        # when response is received (real) this is decremented
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_recv_real: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # stores the number of dummy repsonses yet to receive for backward requests. this will be incremented
        # when we have an additional backward request that we haven't received response for
        # when response is received (dummy) this is decremented
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_recv_dummy: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # stores the number of  repsonses yet to receive for backward requests for sequential modules. this will be incremented
        # when we have an additional backward request that we haven't received response for.
        # When response is received this is decremented
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_recv_sequential: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # stores the dummy backward request to be sent to a parent on a different rank
        # even if the inputs coming from parent module doesnt require grads, we need
        # to send dummy backward request to let the parent module know that backward execution
        # is complete for the child module. This maintains a count of num dummies to send.
        # dict structure is microbatch -> {parent_module_name -> {module_name -> int}}
        self._to_send_dummy: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # flag to indicate if we need to send back dummy backward request to parent or not
        # dict structure is microbatch -> {position -> {parent_module_name -> {module_name -> bool}}}
        self._to_send_count: Dict[int, Dict[int : Dict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        )

        # mapping of node to bwd_counts for (mb, position, (mod, parent_mod))
        # dict structure is microbatch -> {position -> {(parent_module_name, module_name) -> {node -> int}}}
        self._smpinput_bwd_counts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )

        # records SMPParentRecv backward nodes for a microbatch
        # this data structure is used to do keep track of SMPParentRecv nodes, so that
        # it can be used to check for unsupported scenarios at the end of module
        # execution
        self._smpparents: Dict[int, Dict[str, Deque[Any]]] = defaultdict(
            lambda: defaultdict(lambda: set())
        )

        # records SMPParentRecv backward nodes for a microbatch
        # this data structure is used to do keep track of SMPInput nodes, so that
        # it can be used to check for unsupported scenarios at the end of module
        # execution
        self._smpinputs: Dict[int, Dict[str, Deque[Any]]] = defaultdict(
            lambda: defaultdict(lambda: set())
        )

        # records additional parameters which were passed in the forward
        # pass of the module
        self._additional_params: Dict[nn.Module, set] = defaultdict(set)

        # whether tracing is enabled
        self.measurement_enabled = False

        # The tensor parallel split shapes for a certain weight
        # key: weight tensor, value: list of split shapes for each tp rank
        # To record the unbalanced split and used during loading
        self.weight_split_shapes = {}

        # hooks to be called after the model partition
        self._post_partition_hooks = OrderedDict()

        # hooks to be called after the first execution of smp.step
        self._post_step_hooks = OrderedDict()

    def add_scaled_batch_parameter(self, param):
        self._scaled_batch_parameters.add(param)

    def add_distributed_parameter(self, param, axis):
        self._distributed_parameters[param] = axis

    def is_scaled_batch_parameter(self, param):
        return param in self._scaled_batch_parameters

    def get_parameter_distribution_axis(self, param):
        """ If not distributed returns None """
        return self._distributed_parameters.get(param, None)

    def add_scaled_batch_buffer(self, buf):
        self._scaled_batch_buffers.add(buf)

    def add_distributed_buffer(self, buf, axis):
        self._distributed_buffers[buf] = axis

    def add_one_rank_parameter(self, param):
        self._one_rank_parameters.add(param)

    def add_one_rank_buffer(self, buf):
        self._one_rank_buffers.add(buf)

    def is_one_rank_parameter(self, param):
        return param in self._one_rank_parameters

    def is_one_rank_buffer(self, buf):
        return buf in self._one_rank_buffers

    def is_scaled_batch_buffer(self, buf):
        return buf in self._scaled_batch_buffers

    def get_buffer_distribution_axis(self, buf):
        """ If not distributed returns None """
        return self._distributed_buffers.get(buf, None)

    def register_post_step_hook(self, hook):
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model is not None and state.optimizer is not None and state.model.partitioned:
            hook(state.model, state.optimizer)
            return None

        handle = RemovableHandle(self._post_step_hooks)
        self._post_step_hooks[handle.id] = hook
        return handle

    def register_post_partition_hook(self, hook):
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model is not None and state.optimizer is not None and state.model.partitioned:
            hook(state.model, state.optimizer)
            return None

        handle = RemovableHandle(self._post_partition_hooks)
        self._post_partition_hooks[handle.id] = hook
        return handle

    def replace_module(self, old_module, new_module):
        """ Replace the original module with the new one in the internal data structures """
        from smdistributed.modelparallel.torch.state_mod import state

        assert (
            not state.model.partitioned
        ), "Module replacement can only happen before the model partition."

        name = self._module_to_name[old_module]
        self._module_to_name[new_module] = name
        self._name_to_module[name] = new_module
        del self._module_to_name[old_module]

        self._module_partitions[new_module] = self._module_partitions[old_module]
        del self._module_partitions[old_module]

    def save_smpinput_bwd_count(self, microbatch, position, module, parent_module, nodes):
        """Save bwd counts for SMPInput nodes for a (parent_module, module) pair, position, mb
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        for node in nodes:
            self._smpinput_bwd_counts[microbatch][position][parent_module_name, module_name][
                node
            ] = node.pending_smpinput_bwds
            node.pending_smpinput_bwds = 0

    def load_smpinput_bwd_count(self, microbatch, position, module, parent_module):
        """Loads bwd counts to node.bwd_count attribute, adding to the current count
        for a (parent_module, module) pair, position, mb
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        nodes = self._smpinput_bwd_counts[microbatch][position][parent_module_name, module_name]
        for node, count in nodes.items():
            node.pending_smpinput_bwds = node.pending_smpinput_bwds + count

    def push_smpparents(self, microbatch, module_name, tensors):
        """Save SMPParentbackward nodes in the grad fn of tensors,
        for a particular microbatch
        """
        for tensor in tensors:
            self._smpparents[microbatch][module_name].add(tensor.grad_fn)

    def pop_smpparents(self, microbatch, module_name):
        """Load SMPParentBackward nodes and clear self._smpparents for a particular microbatch
        """
        smpparents = self._smpparents[microbatch][module_name].copy()
        self._smpparents[microbatch][module_name].clear()
        return smpparents

    def push_smpinputs(self, microbatch, module_name, tensors):
        """Save SMPInputbackward nodes in the grad fn of tensors,
        for a particular microbatch
        """
        for tensor in tensors:
            self._smpinputs[microbatch][module_name].add(tensor.grad_fn)

    def pop_smpinputs(self, microbatch, module_name):
        """Load SMPInputBackward nodes for a particular microbatch
        """
        smpinputs = self._smpinputs[microbatch][module_name].copy()
        self._smpinputs[microbatch][module_name].clear()
        return smpinputs

    def push_additional_parameters(self, module_name, param):
        """Push additional parameters corresponding to the module
        """
        module = self.get_module(module_name)
        self._additional_params[module].add(param)

    @property
    def execution_stack(self) -> List[str]:
        """
        Returns names of modules in execution stack so they can be sent to other ranks
        """
        names = []
        for m in self._module_execution_stack:
            names.append(self.get_module_name(m))
        return names

    @execution_stack.setter
    def execution_stack(self, mod_stack: List[str]):
        """
        Uses names of modules received from other ranks and loads the stack with
        actual modules. This stack is used to identify parent module and execution count of a module.
        """
        self._module_execution_stack = []
        for m_name in mod_stack:
            self._module_execution_stack.append(self.get_module(m_name))

    def record_traversal_into_module(self, module: nn.Module):
        self._module_execution_stack.append(module)

    def push_output(
        self, mb: int, module: nn.Module, parent_module: nn.Module, output: Tuple[torch.Tensor]
    ):
        """Pushes the output tensor tuple onto the stack given a microbatch, a
        module and a parent module
        """
        module_name = self.get_module_name(module)
        parent_module_name = self.get_module_name(parent_module)
        self._module_outputs[mb][(parent_module_name, module_name)].append(output)
        logger.debug(
            rmsg(
                f"Saved {1 if isinstance(output, torch.Tensor) else len(output)} output tensor(s) for mb: {mb}, parent: {parent_module_name}, module: {module_name}"
            )
        )

    @contextmanager
    def record_memory_usage(self, module: nn.Module, device: str):
        from smdistributed.modelparallel.torch.state_mod import state

        if device == "gpu" and self.measurement_enabled and not state.no_grad_context:
            torch.cuda.set_device(local_rank())
            prev_mem = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
            yield
            after_mem = torch.cuda.memory_stats()["allocated_bytes.all.allocated"]
            self._module_memory_usage[module] = after_mem - prev_mem
        else:
            yield

    @contextmanager
    def enable_measurement(self, enabled):
        prev_state = self.measurement_enabled
        self.measurement_enabled = enabled
        try:
            yield
        finally:
            self.measurement_enabled = prev_state

    @contextmanager
    def record_execution_time(self, module: nn.Module, device: str):
        torch.cuda.set_device(local_rank())
        current_stream = torch.cuda.current_stream()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(current_stream)
        try:
            yield
        finally:
            end.record(current_stream)
            if self.measurement_enabled and device == "gpu":
                self._mod_execution_cuda_events[module] = (start, end)

    def compute_execution_times(self):
        torch.cuda.synchronize()
        for module, (start, end) in self._mod_execution_cuda_events.items():
            # TODO ideally this should be accumulated instead under module
            # reuse, but doing this introduces subtle complications in auto-partition
            # algorithm when the same module is reused by multiple distinct parents.
            # the current version will be a good approximation unless a significant
            # part of the model consists of reused modules.
            self._module_execution_times[module] = start.elapsed_time(end)

    def get_output(
        self, mb: int, module: nn.Module, parent_module: nn.Module, position: int
    ) -> Tuple[torch.Tensor]:
        """Gets the output tensor tuple in the stack for a given microbatch, module and position
        """
        module_name = self.get_module_name(module)
        parent_module_name = self.get_module_name(parent_module)
        try:
            outp = self._module_outputs[mb][(parent_module_name, module_name)][position]
            return outp
        except IndexError as e:
            raise RuntimeError(
                f"Could not fetch output for mb: {mb}, parent: {parent_module_name}, module: {module_name} with position {position} as it was not saved."
            )

    def enqueue_bwd_tensors(
        self,
        mb: int,
        module: nn.Module,
        parent_module: nn.Module,
        tensors: Union[Sequence[Tuple[torch.Tensor]], Sequence[torch.Tensor]],
    ):
        """Pushes tensors from a backward request onto a queue given a microbatch, module and a parent module
        """
        assert (
            len(tensors) == 2
        ), "tensors should contain two tuples or two tensors: one for output and another for grads"
        module_name = self.get_module_name(module)
        parent_module_name = self.get_module_name(parent_module)
        self._bwd_tensors[mb][(parent_module_name, module_name)].append(tensors)

    def dequeue_all_bwd_tensors(self, mb: int, module: nn.Module, parent_module: nn.Module):
        """Pops all backward tensors from queue and returns a (outputs, output_grads) where
        both outputs and output_grads are list of tensors
        """
        module_name = self.get_module_name(module)
        parent_module_name = self.get_module_name(parent_module)
        bwd_tensors = self._bwd_tensors[mb][(parent_module_name, module_name)]
        outputs, grads = [], []
        while len(bwd_tensors) > 0:
            output, grad = bwd_tensors.popleft()
            outputs.append(output) if isinstance(output, torch.Tensor) else outputs.extend(
                list(output)
            )
            grads.append(grad) if isinstance(grad, torch.Tensor) else grads.extend(list(grad))
        return outputs, grads

    def num_dummy_sends(self, mb: int, module: nn.Module, parent_module: nn.Module) -> int:
        """Gets the number of dummy backward requests to be sent for a module and a parent module
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        return self._to_send_dummy[mb][parent_module_name][module_name]

    def set_dummy_bwd_send(
        self, mb: int, position: int, module: nn.Module, parent_module: nn.Module, count: int
    ):
        """sets the dummy send to True for a mb, position, module and a parent_module
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        logger.debug(
            rmsg(
                f"Setting dummy bwd send for mb: {mb}, module: {self.get_module_name(module)}, parent: {self.get_module_name(parent_module)}"
            )
        )
        self._to_send_count[mb][position][parent_module_name][module_name] = count

    def get_dummy_bwd_send(
        self, mb: int, position: int, module: nn.Module, parent_module: nn.Module
    ):
        """checks if the dummy send flag is set for a mb, position, module and a parent_module
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        return self._to_send_count[mb][position][parent_module_name][module_name]

    def _update_dummy_bwd_sends(
        self, mb: int, module: nn.Module, parent_module: nn.Module, count: int
    ) -> Tuple[torch.Tensor]:
        """Updates the number of dummy backward requests to be sent for a mb, module and a parent module
        by count.
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        self._to_send_dummy[mb][parent_module_name][module_name] = (
            self._to_send_dummy[mb][parent_module_name][module_name] + count
        )
        assert (
            self._to_send_dummy[mb][parent_module_name][module_name] >= 0
        ), "shouldnt go less than 0, there is a bug"

    def decrement_dummy_bwd_sends(
        self, mb: int, module: nn.Module, parent_module: nn.Module, count: int
    ):
        logger.debug(
            rmsg(
                f"Decrementing dummy bwd sends count for mb: {mb}, module: {self.get_module_name(module)}, parent: {self.get_module_name(parent_module)}, count: {count}"
            )
        )
        self._update_dummy_bwd_sends(mb, module, parent_module, -count)

    def increment_dummy_bwd_sends(
        self, mb: int, module: nn.Module, parent_module: nn.Module, count: int = 1
    ):
        logger.debug(
            rmsg(
                f"Incrementing dummy bwd sends count for mb: {mb}, module: {self.get_module_name(module)}, parent: {self.get_module_name(parent_module)}"
            )
        )
        self._update_dummy_bwd_sends(mb, module, parent_module, count)

    def _get_pending_backward(
        self, mb: int, module: nn.Module, parent_module: nn.Module, real=True, sequential=False
    ) -> int:
        """Gets the number of responses pending for a backward request for a parent module and module
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        if sequential:
            recv = self._to_recv_sequential
        else:
            recv = self._to_recv_real if real else self._to_recv_dummy
        return recv[mb][parent_module_name][module_name]

    def _update_pending_bwd_count(
        self,
        mb: int,
        module: nn.Module,
        parent_module: nn.Module,
        count: int,
        real: bool = True,
        sequential: bool = False,
    ):
        """Updates the number of responses to be received for a mb, module and a parent module
        by count
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        if sequential:
            recv = self._to_recv_sequential
        else:
            recv = self._to_recv_real if real else self._to_recv_dummy
        recv[mb][parent_module_name][module_name] += count
        assert recv[mb][parent_module_name][module_name] >= 0, (
            mb,
            parent_module_name,
            module_name,
            "shouldnt go less than 0, there is a bug",
        )

        self._update_parent_pending_counts(module_name, parent_module_name, mb, count, set())

    def _update_parent_pending_counts(self, module_name, parent_module_name, mb, count, visited):
        """ Update the pending backward counts of all ancestors """
        if (
            parent_module_name is not None
            and not module_name == "main"
            and module_name not in visited
        ):
            self._pending_bwd_counts[(mb, parent_module_name)] += count
            visited.add(module_name)

            if self._parent_map[parent_module_name] is not None:
                for grandparent_name in self._parent_map[parent_module_name]:
                    self._update_parent_pending_counts(
                        parent_module_name, grandparent_name, mb, count, visited
                    )

    def decrement_bwd_count(
        self,
        mb: int,
        module: nn.Module,
        parent_module: nn.Module,
        count: int,
        real=True,
        sequential=False,
    ):
        """Decrements the expected number of backward responses to be received for a mb, module and a parent module
        by 1 after we are doing executing a backward
        """
        logger.debug(
            rmsg(
                f"Decrementing bwd count for mb: {mb}, module: {self.get_module_name(module)}, parent: {self.get_module_name(parent_module)}"
            )
        )
        self._update_pending_bwd_count(mb, module, parent_module, -count, real, sequential)

    def increment_bwd_count(
        self, mb: int, module: nn.Module, parent_module: nn.Module, real=True, sequential=False
    ):
        """Increments expected number of responses to be received for a mb, module and a parent module
        by 1
        """
        logger.debug(
            rmsg(
                f"Incrementing pending bwd count for mb:{mb}, module: {self.get_module_name(module)}, parent: {self.get_module_name(parent_module)}"
            )
        )
        self._update_pending_bwd_count(mb, module, parent_module, 1, real, sequential)

    def check_no_pending_bwd(self, mb: int, module: nn.Module) -> bool:
        """Checks if the descendants whose children execute on different
        rank have _to_recv[mb][module_name] as 0 for all keys"""
        return self._pending_bwd_counts[(mb, self._module_to_name[module])] == 0

    def find_boundary_ancestors(self, mb: int, module: nn.Module) -> Tuple[str]:
        """Finds boundary ancestors for the module.
        Boundary ancestors means that an ancestor whose executor is not the current
        rank and a child of the ancestor whose executor is the current rank
        """
        current_mod = module
        current_mod_name = self.get_module_name(module)
        assert self.is_executor(current_mod), "the rank needs to be executor of the module"
        if self.is_main_module(current_mod) or not self.is_parent_executor(current_mod):
            # parent of main is None
            return self.get_parent_module(current_mod), current_mod
        while not self.is_main_module(current_mod) and self.is_parent_executor(current_mod):
            child_mod = current_mod
            current_mod = self.get_parent_module(current_mod)
        child_mod = current_mod
        current_mod = self.get_parent_module(current_mod)
        return current_mod, child_mod

    def output_stack_size(self, mb: int, module: nn.Module, parent_module: nn.Module) -> int:
        """Returns output_stack_size given a microbatch, module and parent module
        This output stack size for a parent_module, module and a microbatch should be same
        on the rank executing parent_module and the rank executing module at the end of forward.
        """
        module_name = self.get_module_name(module)
        parent_module_name = self.get_module_name(parent_module)
        if (
            mb in self._module_outputs
            and (parent_module_name, module_name) in self._module_outputs[mb]
        ):
            return len(self._module_outputs[mb][(parent_module_name, module_name)])
        return 0

    def get_parent_module(self, module: nn.Module) -> nn.Module:
        """
        Traverse from end of the module execution stack to identify the immediate parent of given module.
        We do this from the end because we might have gone into this module multiple times.
        Returns the parent module.
        """
        if self.is_main_module(module):
            return None
        module_name = self.get_module_name(module)
        # TODO: Requires optimization
        if len(self.execution_stack):
            if module_name not in self.execution_stack:
                return self._module_execution_stack[-1]
            else:
                index = self.execution_stack.index(module_name)
                return self._module_execution_stack[index - 1] if index > 0 else None
        raise RuntimeError(
            f"Exec stack was {self.execution_stack}; could not find parent_module of {self.get_module_name(module)}"
        )

    def get_parameters(self, module, recurse=True):
        """
        Yields the parameters owned by the model as well as
        additional parameters passed in forward.
        """
        for param in module.parameters(recurse=recurse):
            yield param
        for param in self._additional_params[module]:
            yield param

    def is_executor(self, module: nn.Module) -> bool:
        partition = self.get_partition(module)
        assert partition is not None, (module, "is not assigned any partition")
        return partition == pp_rank()

    def is_parent_executor(self, module: nn.Module) -> bool:
        if self.is_main_module(module):
            is_parent_executor = False
        else:
            parent_module = self.get_parent_module(module)
            is_parent_executor = self.get_partition(parent_module) == pp_rank()
        return is_parent_executor

    def is_main_module(self, module: nn.Module) -> bool:
        """
        Check if the module is main module.
        main module is a module with no parent. In other words,
        its the module on which smp.distribute_model was called
        """
        return self.get_module_name(module) == "main"

    def is_correct_parent(self, module, parent_module):
        """Checks if the parent_module is the the correct parent of module
        """
        parent_module_name = self.get_module_name(parent_module)
        module_name = self.get_module_name(module)
        if self._parent_map[module_name]:
            return parent_module_name in self._parent_map[module_name]
        else:
            return self.is_main_module(module)

    def get_immediate_ancestors(self, module):
        """Gets parent and grand parent of a module"""
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        assert not self.is_main_module(module), "cannot get_immediate_ancestors for main module"
        stack_parent_module = self.get_parent_module(module)
        module_name = self.get_module_name(module)
        stack_parent_name = self.get_module_name(stack_parent_module)
        ancestors = self._get_ancestors(module_name, stack_parent_name)
        assert len(ancestors) >= 2, "ancestors should atleast hold this module and parent"
        if len(ancestors) == 2:
            if self._parent_map[ancestors[-1]]:
                grand_parent_name = self._parent_map[ancestors[-1]][-1]
            else:
                grand_parent_name = None
            grand_parent_module = (
                None if not grand_parent_name else self.get_module(grand_parent_name)
            )
            parent_module = self.get_module(ancestors[-1])
        else:
            parent_module = self.get_module(ancestors[1])
            grand_parent_module = self.get_module(ancestors[2])

        return grand_parent_module, parent_module

    def _get_ancestors(self, module_name, stack_parent_name):
        """For a valid module name and stack parent module name
        returns the path from module to stack_parent"""
        visited = set()
        queue = deque()
        queue.append([module_name])
        while queue:
            current_path = queue.popleft()
            last = current_path[-1]
            if not last:
                continue
            elif last == stack_parent_name:
                return current_path
            if not self._parent_map[last]:
                continue
            for parent in self._parent_map[last]:
                if parent in visited:
                    continue
                visited.add(parent)
                new_path = list(current_path)
                new_path.append(parent)
                queue.append(new_path)
        assert False, f"path not found between {module_name} and {stack_parent_name}"

    def finished_module_exec(self):
        self._module_execution_stack.pop()

    def clear_microbatch_state(self, mb):
        self._to_recv_real.pop(mb, None)
        self._to_recv_dummy.pop(mb, None)
        self._to_recv_sequential.pop(mb, None)

        self._to_send_dummy.pop(mb, None)
        self._to_send_count.pop(mb, None)

        self._module_outputs.pop(mb, None)
        # not clearing self._pending_bwd_counts here as it is not keyed by microbatch
        # this is just int and should have no memory issues,
        # it's cleared at the end of step
        self._smpinput_bwd_counts.pop(mb, None)

    def clear_minibatch_state(self):
        """
        Clear minibatch state before start of a new minibatch
        """
        self._to_recv_real.clear()
        self._to_recv_dummy.clear()
        self._to_recv_sequential.clear()

        self._to_send_dummy.clear()
        self._to_send_count.clear()

        self._module_outputs.clear()
        self._pending_bwd_counts.clear()
        self._smpinput_bwd_counts.clear()

    def clear_tensor_parallelism_modules(self):
        self._tensor_parallelism_modules.clear()

    def auto_partition_model(self, model, root_rank):

        trace_results = TraceResults(
            self._module_execution_order,
            self._traced_input_sizes,
            self._traced_output_sizes,
            self._module_execution_times,
            self._module_memory_usage,
        )
        mp = ModulePartitioner(
            model,
            pp_size(),
            trace_results,
            self.cfg.memory_weight,
            model.trace_execution_times,
            model.trace_memory_usage,
        )
        partitions: Dict[nn.Module, int] = mp.partition()
        for mod, part in partitions.items():
            self.assign_partition(mod, part)

    def assign_partition(self, module: nn.Module, partition: Optional[int] = None):
        # this is called inside a init call, for some reason cant print module object here
        if partition is None:
            partition = self._cur_partition
        self._module_partitions[module] = partition

    def assign_unassigned_modules(self, module: nn.Module):
        for m in module.modules():
            if m not in self._module_partitions or self._module_partitions[m] is None:
                self._module_partitions[m] = self.cfg.default_partition
            if rank() == 0:
                logger.debug(self.get_module_name(m) + " " + str(self._module_partitions[m]))

    def get_partition(self, module: nn.Module):
        return self._module_partitions[module]

    def should_tensor_parallelize(self, module):
        return module in self._tensor_parallelism_modules

    def register_distributed(self, module: nn.Module):
        """ Mark the module and all its descendants as distributed/tensor-parallelized. """

        self._distributed_modules.add(module)
        for c in module.children():
            self.register_distributed(c)

    def is_distributed(self, module: nn.Module):
        return module in self._distributed_modules

    def should_checkpoint_activations(self, module):
        return module in self._activation_checkpoint_modules_config

    def get_checkpoint_activations_config(self, module):
        return self._activation_checkpoint_modules_config[module]

    def get_tp_config(self, module):
        return self._tensor_parallelism_config.get(module, {})

    def check_module_partition(self, module):
        rank = None
        for m in module.modules():
            if rank == None:
                rank = self.get_partition(m)
            else:
                if rank != self.get_partition(m):
                    return False
        return True

    def maybe_mark_for_tensor_parallelism(self, module: nn.Module):
        """ If tensor parallelism is currently enabled and the module is supported,
            mark the module for tensor parallelism. """

        from smdistributed.modelparallel.torch.state_mod import state

        # not using isinstance because sub-classes of supported modules may not be supported
        if self._tensor_parallelism_enabled and state.tp_registry.is_supported(type(module)):
            self._tensor_parallelism_modules.add(module)
            self._tensor_parallelism_config[module] = self._current_tp_config

    def set_activation_checkpointing(
        self, module, preserve_rng_state=True, pack_args_as_tuple=False, strategy="each"
    ):
        from smdistributed.modelparallel.torch.state_mod import state

        if not state.model:
            raise RuntimeError(
                "set_activation_checkpointing can only be called on a module after wrapping the main model with smp.DistributedModel"
            )

        """ Enable activation checkpointing for the given module. """
        if not isinstance(module, nn.Module):
            raise ValueError(
                "Only a module of type nn.Module can be passed for activation checkpointing"
            )
        if not isinstance(module, nn.Sequential):
            if pack_args_as_tuple:
                raise ValueError("pack_args_as_tuple can be True only for Sequential modules")
            if strategy != "each":
                # each is just the default, if user tries to change this, we throw error
                raise ValueError("strategy can only be used when checkpointing Sequential modules")

        self._activation_checkpoint_modules_config[module] = CheckpointConfig(
            enabled=True,
            preserve_rng_state=preserve_rng_state,
            pack_args_as_tuple=pack_args_as_tuple,
            module_name=self.get_module_name(module),
            strategy=strategy,
        )

    def set_tensor_parallelism(self, module, enabled=True, **tp_config):
        """ Enable or disable tensor parallelism for the given module. If disabling, disable for the entire sub-tree.
            If enabling, it is enabled for the top-most level supported modules only. """
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model:
            raise RuntimeError(
                "set_tensor_parallelism API can only be used before creating the smp.DistributedModel object."
            )

        if not enabled:
            # if disabling, disable for the entire subtree
            if module in self._tensor_parallelism_modules:
                name = module.__class__.__name__
                logger.warning(
                    f"Disabling previously-enabled tensor parallelism for module of type {name}."
                )
            self._tensor_parallelism_modules.discard(module)
            for c in module.children():
                self.set_tensor_parallelism(c, False)
        else:
            # if enabling, enable for the topmost-level supported modules only
            stack = [module]
            visited = set()
            while len(stack) > 0:
                m = stack.pop()
                if m not in visited:
                    visited.add(m)
                    if state.tp_registry.is_supported(type(m)):
                        self._tensor_parallelism_modules.add(m)
                        self._tensor_parallelism_config[m] = tp_config
                    else:
                        stack.extend([c for c in m.children()])

    def set_partition(self, module, partition, recurse=True):
        """
        Assign the given module to the specified partition. Can only be called between creation of smp.DistributedModel and
        the first call to a smp.step-decorated function.
        """
        from smdistributed.modelparallel.torch.state_mod import state

        if state.model.partitioned:
            raise RuntimeError(
                "set_partition API can only be used before the first call to an smp.step-decorated function."
            )

        if self.cfg.auto_partition:
            return

        if partition < 0 or partition >= self.cfg.pipeline_parallel_degree:
            raise ValueError(
                "Partition ID must be non-negative, and less than the pipeline parallel degree."
            )

        self._module_partitions[module] = partition

        if recurse:
            for c in module.children():
                self.set_partition(c, partition)

    @contextmanager
    def tensor_parallelism(self, enabled=True, **tp_config):
        """
        Context manager for manual tensor parallellism. If enabled=True, tensor parallelism will
        be applied to any supported Module object created within this context, unless there is
        and inner context manager that sets enabled=False.
        """
        from smdistributed.modelparallel.torch.state_mod import state

        if not state.initialized:
            yield
        else:
            _prev_state = self._tensor_parallelism_enabled
            _prev_config = self._current_tp_config
            self._tensor_parallelism_enabled = enabled
            self._current_tp_config.update(tp_config)
            try:
                yield
            finally:
                self._tensor_parallelism_enabled = _prev_state
                self._current_tp_config = _prev_config

    def simplify_tensor_parallelism_modules(self, model):
        """ If a module is marked for tensor parallelism, unmark all its descendants for tensor parallelism. Also unmark
        modules that share parameters with other modules. """

        params_to_modules = defaultdict(lambda: set())

        # schema: (module, ancestor_marked_for_tp), where the latter is an ancestor module that is marked for tp

        # traverse the modules dfs
        stack = [(model, None)]
        visited = set()
        while len(stack) > 0:
            module, ancestor_marked_for_tp = stack.pop()
            if module not in visited:
                visited.add(module)

                if ancestor_marked_for_tp is not None:
                    topmost_module_marked_for_tp = ancestor_marked_for_tp
                    if module in self._tensor_parallelism_modules:
                        # remove since an ancestor is already marked
                        self._tensor_parallelism_modules.remove(module)
                elif module in self._tensor_parallelism_modules:
                    topmost_module_marked_for_tp = module
                else:
                    topmost_module_marked_for_tp = None

                for p in module.parameters(recurse=False):
                    params_to_modules[p].add((module, topmost_module_marked_for_tp))

                stack.extend([(c, topmost_module_marked_for_tp) for c in module.children()])

        # unmark for tensor parallelism if sharing parameters with another module
        for param, modules_marks in params_to_modules.items():
            tp_ancestors = set([a for _, a in modules_marks])
            if len(tp_ancestors) > 1:
                # there are multiple, distinct, distributed modules sharing parameters
                # disabling tp for all
                for m, ancestor_marked_for_tp in modules_marks:
                    if ancestor_marked_for_tp is not None:
                        logger.warning(
                            f"Disabling tensor parallelism for module of type {type(ancestor_marked_for_tp)} since it shares parameters with another module."
                        )
                        self._tensor_parallelism_modules.discard(ancestor_marked_for_tp)

    @contextmanager
    def partition(self, i: int):
        """
        Context manager to help with manual assignment. A Module object created within this context is assigned the partition i
        """
        from smdistributed.modelparallel.torch.state_mod import state

        if not state.initialized or self.cfg.auto_partition:
            yield
        else:
            if i < 0 or i >= self.cfg.pipeline_parallel_degree:
                raise ValueError(
                    "Partition ID must be non-negative, and less than the pipeline parallel degree."
                )

            _prev_partition = self._cur_partition
            self._cur_partition = i
            try:
                yield
            finally:
                self._cur_partition = _prev_partition

    def save_input_size(self, module: nn.Module, size: int):
        self._traced_input_sizes[module] = size

    def save_output_size(self, module: nn.Module, size: int):
        self._traced_output_sizes[module] = size

    def record_execution_order(self, module: nn.Module):
        self._module_execution_order.append(module)

    def name_modules_and_create_parent_map(self):
        """
        Converts the key in tracing information dictionaries from module object to a string identifier.
        The format of this name is a '/' separated list of module names. The main model is named 'main'.
        Any child module is named with the variable name used for that module in code.
        For example,
        class A1:
            self.b = A2()

        class A2:
            self.c = A3()

        model = A1()

        A1 -> main
        A2 -> main/b
        A3 -> main/b/c
        """
        # local import as module manager itself is a member of state and causes circular imports
        from smdistributed.modelparallel.torch.state_mod import state

        if not self._module_to_name:
            main_module = state.model
            self._module_to_name[main_module] = "main"
            self._name_to_module["main"] = main_module

            def record_child_module_name(parent: nn.Module):
                # sort for deterministic ordering
                named_children = sorted(list(parent.named_children()), key=lambda x: x[0])
                for n, c in named_children:
                    child_name = os.path.join(self._module_to_name[parent], n)
                    if c not in self._module_to_name:
                        self._module_to_name[c] = child_name
                        self._name_to_module[child_name] = c
                        # if a module is held by more than one parent module, there can be more than
                        # one path from root to that module, store first name for now
                        # since this is only for serialization and deserialization,
                        # we only need consistent name across all
                    self._parent_map[self._module_to_name[c]].append(self._module_to_name[parent])
                    record_child_module_name(c)

            self._parent_map[self._module_to_name[main_module]] = None
            record_child_module_name(main_module)

    def get_module_name(self, module: nn.Module):
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        try:
            return self._module_to_name[module]
        except KeyError:
            raise RuntimeError(
                f"Cannot find the name for the module {module}. This is commonly caused by module instantiation inside a forward() method. Please create all child modules inside the __init__ method of its parent module."
            )

    def get_module(self, name: str):
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        return self._name_to_module[name]

    def get_module_names(self):
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()
        return self._module_to_name.values()

    def get_serialized_partitioning_and_tracing_states(self) -> PartitioningAndTraceResults:
        """
        Refer comment in name_modules.
        Used by the sender of rank which did the tracing while sending to other ranks.
        """
        if not self._module_to_name:
            self.name_modules_and_create_parent_map()

        mod_partitions_by_names = {}
        mod_exc_order_by_names = []
        traced_output_sizes_by_names = {}
        traced_input_sizes_by_names = {}
        mod_exc_times_by_names = {}
        mod_mem_usage_by_names = {}
        for m, n in self._module_to_name.items():
            mod_partitions_by_names[n] = self.get_partition(m)
        for m in self._module_execution_order:
            mod_exc_order_by_names.append(self.get_module_name(m))
        for k, s in self._traced_output_sizes.items():
            traced_output_sizes_by_names[self.get_module_name(k)] = s
        for k, s in self._traced_input_sizes.items():
            traced_input_sizes_by_names[self.get_module_name(k)] = s
        for k, s in self._module_execution_times.items():
            mod_exc_times_by_names[self.get_module_name(k)] = s
        for k, s in self._module_memory_usage.items():
            mod_mem_usage_by_names[self.get_module_name(k)] = s

        data = PartitioningAndTraceResults(
            mod_partitions_by_names,
            mod_exc_order_by_names,
            traced_input_sizes_by_names,
            traced_output_sizes_by_names,
            mod_exc_times_by_names,
            mod_mem_usage_by_names,
        )
        return data

    def load_partitioning_and_trace_results(self, trace_results: PartitioningAndTraceResults):
        """
        Ranks other than the one which did the tracing receive tracing information and load them.
        """
        if not self._module_to_name or not self._name_to_module:
            self.name_modules_and_create_parent_map()

        # TODO: raise user friendly message when wrong results are loaded
        mod_partitions, mod_exc_order, traced_input_sizes, traced_output_sizes, mod_exc_times, mod_mem_usage = (
            trace_results
        )
        for m in mod_partitions:
            try:
                self._module_partitions[self.get_module(m)] = mod_partitions[m]
            except KeyError:
                # skip modules not part of original model, as this rank hasn't seen those modules
                # these will be treated as partition: None anyway
                pass

        for m in mod_exc_order:
            try:
                self._module_execution_order.append(self.get_module(m))
            except KeyError:
                pass

        for n, s in traced_input_sizes.items():
            try:
                self._traced_input_sizes[self.get_module(n)] = s
            except KeyError:
                pass

        for n, s in traced_output_sizes.items():
            try:
                self._traced_output_sizes[self.get_module(n)] = s
            except KeyError:
                pass

        for n, s in mod_exc_times.items():
            try:
                self._module_execution_times[self.get_module(n)] = s
            except KeyError:
                pass

        for n, s in mod_mem_usage.items():
            try:
                self._module_memory_usage[self.get_module(n)] = s
            except KeyError:
                pass

    def get_metrics(self):
        """
        Calculate the metrics to upload to Sagemaker studio
        return:
            var_size: List[num_partitions], variable size for each device
            module_fraction: List[num_partitions], module fraction for each device
            comm_vol: scalar, The total communication volume
        """
        total_modules = len(self._traced_output_sizes)
        module_fraction = [0 for _ in range(pp_size())]
        var_size = [0 for _ in range(pp_size())]
        for module in self._traced_output_sizes.keys():
            dev = self._module_partitions[module]
            if dev == None:  # Loss modules will always be executed locally
                continue
            module_fraction[dev] += 1
            for var in module.parameters(recurse=False):
                var_size[dev] += get_tensor_size(var)
        if total_modules > 0:
            module_fraction = [x / total_modules for x in module_fraction]

        comm_vol = self._get_total_communication_volume()

        return var_size, module_fraction, comm_vol

    def _get_total_communication_volume(self, unit="MB"):
        """
        Using BFS to travers the module tree to collect total communication volume
        Only the forward communication volume will be collected
        """
        comm_vol = 0
        root = self._name_to_module["main"]
        current_level = [root]

        while len(current_level) > 0:
            next_level = []
            for node in current_level:
                parent_dev = self._module_partitions[node]
                for child in node.children():
                    child_dev = self._module_partitions[child]
                    if child_dev == None:
                        continue
                    if parent_dev != child_dev:
                        if child in self._traced_input_sizes:
                            comm_vol += self._traced_input_sizes[child]
                        if child in self._traced_output_sizes:
                            comm_vol += self._traced_output_sizes[child]
                    next_level.append(child)
            current_level = next_level

        if unit == "MB":
            return comm_vol / 1e6
        elif unit == "GB":
            return comm_vol / 1e9
        else:
            raise ValueError(f"Unsupported unit {unit}")
