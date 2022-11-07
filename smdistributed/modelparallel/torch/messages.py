# Standard Library
from typing import Any, Dict, List, NamedTuple, Tuple, Union

# Third Party
import torch
import torch.nn as nn

# First Party
from smdistributed.modelparallel.torch.core import pp_rank
from smdistributed.modelparallel.torch.exceptions import UnsupportedMessageError
from smdistributed.modelparallel.torch.patches.checkpoint import CheckpointConfig
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.state_mod import state


"""
This module holds the different messages that module Servers send to each other, and their dependencies.
"""


class StepEndNotification:
    pass


class StubbedMessage:
    def stubify(self):
        """
        Convert the tensors which are members of this object to TensorStubs
        """
        raise UnsupportedMessageError

    @classmethod
    def destubify(cls, msg, tensors: List[torch.Tensor]):
        """
        Replace tensor stubs in msg with actual tensors
        """
        return state.serialization_manager.deserialize(msg, tensors)


class Request:
    pass


class ExecutionRequest(Request):
    pass


class StepExecutionRequest(StubbedMessage, ExecutionRequest):
    def __init__(
        self,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        mb: int,
        step_fn_id: int,
        enable_grads: bool,
        enable_autocast: bool = False,
        requester: int = 0,
    ):
        self.args = args
        self.kwargs = kwargs
        self.mb = mb
        self.step_fn_id = step_fn_id
        self.enable_grads = enable_grads
        self.enable_autocast = enable_autocast
        self.requester = requester

    def __repr__(self):
        return f"<StepExecReq::mb:{self.mb}, requester:{self.requester}>"

    def strip_tensors(self):
        self.args = ()
        self.kwargs = {}

    def __eq__(self, other):
        return (
            self.mb == other.mb
            and self.enable_grads == other.enable_grads
            and self.enable_autocast == other.enable_autocast
            and self.requester == other.requester
        )

    def __hash__(self):
        return (
            hash("StepExecutionRequest")
            + hash(self.mb)
            + hash(self.step_fn_id)
            + hash(self.enable_grads)
            + hash(enable_autocast)
            + hash(requester)
        )


class TraceStepExecutionRequest(StepExecutionRequest):
    def __repr__(self):
        return f"<TraceReq::mb:{self.mb}, requester:{self.requester}>"


class ModuleExecutionRequest(ExecutionRequest, StubbedMessage):
    _id = 0

    def __init__(
        self,
        module: str,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        execution_stack: List[str],
        executor: int,
        requester: int,
        mb: int,
        phase: MbStatus,
        enable_grads: bool = True,
        position: int = -1,
        sender_module: str = None,
        tensor_idx: int = -1,
        enable_autocast: bool = False,
        checkpoint_activations_config: CheckpointConfig = None,
    ):
        self.id = (pp_rank(), ModuleExecutionRequest._id)
        ModuleExecutionRequest._id += 1

        self.module = module
        self.args: Tuple[Any] = args
        self.kwargs = kwargs
        self.execution_stack = execution_stack
        self.executor = executor
        self.requester = requester
        self.mb = mb
        self.phase = phase
        self.enable_grads = enable_grads
        self.sender_module = sender_module
        self.position = position
        self.tensor_idx = tensor_idx
        self.enable_autocast = enable_autocast
        self.checkpoint_activations_config = checkpoint_activations_config

    def __repr__(self):
        if self.phase == MbStatus.BWD:
            return f"<ModExecReq::BWD::mb:{self.mb}, module:{self.module}, sender_module: {self.sender_module}, requester:{self.requester}, executor:{self.executor}, position: {self.position}>"
        else:
            return f"<ModExecReq::FWD::mb:{self.mb}, module:{self.module}, requester:{self.requester}, executor:{self.executor}, {self.execution_stack}, checkpoint: {self.checkpoint_activations_config}>"

    def __hash__(self):
        return (
            hash(self.module)
            + hash(tuple(self.execution_stack))
            + hash(self.executor)
            + hash(self.requester)
            + hash(self.mb)
            + hash(self.phase)
            + hash(self.enable_grads)
            + hash(self.position)
            + hash(self.tensor_idx)
            + hash(self.enable_autocast)
            + hash(self.checkpoint_activations_config)
        )

    def __eq__(self, other):
        return (
            self.module == other.module
            and self.execution_stack == other.execution_stack
            and self.executor == other.executor
            and self.requester == other.requester
            and self.mb == other.mb
            and self.phase == other.phase
            and self.enable_grads == other.enable_grads
            and self.position == other.position
            and self.tensor_idx == other.tensor_idx
            and self.enable_autocast == other.enable_autocast
            and self.checkpoint_activations_config == other.checkpoint_activations_config
        )

    @classmethod
    def create_forward_req(
        cls,
        module: nn.Module,
        args: List[Any],
        kwargs: Dict[str, Any],
        enable_grads: bool,
        enable_autocast: bool,
        checkpoint_activations_config: CheckpointConfig = None,
        position: int = -1,
    ):
        return cls(
            module=state.module_manager.get_module_name(module),
            args=args,
            kwargs=kwargs,
            execution_stack=state.module_manager.execution_stack,
            executor=state.module_manager.get_partition(module),
            requester=pp_rank(),
            mb=state.microbatch,
            phase=MbStatus.FWD,
            enable_grads=enable_grads,
            enable_autocast=enable_autocast,
            checkpoint_activations_config=checkpoint_activations_config,
            position=position,
        )

    @classmethod
    def create_backward_req(
        cls,
        module: nn.Module,
        output_grads: Union[torch.Tensor, Tuple[torch.Tensor]],
        position: int,
        sender_module: nn.Module,
        execution_stack: List[str],
        tensor_idx: int = -1,
    ):
        # enable_grads is True and enable_autocast is False by default
        return cls(
            module=state.module_manager.get_module_name(module),
            args=output_grads,
            kwargs=None,
            execution_stack=execution_stack,
            executor=state.module_manager.get_partition(module),
            requester=pp_rank(),
            mb=state.microbatch,
            phase=MbStatus.BWD,
            position=position,
            sender_module=state.module_manager.get_module_name(sender_module),
            tensor_idx=tensor_idx,
        )

    def stubify(self, peers):
        return state.serialization_manager.serialize(obj=self, c2c_possible=True, peers=peers)

    def strip_tensors(self):
        self.args = ()
        self.kwargs = {}


class SequentialModulesExecutionRequest(ExecutionRequest, StubbedMessage):
    _id = 0

    def __init__(
        self,
        module: str,
        start_layer_idx: int,  # inclusive
        end_layer_idx: int,  # exclusive
        args: Tuple[Any],
        execution_stack: List[str],  # only has till sequential
        executor: int,
        requester: int,
        mb: int,
        phase: MbStatus,
        enable_grads: bool = True,
        position: int = -1,
        sender_module: str = None,
        enable_autocast: bool = False,
        checkpoint_activations_config: CheckpointConfig = None,
    ):
        self.id = (pp_rank(), ModuleExecutionRequest._id)
        ModuleExecutionRequest._id += 1

        self.module = module
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx
        self.args = args
        self.kwargs = None
        self.execution_stack = execution_stack
        self.executor = executor
        self.requester = requester
        self.mb = mb
        self.phase = phase
        self.enable_grads = enable_grads
        self.sender_module = sender_module
        self.position = position
        self.enable_autocast = enable_autocast
        self.checkpoint_activations_config = checkpoint_activations_config

    def __repr__(self):
        return f"<SeqModExecReq::mb:{self.mb}, module:{self.module}[{self.start_layer_idx}:{self.end_layer_idx}] requester:{self.requester}, executor:{self.executor}, {self.execution_stack}, checkpoint: {self.checkpoint_activations_config}>"

    def __hash__(self):
        cur_hash = (
            hash(self.module)
            + hash(self.start_layer_idx)
            + hash(self.end_layer_idx)
            + hash(tuple(self.execution_stack))
            + hash(self.executor)
            + hash(self.requester)
            + hash(self.mb)
            + hash(self.phase)
            + hash(self.enable_grads)
            + hash(self.enable_autocast)
            + hash(self.checkpoint_activations_config)
        )
        return cur_hash

    def __eq__(self, other):
        return (
            self.module == other.module
            and self.start_layer_idx == other.start_layer_idx
            and self.end_layer_idx == other.end_layer_idx
            and self.execution_stack == other.execution_stack
            and self.executor == other.executor
            and self.requester == other.requester
            and self.mb == other.mb
            and self.phase == other.phase
            and self.enable_grads == other.enable_grads
            and self.enable_autocast == other.enable_autocast
            and self.checkpoint_activations_config == other.checkpoint_activations_config
        )

    @classmethod
    def create_forward_req(
        cls,
        module: nn.Module,
        start_layer_idx: int,
        end_layer_idx: int,
        args: Tuple[Any],
        enable_grads: bool,
        enable_autocast: bool,
        checkpoint_activations_config: CheckpointConfig = None,
        position: int = -1,
    ):
        return cls(
            module=state.module_manager.get_module_name(module),
            start_layer_idx=start_layer_idx,
            end_layer_idx=end_layer_idx,
            args=args,
            execution_stack=state.module_manager.execution_stack,
            executor=state.module_manager.get_partition(module[start_layer_idx]),
            requester=pp_rank(),
            mb=state.microbatch,
            phase=MbStatus.FWD,
            enable_grads=enable_grads,
            enable_autocast=enable_autocast,
            checkpoint_activations_config=checkpoint_activations_config,
            position=position,
        )

    def stubify(self, peers):
        return state.serialization_manager.serialize(obj=self, c2c_possible=True, peers=peers)

    def strip_tensors(self):
        self.args = ()


class ExecutionResult:
    pass


class ForwardExecutionResult(ExecutionResult, StubbedMessage):
    def __init__(self, request: ExecutionRequest, outputs: Any):
        self.request = request
        self.mb = request.mb
        self.outputs = outputs
        self.request.strip_tensors()

    def __repr__(self):
        return f"<ExecResult::{self.request}>"

    def stubify(self, peers):
        return state.serialization_manager.serialize(obj=self, c2c_possible=True, peers=peers)

    def __hash__(self):
        return hash("ForwardExecutionResult") + hash(self.request)

    def __eq__(self, other):
        return self.mb == other.mb and self.request == other.request

    def strip_outputs(self):
        self.outputs = None


class MicrobatchEndResult(StubbedMessage):
    def __init__(self, mb: int, outputs: Any):
        self.mb = mb
        self.outputs = outputs

    def __repr__(self):
        return f"<MbEndResult::mb:{self.mb}>"

    def stubify(self, peers):
        return state.serialization_manager.serialize(obj=self, c2c_possible=False, peers=peers)

    def __eq__(self, other):
        return self.mb == other.mb

    def strip_outputs(self):
        self.outputs = None

    def __hash__(self):
        return hash("MicrobatchEndResult") + hash(self.mb)


class WaitForBackwardRequest(Request, NamedTuple):
    """
    Sent from worker thread of pp_rank=0 to server. This wait helps us control
    how forward and backward passes are executed according to pipelining strategy.
    """

    mb: int

    def __repr__(self):
        return f"<WaitForBwd::mb:{self.mb}>"

    def __eq__(self, other):
        return self.mb == other.mb

    def __hash__(self):
        return hash("WaitForBackwardRequest") + hash(self.mb)


class BackwardExecutionResult(ExecutionResult):
    def __init__(self, req):
        self.req = req
        self.mb = self.req.mb
        self.req.strip_tensors()

    def __repr__(self):
        return f"<BackwardExecResult::{self.req}>"

    def __hash__(self):
        return hash("BackwardExecutionResult") + hash(self.req)


class PartialBackwardResult(ExecutionResult):
    def __init__(self, req):
        self.req = req
        self.mb = self.req.mb
        self.req.strip_tensors()

    def __repr__(self):
        return f"<PartialBackwardResult::{self.req}>"

    def __hash__(self):
        return hash("PartialBackwardResult") + hash(self.req)


class DummyBackwardResult(ExecutionResult):
    def __init__(
        self,
        mb: int,
        module: str,
        executor: int,
        sender_module: str,
        execution_stack: List[str],
        num_calls: int,
    ):
        self.mb = mb
        self.module = module
        self.executor = executor
        self.sender_module = sender_module
        self.execution_stack = execution_stack
        # number to decrement the recv count on destination
        # this is higher than 1 when multiple backwards triggered from parent to child
        self.num_calls = num_calls

    @classmethod
    def create(cls, mb: int, module: nn.Module, sender_module: nn.Module, num_calls: int):
        # TODO: Verify the correctness of sending the current execution stack
        return cls(
            mb=mb,
            module=state.module_manager.get_module_name(module),
            executor=state.module_manager.get_partition(module),
            sender_module=state.module_manager.get_module_name(sender_module),
            execution_stack=state.module_manager.execution_stack,
            num_calls=num_calls,
        )

    def __repr__(self):
        return f"<DummyBwdRes::mb:{self.mb}, module:{self.module}, sender_module:{self.sender_module}, executor:{self.executor}, num_calls:{self.num_calls}>"

    def __eq__(self, other):
        return (
            self.mb == other.mb
            and self.module == other.module
            and self.executor == other.executor
            and self.sender_module == other.sender_module
            and self.execution_stack == other.execution_stack
            and self.num_calls == other.num_calls
        )

    def __hash__(self):
        return (
            hash("DummyBackwardResult")
            + hash(self.mb)
            + hash(self.module)
            + hash(self.executor)
            + hash(self.sender_module)
            + hash(tuple(self.execution_stack))
            + hash(self.num_calls)
        )


class WaitForBackwardDoneRequest(Request, NamedTuple):
    """
    Sent from worker thread of pp_rank=0 to server. This wait helps give execution
    back to server, and come back and check repeatedly if the backward call is
    complete.
    """

    mb: int

    def __repr__(self):
        return f"<WaitForBwdDone::mb:{self.mb}>"

    def __eq__(self, other):
        return self.mb == other.mb

    def __hash__(self):
        return hash("WaitForBackwardDoneRequest") + hash(self.mb)


class ResumeBackward:
    """
    Sent from server to worker thread waiting to start backward pass based on
    pipelining strategy
    """

    def __init__(self, req):
        self.req: WaitForBackwardRequest = req
        self.mb = req.mb

    def __repr__(self):
        return f"<ResumeBwd::mb:{self.req.mb}>"

    def __hash__(self):
        return hash("ResumeBackward") + hash(self.req)
