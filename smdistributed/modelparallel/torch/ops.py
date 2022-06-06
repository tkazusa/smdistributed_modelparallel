# Standard Library
import time
from enum import Enum
from typing import Optional, Set, Tuple

# Third Party
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.messages import ModuleExecutionRequest
from smdistributed.modelparallel.torch.module_manager import TensorModuleInfo
from smdistributed.modelparallel.torch.state_mod import state

ContextMixin = torch.autograd.function._ContextMethodMixin

logger = get_logger()


class CommDirection(Enum):
    SEND = 0
    RECV = 1


def get_fn(type: str):
    fn_name = f"smp_torch_{type}"
    if hasattr(smplib, fn_name):
        return getattr(smplib, fn_name)
    else:
        raise NotImplementedError("Queried function does not exist in smplib")


def send(
    tensor: torch.Tensor,
    rank: int,
    link_id: int,
    name: Optional[str] = None,
    needs_meta_transmission: bool = True,
    server: bool = True,
    release_after_send: bool = True,
) -> int:
    """ If the sender will wait for the transmission, must set release_after_send to False, and
    then use wait_and_clear. Otherwise, set release_after_send=True, and use synchronize. """

    if name is None:
        name = "_SMPSend"
    # dummy tensor
    output = tensor.new_empty((1,), dtype=tensor.dtype, device=tensor.device)
    handle = get_fn("send")(
        tensor, rank, link_id, name, needs_meta_transmission, server, release_after_send
    )
    state.handle_manager.register_handle(handle, CommDirection.SEND, output)
    return handle


def recv(
    tensor: torch.Tensor,
    rank: int,
    link_id: int,
    name: Optional[str] = None,
    metadata: Optional[smplib.TorchTensorMeta] = None,
    server: bool = True,
) -> int:
    """
    If needs_meta_transmission is False, backend uses the dtype, ndims of given tensor
    Else it first receives this meta info from sender, allocates recv tensor
    """
    if name is None:
        name = "_SMPRecv"
    if metadata == None:
        handle = get_fn("recv")(tensor, rank, link_id, name, server)
    else:
        assert isinstance(
            metadata, smplib.TorchTensorMeta
        ), f"metadata must be TorchTensorMeta type, but got {type(metadata)}"
        handle = get_fn("recv")(tensor, rank, link_id, name, metadata, server)
    # actual recv tensor can change if dtype was not sent
    # we will need to create a new tensor then
    state.handle_manager.register_handle(handle, CommDirection.RECV, None)
    return handle


def synchronize(handle: int) -> torch.Tensor:
    if not state.handle_manager.is_valid_handle(handle):
        return
    if state.handle_manager.get_handle_direction(handle) == CommDirection.RECV:
        wait(handle)
        output = get_fn("get_ready_tensor")(handle)
    else:
        # send has immediate output after we enqueue tensor to backend
        output = state.handle_manager.get_handle_output(handle)
    state.handle_manager.clear_handle(handle)
    return output


def wait_and_clear(handle: int):
    if not state.handle_manager.is_valid_handle(handle):
        return
    wait(handle)
    get_fn("clear")(handle)
    state.handle_manager.clear_handle(handle)


def poll(handle: int) -> int:
    if not state.handle_manager.is_valid_handle(handle):
        return
    return get_fn("poll")(handle)


def wait(handle: int):
    """Wait on python side so that the gil could be released"""
    while poll(handle) == 0:
        time.sleep(0.000001)  # sleep for 1 microseconds same as c++


def increment_prev_parent_recv_counts(prev_parent_recvs):
    for pr in prev_parent_recvs:
        pr.next_parent_recvs_bwd_pending += 1


def aggregate_grads(
    prev_out_grads: Tuple[Optional[torch.Tensor]], curr_out_grads: Tuple[Optional[torch.Tensor]]
) -> Tuple[Optional[torch.Tensor]]:
    if prev_out_grads is None:
        return curr_out_grads
    else:
        total_grads = []
        for prev, curr in zip(prev_out_grads, curr_out_grads):
            if prev is None and curr is None:
                total = None
            elif prev is None:
                total = curr
            elif curr is None:
                total = prev
            else:
                # we do not preserve _smp_module_info attributes of the original tensors here, because
                # if aggregation is required then parent needs to do it, so we do not do child-to-child
                # transmissions
                total = torch.add(curr, prev)
            total_grads.append(total)
        return tuple(total_grads)


class SMPSequentialInput(torch.autograd.Function):
    """Similar to SMPInput op but takes parent_module as an input so we can control what is set as parent
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx: ContextMixin,
        module: nn.Module,
        parent_module: nn.Module,
        idx: int,
        *args: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Forward is a pass through which returns args
        """
        ctx.module = module
        ctx.is_parent_executor = state.module_manager.is_executor(parent_module)
        ctx.is_main = state.module_manager.is_main_module(ctx.module)
        ctx.idx = idx
        ctx.pending_smpinput_bwds = 0
        ctx.out_grads = None
        if not ctx.is_main and not ctx.is_parent_executor:
            # Saving all operations related to module execution stack in forward
            # since these calls only work within context of forward, and will return
            # incorrect results inside the backward call
            ctx.parent_module = parent_module
            ctx.execution_stack = state.module_manager.execution_stack
            ctx.position = state.module_manager.output_stack_size(
                state.microbatch, ctx.module, ctx.parent_module
            )
        return args

    @staticmethod
    @custom_bwd
    def backward(ctx: ContextMixin, *out_grads: torch.Tensor) -> Tuple[torch.Tensor]:
        """Creates the backward execution request if the current module is not main,
        and is not the executor for the current module.
        Otherwise, allows backward pass to continue.
        """
        ctx.pending_smpinput_bwds -= 1
        assert ctx.pending_smpinput_bwds >= 0, "Pending smpinput bwd count is less than 0"
        ctx.out_grads = aggregate_grads(ctx.out_grads, out_grads)
        if not ctx.is_main and not ctx.is_parent_executor and ctx.pending_smpinput_bwds == 0:
            if state.cfg.fast_mode:
                # mark out_grads with ModuleInfo
                for i, grad in enumerate(ctx.out_grads):
                    module_name = state.module_manager.get_module_name(ctx.module)
                    count = state.current_step_func().get_bwd_module_execution_count(
                        ctx.module, state.microbatch
                    )
                    state.current_step_func().increment_bwd_module_execution_count(
                        ctx.module, state.microbatch
                    )
                    grad._smp_module_info = TensorModuleInfo(module_name, count, i, False, None)

            # not inside main and current rank not the executor,
            # need to dispatch the request to another module
            request = ModuleExecutionRequest.create_backward_req(
                ctx.parent_module,
                ctx.out_grads,
                position=ctx.position,
                sender_module=ctx.module,
                execution_stack=ctx.execution_stack,
                tensor_idx=ctx.idx,
            )
            state.current_worker.thread_get_backward_result(request)
            ctx.out_grads = None
        out_grads = (None, None, None) + out_grads
        return out_grads


class SMPInput(torch.autograd.Function):
    """Records the current_module in the forward pass.
    In the backward pass, sends a backward pass request to the module if
    current module is not the main module (root of the tree).
    If current module is the main, it returns the out_grads
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx: ContextMixin, module: nn.Module, idx: int, *args: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Forward is a pass through which returns args
        """
        # This tracks the pending backward calls for SMPInput
        # This count is set in the forward pass, by doing backward traversal
        # to the SMPInput node and decremented when backward is called.
        # When this count reaches 0, a backward request is sent
        # for the parent_module
        ctx.pending_smpinput_bwds = 0
        ctx.module = module
        ctx.is_parent_executor = state.module_manager.is_parent_executor(ctx.module)
        ctx.is_main = state.module_manager.is_main_module(ctx.module)
        ctx.idx = idx
        ctx.out_grads = None
        if not ctx.is_main and not ctx.is_parent_executor:
            # Saving all operations related to module execution stack in forward
            # since these calls only work within context of forward, and will return
            # incorrect results inside the backward call
            ctx.parent_module = state.module_manager.get_parent_module(ctx.module)
            ctx.execution_stack = state.module_manager.execution_stack
            ctx.position = state.module_manager.output_stack_size(
                state.microbatch, ctx.module, ctx.parent_module
            )
        return args

    @staticmethod
    @custom_bwd
    def backward(ctx: ContextMixin, *out_grads: torch.Tensor) -> Tuple[torch.Tensor]:
        """Creates the backward execution request if the current module is not main,
        and is not the executor for the current module.
        Otherwise, allows backward pass to continue.
        """
        ctx.pending_smpinput_bwds -= 1
        assert ctx.pending_smpinput_bwds >= 0, "Pending smpinput bwd count is less than 0"
        ctx.out_grads = aggregate_grads(ctx.out_grads, out_grads)
        if not ctx.is_main and not ctx.is_parent_executor and ctx.pending_smpinput_bwds == 0:
            if state.cfg.fast_mode:
                # mark out_grads with ModuleInfo
                for i, grad in enumerate(ctx.out_grads):
                    module_name = state.module_manager.get_module_name(ctx.module)
                    count = state.current_step_func().get_bwd_module_execution_count(
                        ctx.module, state.microbatch
                    )
                    state.current_step_func().increment_bwd_module_execution_count(
                        ctx.module, state.microbatch
                    )
                    grad._smp_module_info = TensorModuleInfo(module_name, count, i, False, None)

            # not inside main and current rank not the executor,
            # need to dispatch the request to another module
            request = ModuleExecutionRequest.create_backward_req(
                ctx.parent_module,
                ctx.out_grads,
                position=ctx.position,
                sender_module=ctx.module,
                execution_stack=ctx.execution_stack,
                tensor_idx=ctx.idx,
            )
            state.current_worker.thread_get_backward_result(request)
            ctx.out_grads = None
        out_grads = (None, None) + out_grads
        return out_grads


class SMPParentRecv(torch.autograd.Function):
    """Identity function which creates a copy of inputs
    in forward. This autograd function enqueues a backward
    request (to be sent to a different rank) in the backward
    pass.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx: ContextMixin,
        parent_module: nn.Module,
        module: nn.Module,
        count: int,
        prev_parent_recvs: Set,
        sequential: bool,
        *args: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Forward saves the module and creates a copy
        of inputs, returns it"""
        ctx.module = module
        # Saving all operations related to module execution stack in forward
        # since these calls only work within context of forward, and will return
        # incorrect results inside the backward call
        ctx.parent_module = parent_module
        ctx.execution_stack = state.module_manager.execution_stack
        ctx.bwd_req_count = count
        assert (
            state.module_manager.output_stack_size(state.microbatch, ctx.module, ctx.parent_module)
            > 0
        ), "output_stack_size should be greater than 0, this is a bug"
        ctx.position = (
            state.module_manager.output_stack_size(state.microbatch, ctx.module, ctx.parent_module)
            - 1
        )

        # below helps keep track of previous parent recvs that inputs to this parent recv depended on
        # so we can aggregate backward requests from parent to child
        # that also helps us calculate number of grads for a param
        ctx.prev_parent_recvs = prev_parent_recvs
        ctx.next_parent_recvs_bwd_pending = 0
        increment_prev_parent_recv_counts(ctx.prev_parent_recvs)
        ctx.out_grads = None
        ctx.sequential = sequential
        outputs = args
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: ContextMixin, *out_grads: torch.Tensor) -> Tuple[torch.Tensor]:
        """Backward enqueues a request to be sent to a different rank
        and returns
        """
        ctx.next_parent_recvs_bwd_pending -= 1
        assert (
            ctx.next_parent_recvs_bwd_pending >= 0
        ), "Pending parent recvs for this backward can not be less than 0"

        ctx.out_grads = aggregate_grads(ctx.out_grads, out_grads)

        if ctx.next_parent_recvs_bwd_pending == 0:
            request = ModuleExecutionRequest.create_backward_req(
                ctx.module,
                ctx.out_grads,
                position=ctx.position,
                sender_module=ctx.parent_module,
                execution_stack=ctx.execution_stack,
            )
            state.current_worker.thread_get_backward_result(request)
            # count needs to be atleast one, since a dummy backward is guaranteed to be sent
            if ctx.bwd_req_count == 0:
                state.module_manager.increment_bwd_count(
                    state.microbatch,
                    ctx.module,
                    ctx.parent_module,
                    real=False,
                    sequential=ctx.sequential,
                )
            else:
                for _ in range(ctx.bwd_req_count):
                    state.module_manager.increment_bwd_count(
                        state.microbatch, ctx.module, ctx.parent_module, sequential=ctx.sequential
                    )
            ctx.out_grads = None
        return (None, None, None, None, None) + out_grads
