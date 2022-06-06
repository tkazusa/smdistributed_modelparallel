# Standard Library
import threading
import traceback
from enum import Enum
from queue import Queue
from typing import Union

# Third Party
import torch
from torch import set_grad_enabled

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import local_rank
from smdistributed.modelparallel.torch.messages import (
    BackwardExecutionResult,
    DummyBackwardResult,
    ExecutionRequest,
    ExecutionResult,
    ForwardExecutionResult,
    ModuleExecutionRequest,
    PartialBackwardResult,
    ResumeBackward,
    SequentialModulesExecutionRequest,
    StepExecutionRequest,
    TraceStepExecutionRequest,
    WaitForBackwardDoneRequest,
    WaitForBackwardRequest,
)
from smdistributed.modelparallel.torch.patches.checkpoint import CheckpointConfig
from smdistributed.modelparallel.torch.patches.tracing import TracingEnd
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import (
    check_requires_grad,
    convert_args_to_device,
    flattened,
    rmsg,
)

logger = get_logger()

SKIP_TRACING_MODEL_SIZE_THRESHOLD_BYTES = 1.2e11  # 120 GB


class WorkerExecStatus(Enum):
    # waiting for an Request
    IDLE = 0
    # pending result from an Request to other ranks or server
    PENDING = 1
    # computing results for the Request
    EXECUTING = 2


class ThreadName(Enum):
    WORKER = 0
    SERVER = 1


class WorkerHolder:
    """
    This holds a worker thread in it, and executes tasks using that thread.
    A semaphore coordinates between the worker holder's execution thread (server's thread)
    and the task executing 'worker' thread.
    Methods of this class which start with thread or _thread are executed by the worker thread.
    Others are called by the server thread.
    There's also a queue through which inputs and outputs are communicated across the threads.
    """

    _id = 0

    def __init__(self):
        self.id = WorkerHolder._id
        WorkerHolder._id += 1
        self.comm_queue = Queue()
        self.status = WorkerExecStatus.IDLE
        self.semaphore = threading.Condition()
        # used to signal to thread that it can proceed
        self.ready_events = {
            ThreadName.WORKER: threading.Event(),
            ThreadName.SERVER: threading.Event(),
        }

        self.thread = threading.Thread(target=self._thread_compute, args=(), daemon=True)
        self.thread.start()

    def reset(self):
        # reset state before each request
        self.req = None
        self.exception = None

    def acquire_and_wait(self, thread: ThreadName, timeout: float = 0.5):
        while not self.ready_events[thread].is_set():
            with self.semaphore:
                self.semaphore.wait(timeout=timeout)

        self.ready_events[thread].clear()

    def acquire_and_notify(self, thread: ThreadName):
        with self.semaphore:
            self.ready_events[thread].set()
            self.semaphore.notify()

    def _check_exception(self):
        if self.exception is not None:
            # printing this even though we raise as currently sometimes exceptions are hidden and
            # process appears hung
            logger.fatal(
                rmsg(
                    f"Hit an exception for {state.current_minibatch()}/{state.microbatch} on thread {state.exec_thread_id}: {self.exception}"
                )
            )
            logger.fatal(rmsg(f"{''.join(traceback.format_tb(self.exception.__traceback__))}"))
            logger.fatal(rmsg(f"Parent exec stack {state.module_manager.execution_stack}"))
            logger.fatal(rmsg(f"Req {self.req}"))
            raise self.exception

    def _check_queue_after_thread_return(self):
        self._check_exception()
        r = self.comm_queue.get()
        if isinstance(r, ExecutionRequest):
            state.exec_server.process_request(r)
            # if its the forward pass, the worker needs to wait on the outputs and cannot pick
            # other workloads to execute.
            if r.phase == MbStatus.FWD:
                self.status = WorkerExecStatus.PENDING
            # if its the backward pass, need to wake the worker after dummy send_backward calls
            # are added to the queue. Will come here only when it has to sent some requests
            # to other rank. Moves the state to PENDING
            elif r.phase == MbStatus.BWD:
                self.status = WorkerExecStatus.PENDING
                self.resume_backward()
        elif isinstance(r, (WaitForBackwardRequest, WaitForBackwardDoneRequest)):
            self.status = WorkerExecStatus.PENDING
            state.exec_server.process_wait_backward(r)
        elif isinstance(r, ExecutionResult):
            state.exec_server.process_result(r)
            self.reset()
        else:
            raise NotImplementedError(f"{req}")

    def execute(self, req: ExecutionRequest):
        """
        Starts execution of a new request.

        Blocks till thread finishes or till it requests another Execution.
        """
        self.reset()
        self.req = req
        # pass the request
        self.comm_queue.put(req)
        state.switching_to_worker(self.id, self.req)
        self._resume_thread_common()

    def resume(self, result: ForwardExecutionResult):
        """
        When the thread was moved to pending after requesting some execution,
        this method resumes the execution in thread by passing result for
        that execution request.

        Blocks till thread finishes or till it requests another Execution.
        """
        # pass the outputs of request
        self.comm_queue.put(result.outputs)
        state.switching_to_worker(self.id, result.request)
        self._resume_thread_common()

    def resume_backward(self):
        """
        When the thread was moved to pending after requesting some execution,
        this method resumes the execution in thread by passing result for
        that execution request.

        Blocks till thread finishes or till it requests another Execution.
        """
        state.switching_to_worker(self.id, self.req)
        # todo: maybe set correct backward request here
        self._resume_thread_common()

    def _resume_thread_common(self):
        self.status = WorkerExecStatus.EXECUTING
        self.acquire_and_notify(ThreadName.WORKER)
        self.acquire_and_wait(ThreadName.SERVER)
        self._check_queue_after_thread_return()

    def thread_get_forward_result(self, request: ExecutionRequest):
        """
        Functions prefixed by _thread are called by self.thread.

        Called by thread when waiting for an execution result by different rank.
        The thread loses execution at this point, and expects
        to see the output when it is resumed.
        """
        # we need to keep track of
        grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

        # below line is needed to ensure we reset state of config back to what it was when request was made
        checkpoint_activations_config = state.checkpoint_activations_config
        state.checkpoint_activations_config = CheckpointConfig()

        self.comm_queue.put(request)
        self.acquire_and_notify(ThreadName.SERVER)
        self.acquire_and_wait(ThreadName.WORKER)
        req_outputs = self.comm_queue.get()
        torch.set_grad_enabled(grad_enabled)
        torch.set_autocast_enabled(autocast_enabled)
        state.checkpoint_activations_config = checkpoint_activations_config
        return req_outputs

    def thread_get_backward_result(self, request: ModuleExecutionRequest):
        """
        Functions prefixed by thread_ are called by self.thread.

        Called by the thread (ExecutionWorker) when it wants to trigger a backward
        pass on a different rank. Unlike forward pass, the worker thread doesn't need
        to wait for outputs to return from this method.
        """
        grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()
        self.comm_queue.put(request)
        self.acquire_and_notify(ThreadName.SERVER)
        self.acquire_and_wait(ThreadName.WORKER)
        torch.set_grad_enabled(grad_enabled)
        torch.set_autocast_enabled(autocast_enabled)
        return

    def thread_wait_for_backward_start(self):
        """
        Functions prefixed by (_)?thread are called by self.thread.

        Called by thread when waiting for a signal to start backward pass.
        The thread loses execution at this point, and expects
        to see the signal when it is resumed.
        """
        grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()
        self.comm_queue.put(WaitForBackwardRequest(self.req.mb))
        self.acquire_and_notify(ThreadName.SERVER)
        self.acquire_and_wait(ThreadName.WORKER)
        torch.set_grad_enabled(grad_enabled)
        torch.set_autocast_enabled(autocast_enabled)

    def thread_wait_for_backward_done(self):
        self.comm_queue.put(WaitForBackwardDoneRequest(self.req.mb))
        self.acquire_and_notify(ThreadName.SERVER)
        self.acquire_and_wait(ThreadName.WORKER)

    def _exec_trace_on_device(
        self, req: Union[TraceStepExecutionRequest, StepExecutionRequest], device: torch.device
    ):
        """
        Helper function for tracing
        """
        args, kwargs = convert_args_to_device(req.args, req.kwargs, device)
        step_fn = state.step_func[req.step_fn_id].func

        if (
            state.cfg.auto_partition
            and not state.cfg.skip_tracing
            and state.model.size() <= SKIP_TRACING_MODEL_SIZE_THRESHOLD_BYTES
        ):
            for trial in range(5):
                # some runs to warmup, times at the beginning may not be representative of actual time
                try:
                    with state.module_manager.enable_measurement(trial == 4):
                        outputs = step_fn(*args, **kwargs)
                except TracingEnd:
                    # this exception lets us bypass any logic in step function
                    # that is not forward of model.
                    # we dont want to execute loss module if it exists and backward
                    # as the loss module won't be patched
                    outputs = None
        else:
            outputs = None
        state.module_manager.compute_execution_times()
        self.comm_queue.put(ForwardExecutionResult(self.req, outputs))

    def _exec_step_on_device(self, req: Union[StepExecutionRequest], device: torch.device):
        """
        Helper function for step execution
        """
        args, kwargs = convert_args_to_device(req.args, req.kwargs, device)
        step_fn = state.step_func[req.step_fn_id].func
        outputs = step_fn(*args, **kwargs)
        self.comm_queue.put(ForwardExecutionResult(self.req, outputs))

    def thread_execute_tracing(self, req: TraceStepExecutionRequest):
        """
        Functions prefixed by thread_ are called by self.thread.
        Traces the execution path for the model for a single step.
        """
        if state.model.trace_device == "cpu":
            device = torch.device("cpu")
            logger.info(
                "Tracing on CPU. For models whose parameters fit in a single device, setting trace_device to `gpu` is recommended for better partitioning decisions."
            )
        else:
            logger.info(
                "Tracing on GPU. If the model parameters do not fit in a single GPU, you can set trace_device to `cpu`."
            )
            device = torch.device("cuda", local_rank())

        # maintain the same rng state after tracing so rank 0 is consistent with other ranks
        with state.fork_smp_rng_state():
            with state.fork_torch_rng_state(device):
                with state.patch_manager.patch_for_trace(state.model, device):
                    self._exec_trace_on_device(req, device)

    def thread_execute_step(self, req: StepExecutionRequest):
        """
        Functions prefixed by thread_ are called by self.thread.
        Runs a single step and puts the result in the queue
        for the server.
        """
        device = torch.device("cuda", local_rank())
        self._exec_step_on_device(req, device)

    def thread_execute_forward(
        self, req: Union[ModuleExecutionRequest, SequentialModulesExecutionRequest]
    ):
        """
        Functions prefixed by thread_ are called by self.thread.
        Runs forward for a module and sets dummy result flags to
        send back if required.
        """
        mod = state.module_manager.get_module(req.module)
        if state.cfg.fast_mode and state.current_minibatch() == 0 and state.microbatch == 0:
            # mark module info
            step_func = state.current_step_func()
            assert step_func is not None

            with flattened((req.args, req.kwargs)) as flat_args:
                # if requester is already the rank that generated the tensor, exclude it from
                # direct consumer map
                if isinstance(req, SequentialModulesExecutionRequest):
                    start_index = req.start_layer_idx
                else:
                    start_index = None
                step_func.update_direct_fwd_consumers(
                    flat_args,
                    module=mod,
                    exclude_partitions=[req.requester],
                    start_index=start_index,
                )

        if isinstance(req, ModuleExecutionRequest):
            with torch.cuda.amp.autocast(req.enable_autocast):
                with state.enable_activation_checkpoints(req.checkpoint_activations_config):
                    outputs = mod.forward(*req.args, **req.kwargs)
            if (
                not state.module_manager.is_main_module(mod)
                and not check_requires_grad((req.args, req.kwargs))
                and check_requires_grad(outputs)
            ):
                parent_mod = state.module_manager.get_parent_module(mod)
                state.module_manager.set_dummy_bwd_send(req.mb, req.position, mod, parent_mod, 1)

        elif isinstance(req, SequentialModulesExecutionRequest):
            with torch.cuda.amp.autocast(req.enable_autocast):
                with state.enable_activation_checkpoints(req.checkpoint_activations_config):
                    outputs, send_real_bwd, num_inputs = mod.execute_chain(
                        req.start_layer_idx, req.end_layer_idx, req.args
                    )
            mod = mod[req.end_layer_idx - 1]
            if (
                not state.module_manager.is_main_module(mod)
                and not send_real_bwd
                and check_requires_grad(outputs)
            ):
                parent_mod = state.module_manager.get_parent_module(mod)
                state.module_manager.set_dummy_bwd_send(
                    req.mb, req.position, mod, parent_mod, num_inputs
                )

        else:
            raise NotImplementedError
        self.comm_queue.put(ForwardExecutionResult(req, outputs))

    def _bwd_aggregated_execute(self, req, mod, parent_mod):
        sequential = isinstance(parent_mod, torch.nn.Sequential)
        count = state.module_manager._get_pending_backward(
            req.mb, mod, parent_mod, sequential=sequential
        )
        if (
            state.module_manager.is_executor(parent_mod)
            and count > 0
            and not (state.module_manager.is_main_module(parent_mod) and parent_mod == mod)
        ):
            self.comm_queue.put(PartialBackwardResult(req))
            return
        all_outputs, all_grads = state.module_manager.dequeue_all_bwd_tensors(
            req.mb, mod, parent_mod
        )
        if state.cfg.fast_mode:
            state.serialization_manager.prepare_dummy_tensors_for_backward(all_outputs)

        # retain graph in memory after backward only if there are parameters in this
        # partition that are expecting more backward calls later
        retain = False
        for p in state.model.local_parameters():
            name = state.model.get_param_name(p)
            expected_count = state.model.grad_counter.get_param_grad_count(state.microbatch, name)
            seen_count = state.model.grad_counter.get_seen_grad_count(state.microbatch, name)
            if expected_count - seen_count > 1:
                retain = True
                break

        state.model.grad_counter.set_microbatch(state.microbatch)
        torch.autograd.backward(all_outputs, all_grads, retain_graph=retain)
        state.model.grad_counter.set_microbatch(-1)
        self.comm_queue.put(BackwardExecutionResult(req))

    def thread_execute_backward(self, req: ModuleExecutionRequest):
        """
        Functions prefixed by thread_ are called by self.thread.
        Resumes backward request for a module given the position
        of saved output and output grads in the request.
        """
        # set this so we can keep track of whether this step had backward pass
        # on all mp ranks, used to determine whether/when to synchronize reducer
        state.model._mark_backward_in_step()

        mod = state.module_manager.get_module(req.module)
        parent_mod = state.module_manager.get_parent_module(mod)
        sender_mod = state.module_manager.get_module(req.sender_module)
        sender_is_parent = False
        if state.module_manager.is_main_module(mod) and mod == sender_mod:
            parent_mod = mod
            mod = mod
        elif state.module_manager.is_main_module(mod) or (
            not req.sender_module == state.module_manager.get_module_name(parent_mod)
        ):
            # then sender is a child of current module
            # so the parent in this context becomes current module
            parent_mod = mod
            mod = sender_mod
            sequential = isinstance(parent_mod, torch.nn.Sequential)
            state.module_manager.decrement_bwd_count(
                req.mb, mod, parent_mod, 1, real=True, sequential=sequential
            )
        else:
            assert parent_mod == sender_mod
            sender_is_parent = True
            state.module_manager.load_smpinput_bwd_count(req.mb, req.position, mod, parent_mod)

        output_grads = req.args
        state.position = req.position
        outputs = state.module_manager.get_output(req.mb, mod, parent_mod, req.position)
        if req.tensor_idx >= 0 and not isinstance(outputs, torch.Tensor):
            outputs = (outputs[req.tensor_idx],)
        required_outputs = []
        required_grads = []
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            assert len(outputs) == len(output_grads), "output grads and outputs length should match"
            for output, output_grad in zip(outputs, output_grads):
                if output.requires_grad:
                    required_outputs.append(output)
                    required_grads.append(output_grad)
        else:
            required_outputs = outputs
            required_grads = output_grads

        if (
            state.cfg.fast_mode
            and state.current_minibatch() == 0
            and sender_is_parent
            and state.microbatch == 0
        ):
            step_func = state.current_step_func()
            assert step_func is not None
            step_func.update_direct_bwd_consumers(required_grads, mod)

        state.module_manager.enqueue_bwd_tensors(
            req.mb, mod, parent_mod, (required_outputs, required_grads)
        )
        self._bwd_aggregated_execute(req, mod, parent_mod)

    def _thread_compute(self):
        """
        Main thread execution loop. Waits for request in queue and executes it.
        Return to execution in coordinator thread can be from either here or the above wait_for_exec_result function

        thread should put a response on the comm queue after its done processing a request

        """
        while True:
            # blocks on below till main thread notifies
            self.acquire_and_wait(ThreadName.WORKER)
            try:
                with torch.cuda.stream(state.stream):
                    req = self.comm_queue.get()
                    # Depending on whether we want worker to require grads
                    # based on the context in which the step function is running on mprank0
                    set_grad_enabled(req.enable_grads)
                    if isinstance(req, TraceStepExecutionRequest):
                        self.thread_execute_tracing(req)
                    elif isinstance(req, StepExecutionRequest):
                        self.thread_execute_step(req)
                    elif (
                        isinstance(req, (ModuleExecutionRequest, SequentialModulesExecutionRequest))
                        and req.phase == MbStatus.FWD
                    ):
                        self.thread_execute_forward(req)
                    elif isinstance(req, ModuleExecutionRequest) and req.phase == MbStatus.BWD:
                        self.thread_execute_backward(req)
                    req = None
            except Exception as e:
                self.exception = e
            self.status = WorkerExecStatus.IDLE
            self.acquire_and_notify(ThreadName.SERVER)
