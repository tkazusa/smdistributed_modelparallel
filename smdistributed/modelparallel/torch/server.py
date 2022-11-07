# Standard Library
import time
from typing import Any, Dict, List, Tuple, Union

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import pp_rank, rank
from smdistributed.modelparallel.torch.exceptions import InvalidRequestError, SMPRuntimeError
from smdistributed.modelparallel.torch.messages import (
    BackwardExecutionResult,
    DummyBackwardResult,
    ExecutionRequest,
    ExecutionResult,
    ForwardExecutionResult,
    MicrobatchEndResult,
    ModuleExecutionRequest,
    PartialBackwardResult,
    Request,
    ResumeBackward,
    SequentialModulesExecutionRequest,
    StepEndNotification,
    StepExecutionRequest,
    TraceStepExecutionRequest,
    WaitForBackwardDoneRequest,
    WaitForBackwardRequest,
)
from smdistributed.modelparallel.torch.module_manager import PartitioningAndTraceResults
from smdistributed.modelparallel.torch.patches.execution import detach_outputs
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.server_comm import ServerCommunicator
from smdistributed.modelparallel.torch.server_queue import (
    DeterministicServerQueue,
    IncomingMessageTask,
    NoTask,
    OpportunisticServerQueue,
)
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import rmsg
from smdistributed.modelparallel.torch.worker import WorkerExecStatus, WorkerHolder


class ExecutionServer:
    def __init__(self):
        self._init_step()

        self._current_task = None
        self._workers: Dict[int, WorkerHolder] = {}

        # from request to worker_id
        self._pending_requests_to_workers: Dict[ModuleExecutionRequest, int] = {}

        self.comm = ServerCommunicator()

        if (
            state.cfg.tensor_parallel_degree > 1
            or state.cfg.static_mode
            or state.cfg.fast_mode
            or state.cfg.offload_activations
        ):
            self.server_queue = DeterministicServerQueue(self.comm, state.pipeline)
        else:
            self.server_queue = OpportunisticServerQueue(self.comm, state.pipeline)

        self.logger = get_logger()
        self.step = 0

    def _init_step(self):
        # final outputs of the step function indexed by microbatch
        self.outputs: Dict[int, Any] = {}

    @property
    def current_task(self):
        return self._current_task

    @current_task.setter
    def current_task(self, task):
        self._current_task = task

    def get_worker(self, worker_id: int) -> WorkerHolder:
        return self._workers[worker_id]

    def pop_pending_request(self, req: ModuleExecutionRequest) -> int:
        try:
            self.logger.debug(rmsg(f"Popping pending request {req}"))
            worker_id = self._pending_requests_to_workers[req]
            del self._pending_requests_to_workers[req]
            return worker_id
        except KeyError:
            raise InvalidRequestError(req)

    def record_pending_request(self, req: ModuleExecutionRequest, worker_id: int = None):
        if worker_id is None:
            worker_id = state.exec_thread_id
        self._pending_requests_to_workers[req] = worker_id
        self.logger.debug(rmsg(f"Recording pending request {req}"))

    def execute_request(self, req: ExecutionRequest):
        """
        This is a blocking call to be used for local requests.
        It blocks till worker thread goes to pending state or finishes.
        """
        # use any idle worker
        self.logger.debug(rmsg(f"Executing {req}"))
        chosen_worker = None
        for i, w in self._workers.items():
            if w.status == WorkerExecStatus.IDLE:
                chosen_worker = w
                break

        if chosen_worker is None:
            # no idle worker found, create new one
            new_worker = WorkerHolder()
            self._workers[new_worker.id] = new_worker
            chosen_worker = new_worker

        chosen_worker.execute(req)

    def _get_pending_wait_for_req(self, mb, req_type):
        # hacky, todo clean this up
        matched_reqs = [
            x
            for x in self._pending_requests_to_workers.keys()
            if isinstance(x, req_type) and x.mb == mb
        ]
        if len(matched_reqs) != 1:
            raise SMPRuntimeError("matched_reqs must be unique")
        return matched_reqs[0]

    def process_wait_backward(self, req: Union[WaitForBackwardRequest, WaitForBackwardDoneRequest]):
        """
        Worker thread has finished forward pass, and wants to execute backward pass.
        We want to block that though to ensure that execution matches our pipelining strategy.
        This function is called by the worker thread when it is about to start backward pass.
        """
        # let pipeline know mb is ready for backward
        if isinstance(req, WaitForBackwardRequest):
            state.pipeline.mark_ready_for_backward(req.mb)
        elif isinstance(req, WaitForBackwardDoneRequest):
            pass
        self.record_pending_request(req)

    def process_request(self, req: ExecutionRequest):
        """
        Process the execution request.
        Communicate args, kwargs for the request.
        If it's to be executed by current process, route it to appropriate worker thread.
            Picks some idle worker thread. If none is available creates a new thread.
        Else, send to other process.
        """
        if req.executor == pp_rank():
            self.execute_request(req)
        else:
            self.logger.debug(rmsg(f"Sending {req}"))
            self.comm.send(req, req.executor)

            if req.phase == MbStatus.FWD:
                self.record_pending_request(req)

    def handle_backwards_bookkeeping(
        self, res: Union[BackwardExecutionResult, DummyBackwardResult]
    ):
        """
        BackwardExecutionResult is only processed on same process, it is never sent over to other processes.
        As the actual backward result communication will be done through autograd ops by torch.

        Here we do send DummyBackwardResult sometimes to let receiver know that children are all done with backward pass.
        """
        self.logger.debug(rmsg(f"Processing {res}"))
        if isinstance(res, BackwardExecutionResult):
            rr = res.req
            if rr.sender_module == rr.module:
                # Only do backwards bookkeeping if the request is not backward
                # on the main module. For main module, since there is no parent
                # there is no need for dummy tensors to be sent
                return
        else:
            rr = res
        state.module_manager.execution_stack = rr.execution_stack

        mod = state.module_manager.get_module(rr.module)
        mb = rr.mb
        sender_module = rr.sender_module
        executor = rr.executor
        parent_mod = state.module_manager.get_parent_module(mod)
        sender_is_parent = (
            False
            if state.module_manager.is_main_module(mod)
            else rr.sender_module == state.module_manager.get_module_name(parent_mod)
        )

        if not sender_is_parent:
            # then sender is a child of current module
            # so the parent in this context becomes current module
            parent_mod = mod
            mod = state.module_manager.get_module(sender_module)
            # backward request came from a child
            # were expecting a certain number of recvs,
            # we got one request, finished it, so reduce the number of recvs expected
            if isinstance(rr, DummyBackwardResult):
                sequential = isinstance(parent_mod, torch.nn.Sequential)
                state.module_manager.decrement_bwd_count(
                    mb, mod, parent_mod, rr.num_calls, real=False, sequential=sequential
                )
        else:
            position = rr.position
            if state.module_manager.get_dummy_bwd_send(mb, position, mod, parent_mod) > 0:
                count = state.module_manager.get_dummy_bwd_send(mb, position, mod, parent_mod)
                state.module_manager.increment_dummy_bwd_sends(mb, mod, parent_mod, count)

        # if recv is 0 for current mod for all children of the mod, then
        # find ancestors on boundary, send the dummy grads and reduce send
        # count
        current_mod = mod if sender_is_parent else parent_mod
        if state.module_manager.check_no_pending_bwd(mb, current_mod):
            if state.module_manager.is_main_module(
                current_mod
            ) and state.module_manager.is_executor(current_mod):
                if pp_rank() != 0:
                    raise SMPRuntimeError(
                        f"main module and executor should have pp rank 0, but get {pp_rank()}"
                    )
                req = self._get_pending_wait_for_req(mb, WaitForBackwardDoneRequest)
                wid = self.pop_pending_request(req)
                self._workers[wid].resume_backward()
                # resume step worker so it continue with what's after backward as it is now done
            else:
                ancestor_parent, ancestor = state.module_manager.find_boundary_ancestors(
                    mb, current_mod
                )
                if state.module_manager.check_no_pending_bwd(mb, ancestor):
                    if state.module_manager.is_main_module(
                        ancestor
                    ) and state.module_manager.is_executor(ancestor):
                        # If ancestor on boundary is main, backward is done, resume main thread
                        req = self._get_pending_wait_for_req(mb, WaitForBackwardDoneRequest)
                        wid = self.pop_pending_request(req)
                        self._workers[wid].resume_backward()
                    else:
                        send_count = state.module_manager.num_dummy_sends(
                            mb, ancestor, ancestor_parent
                        )
                        if send_count > 0:
                            dummy = DummyBackwardResult.create(
                                mb, ancestor_parent, ancestor, send_count
                            )
                            self.comm.send(dummy, dummy.executor)
                            state.module_manager.decrement_dummy_bwd_sends(
                                mb, ancestor, ancestor_parent, send_count
                            )

    def _assign_and_partition(self):
        # this function is now only called after tracing, i.e. during auto partition
        # if pipeline parallel degree is 1, the work here is done during smp.DistributedModel call
        if state.cfg.auto_partition:
            state.module_manager.auto_partition_model(state.model, root_rank=0)
        # if manual it would already have assigned partitions by this point

        # Send to all ranks even those outside of own pp_group
        # as only rank==0 does tracing

        self.comm.send(
            state.module_manager.get_serialized_partitioning_and_tracing_states(), group=False
        )
        state.model.post_partition()
        self.server_queue.set_partitioned()

    def process_result(self, res: ExecutionResult):
        """
        Process the execution result.
        If the requester was own rank, resume the worker waiting on it.
        Else, send to the requester rank.
        """
        if isinstance(res, (DummyBackwardResult, BackwardExecutionResult)):
            self.handle_backwards_bookkeeping(res)
        elif isinstance(res, (PartialBackwardResult)):
            pass
        elif isinstance(res, ForwardExecutionResult) and res.request.requester == pp_rank():
            self.logger.debug(rmsg(f"Processing {res}"))
            if isinstance(res.request, TraceStepExecutionRequest):
                # rank0
                self._assign_and_partition()
            elif isinstance(res.request, StepExecutionRequest):
                # end of mb execution
                if state.current_step_func().detach_outputs is True:
                    # detaching here instead of at end of step allows us to detach at end of microbatch and clear unnecessary memory
                    res.outputs = detach_outputs(res.outputs)
                self.outputs[res.request.mb] = res.outputs
                mbend_req = MicrobatchEndResult(res.request.mb, res.outputs)

                state.module_manager.clear_microbatch_state(res.request.mb)
                self.comm.clear_microbatch_state(res.request.mb)
                self.server_queue.decrement_in_flight_mbs()

                self.logger.debug(rmsg(f"Sending {mbend_req}"))
                self.comm.send(mbend_req)
                if pp_rank() == 0:
                    state.pipeline.mark_done(res.request.mb)
            elif isinstance(
                res.request, (ModuleExecutionRequest, SequentialModulesExecutionRequest)
            ):
                # resume the worker waiting on this request
                worker_id = self.pop_pending_request(res.request)
                self.resume_worker(worker_id, res)
        else:
            self.logger.debug(rmsg(f"Sending {res}"))
            self.comm.send(res, res.request.requester)

    def resume_worker(self, worker_id: int, res: Union[ForwardExecutionResult, ResumeBackward]):
        """
        Blocks till worker pauses or finishes
        """
        self.logger.debug(rmsg(f"Resuming worker with {res}"))
        if isinstance(res, ForwardExecutionResult):
            self._workers[worker_id].resume(res)
        elif isinstance(res, ResumeBackward):
            self._workers[worker_id].resume_backward()
        else:
            raise SMPUnsupportedError

    def run_step_leader(
        self, mb_args: List[Tuple[Any]], mb_kwargs: List[Dict[str, Any]], step_func_id: int
    ):
        """
        Leader of a step is the process with pp_rank == 0 in each pp_group.
        It does a few things in addition to what a follower does.
        - Identifies the next microbatch to be
        executed according to pipelining strategy when there is no request in
        queue, and picks that up.
        - Enqueues execution requests for step and traceStep. Other processes only
        handle module execution requests.
        - Sends StepEndNotification to all ranks
        - Sends MicrobatchEndResult with the outputs of the function in step decorator.

        Note that messages from a worker in same process don't go to comm's queue, so they
        don't show up in this loop. Those would directly go to process_result or process_request.
        """
        # If manually partition or load partition file, do post-partition to assign/bcast params
        if state.model._partitions_assigned and not state.model.partitioned:
            state.model.post_partition()
            self.server_queue.set_partitioned()

        if rank() == 0 and not state.model._partitions_assigned and not state.model.partitioned:
            self._init_step()
            # TODO: do we need to trace for both train and eval? currently we dont
            self.execute_request(
                TraceStepExecutionRequest(
                    mb_args[0], mb_kwargs[0], 0, step_func_id, enable_grads=False
                )
            )

        self.in_flight_mbs = 0
        # clear outputs for each minibatch
        self._init_step()
        while self.server_queue.has_more_tasks():
            time.sleep(0.0001)
            task = self.server_queue.get_next_task()
            self.current_task = task
            if isinstance(task, NoTask):
                continue
            elif isinstance(task, IncomingMessageTask):
                r = task.get_message()
                if isinstance(r, ExecutionRequest):
                    state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                    if not state.has_uploaded_metrics and r.mb == 0:
                        state.num_hops += 1
                    self.process_request(r)
                    state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                elif isinstance(r, ExecutionResult):
                    state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                    self.process_result(r)
                    state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                elif isinstance(r, PartitioningAndTraceResults):
                    # on ranks with pp_ranks=0 but rank!=0
                    state.model.load_partition(r)
                    state.model.post_partition()
                    self.server_queue.set_partitioned()
                else:
                    raise SMPUnsupportedError
            elif state.model.partitioned:
                mb = task.get_microbatch()
                if mb is not None:
                    mb_status = state.pipeline.get_status(mb)
                    self.logger.debug(
                        rmsg(
                            f"Got {mb} from pipeline, with status {mb_status.name} for batch {state.current_minibatch()}"
                        )
                    )
                    if mb_status == MbStatus.READY_FOR_FWD:
                        self.server_queue.increment_in_flight_mbs()
                        req = StepExecutionRequest(
                            mb_args[mb],
                            mb_kwargs[mb],
                            mb,
                            step_func_id,
                            enable_grads=torch.is_grad_enabled(),
                        )
                        state.core.timeline_record_pipeline_event(req.mb, req.__repr__())
                        state.pipeline.promote_status(mb)
                        self.execute_request(req)
                        state.core.timeline_record_pipeline_event(req.mb, req.__repr__())
                    elif mb_status == MbStatus.READY_FOR_BWD:
                        req = self._get_pending_wait_for_req(mb, WaitForBackwardRequest)
                        state.core.timeline_record_pipeline_event(req.mb, req.__repr__())
                        res = ResumeBackward(req)
                        state.pipeline.promote_status(mb)
                        if not state.cfg.zero2d_enabled():
                            state.model.grad_counter.mark_fwd_pass_done(mb)
                        worker_id = self.pop_pending_request(req)
                        self.resume_worker(worker_id, res)
                        state.core.timeline_record_pipeline_event(req.mb, req.__repr__())
        self.comm.clear_minibatch_state()
        self.server_queue.mark_step_done()
        self.step += 1
        self.current_task = None
        return self.outputs

    def run_step_follower(self):
        """
        Waits for a message from other processes.
        These could be
        - ModuleExecutionRequest
        - ForwardExecutionResult that some worker of this server was waiting on
        - PartitioningAndTraceResults from rank0
        - MicrobatchEndResult: output of step function for a microbatch
        - StepEndNotification: simple notification that a minibatch is done
        Note that messages from a worker in same process don't go to comm's queue, so they
        don't show up in this loop. Those would directly go to process_result or process_request.
        """
        # If manually partition or load partition file, do post-partition to assign/bcast params
        if state.model._partitions_assigned and not state.model.partitioned:
            state.model.post_partition()
            self.server_queue.set_partitioned()
        # clear outputs for each minibatch
        self._init_step()
        while True:
            # Note the block here is different from leader.
            # Follower has nothing to do if there is no message, hence the block
            task = self.server_queue.get_next_task()
            self.current_task = task
            if not isinstance(task, IncomingMessageTask):
                raise SMPRuntimeError(
                    f"task should be IncomingMessageTask at step follower, instead getting {type(task)}"
                )
            r = task.get_message()
            if isinstance(r, ExecutionRequest):
                state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                if not state.has_uploaded_metrics and r.mb == 0:
                    state.num_hops += 1
                # follower should never have step or trace requests so below check is always valid
                if r.phase == MbStatus.BWD:
                    # its okay if this method is called for each bwd request
                    state.model.grad_counter.mark_fwd_pass_done(r.mb)
                self.process_request(r)
                state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
            elif isinstance(r, ExecutionResult):
                state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                self.process_result(r)
                state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
            elif isinstance(r, PartitioningAndTraceResults):
                # on mprank!=0
                state.model.load_partition(r)
                state.model.post_partition()
                self.server_queue.set_partitioned()
            elif isinstance(r, MicrobatchEndResult):
                state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                self.outputs[r.mb] = r.outputs
                state.module_manager.clear_microbatch_state(r.mb)
                self.comm.clear_microbatch_state(r.mb)
                if len(self.outputs) == state.num_microbatches():
                    self.comm.clear_minibatch_state()
                    self.server_queue.mark_step_done()
                    state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
                    self.current_task = None
                    return self.outputs
                state.core.timeline_record_pipeline_event(r.mb, r.__repr__())
