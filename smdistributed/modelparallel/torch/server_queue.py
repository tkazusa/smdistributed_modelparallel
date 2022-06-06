# Standard Library
import collections
import copy
import time
from typing import Any, NamedTuple

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.collectives import (
    CommGroup,
    RankType,
    TransactionIdentifier,
)
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.core import dp_rank, pp_rank, rank
from smdistributed.modelparallel.torch.messages import (
    DummyBackwardResult,
    ExecutionResult,
    ForwardExecutionResult,
    MicrobatchEndResult,
    ModuleExecutionRequest,
    ResumeBackward,
    SequentialModulesExecutionRequest,
    StepExecutionRequest,
    StubbedMessage,
    TraceStepExecutionRequest,
    WaitForBackwardDoneRequest,
    WaitForBackwardRequest,
)
from smdistributed.modelparallel.torch.module_manager import PartitioningAndTraceResults
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import rmsg

logger = get_logger()

# for debugging
TorchTensorMeta_ = collections.namedtuple(
    "TorchTensorMeta_", "dims dummy_tensor shape_vec source link_id"
)


class TaskMetadata(NamedTuple):
    """ Metadata for a ServerTask. Used to match corresponding tasks across dp_ranks. """

    is_no_task: bool
    is_incoming_message: bool
    message_type: str
    is_response: bool
    is_tracing_request: bool
    is_tracing_result: bool
    mb: int
    message_meta: Any

    def __eq__(self, other):
        return (
            self.is_no_task == other.is_no_task
            and self.is_incoming_message == other.is_incoming_message
            and self.message_type == other.message_type
            and self.is_response == other.is_response
            and self.is_tracing_request == other.is_tracing_request
            and self.mb == other.mb
            and self.message_meta == other.message_meta
        )

    def __hash__(self):
        return hash(
            (
                self.is_no_task,
                self.is_incoming_message,
                self.message_type,
                self.is_response,
                self.is_tracing_request,
                self.mb,
                self.message_meta,
            )
        )


class ServerTask:
    """ A task to be executed by the server. """

    def __init__(self):
        self._task_metadata = None

    @property
    def task_metadata(self):
        return self._task_metadata

    @task_metadata.setter
    def task_metadata(self, metadata):
        self._task_metadata = metadata

    def matches(self, metadata):
        return self.task_metadata == metadata


class IncomingMessageTask(ServerTask):
    """ A task that is defined by an underlying message. Can be an execution request,
        result, or control messaging. """

    def __init__(self, message):
        self.message = message

        message_type = message.__class__.__name__
        is_response = isinstance(message, (ExecutionResult, ResumeBackward))
        is_tracing_request = isinstance(message, TraceStepExecutionRequest)
        is_tracing_result = isinstance(message, PartitioningAndTraceResults)

        if is_tracing_request or is_tracing_result:
            message_meta = None
        elif isinstance(
            message,
            (StepExecutionRequest, ModuleExecutionRequest, SequentialModulesExecutionRequest),
        ):
            message_meta = copy.copy(message)
            message_meta.strip_tensors()
        elif isinstance(message, ForwardExecutionResult):
            message_meta = copy.copy(message.request)
            message_meta.strip_tensors()
        elif isinstance(message, DummyBackwardResult):
            message_meta = copy.copy(message)
        elif isinstance(message, ExecutionResult):
            message_meta = copy.copy(message.req)
        elif isinstance(message, MicrobatchEndResult):
            message_meta = copy.copy(message)
            message_meta.strip_outputs()
        elif isinstance(message, WaitForBackwardRequest):
            message_meta = copy.copy(message)
        elif isinstance(message, WaitForBackwardDoneRequest):
            message_meta = copy.copy(message)
        elif isinstance(message, ResumeBackward):
            message_meta = copy.copy(message.req)
        else:
            raise RuntimeError(f"Unsupported message type {type(message)}.")

        self.task_metadata = TaskMetadata(
            False,
            True,
            message_type,
            is_response,
            is_tracing_request,
            is_tracing_result,
            None,
            message_meta,
        )

    def get_message(self):
        return self.message

    def __repr__(self):
        return f"<IncomingMessageTask type={self.task_metadata.message_type}, is_response={self.task_metadata.is_response}>"


class NextMicrobatchTask(ServerTask):
    """ A task that represents the start of a forward or backward pass. Only available at pp_rank 0.
        The only relevant attribute of a NextMicrobatch task is the microbatch itself. """

    def __init__(self, microbatch):
        self.microbatch = microbatch
        self.task_metadata = TaskMetadata(False, False, None, False, False, False, microbatch, None)

    def get_microbatch(self):
        return self.microbatch

    def __repr__(self):
        return f"<NextMicrobatchTask::mb::{self.microbatch}>"


class NoTask(ServerTask):
    """ A non-task that requires no action from server. """

    def __init__(self):
        self.task_metadata = TaskMetadata(True, False, None, False, False, False, None, None)

    def __repr__(self):
        return f"<NoTask>"


class ServerQueue:
    """ Represents a queue of ServerTasks, although not implemented as a real queue. Offers an API
        consisting mainly of has_more_tasks()/get_next_task(), which the server can use to retrieve
        the next task it needs to work on. This abstracts task prioritization from the server. """

    def __init__(self, comm, pipeline):
        self.comm = comm
        self.pipeline = pipeline
        self.partitioned = False
        self.in_flight_mbs = 0

    def has_more_tasks(self):
        raise NotImplementedError

    def get_next_task(self, block=False):
        raise NotImplementedError

    def mark_step_done(self):
        self.in_flight_mbs = 0

    def set_partitioned(self):
        self.partitioned = True

    def increment_in_flight_mbs(self):
        self.in_flight_mbs += 1

    def decrement_in_flight_mbs(self):
        self.in_flight_mbs -= 1


class DeterministicServerQueue(ServerQueue):
    """ A server queue that enforces a deterministic pipeline schedule regardless of message arrival times.
        In the record_step dp_rank 0 records the naturally-occuring order of tasks (based on message arrival
        timings) and broadcasts this to the other dp_ranks, who follow the same order. This task order
        is then followed by all dp_ranks in subsequent steps. Implemented as a wrapper around OpportunisticServerQueue.
    """

    def __init__(self, comm, pipeline):
        super(DeterministicServerQueue, self).__init__(comm, pipeline)
        self.current_task = -1
        self.task_order = collections.defaultdict(list)
        self.task_order_set = collections.defaultdict(bool)
        self.step = collections.defaultdict(int)

        self.recorded_execution_times = collections.defaultdict(list)
        self.recorded_task_orders = collections.defaultdict(list)
        self.best_step = -1

        # record the task order in this step. should be set larger than 0
        # since the order of tasks in step 0 is not representative and
        # replicating this order causes performance loss.
        self.record_step = 5

        self.opportunistic_queue = OpportunisticServerQueue(self.comm, self.pipeline)
        self.received_tasks = {}
        self.received_tasks_artifact = {}

        # mapping from task idx to the stubs inside for each step function
        # for static mode
        self.recorded_task_idx_to_stubs = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self.task_idx_to_stubs = collections.defaultdict(list)
        # Used to record possible stubs and stub mesaage for an incoming message
        self.task_artifact = None

        self.has_recorded_bwd_fwd_tasks = False
        self.recorded_bwd_task_to_fwd_tasks = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(set))
        )
        self.bwd_task_to_fwd_tasks = {}
        self.load_scheduled = collections.defaultdict(list)

    def set_task_artifact(self, stubbed_msg, stubs):
        self.task_artifact = stubbed_msg, stubs

    def set_partitioned(self):
        super(DeterministicServerQueue, self).set_partitioned()
        self.opportunistic_queue.set_partitioned()

    def increment_in_flight_mbs(self):
        self.opportunistic_queue.increment_in_flight_mbs()

    def decrement_in_flight_mbs(self):
        self.opportunistic_queue.decrement_in_flight_mbs()

    def maybe_record_task_pair(self, fwd_task_and_index, bwd_task):
        if self.step[state.current_step_func()] <= self.record_step:
            step_func = state.current_step_func()
            cur_step = self.step[step_func]

            self.recorded_bwd_task_to_fwd_tasks[step_func][cur_step][bwd_task].add(
                fwd_task_and_index
            )

    def has_more_tasks(self):
        if not self.partitioned:
            return True

        if not self.task_order_set[state.current_step_func()]:
            return self.opportunistic_queue.has_more_tasks()
        else:
            return self.current_task < len(self.task_order[state.current_step_func()]) - 1

    def should_record(self):
        return self.step[state.current_step_func()] <= self.record_step

    def record_task_sequence_static(self, next_task, cur_step):
        """ During recording:
            - Replace the message in IncomingMessageTask with the StubbedMessage if possible
            - Record the stubs for each task if possible
            Used only for static mode.
        """
        if self.task_artifact != None:
            stubmsg, stubs = self.task_artifact
            self.recorded_task_orders[state.current_step_func()][cur_step].append(
                IncomingMessageTask(stubmsg)
            )
            self.recorded_task_idx_to_stubs[state.current_step_func()][cur_step].append(stubs)
            self.task_artifact = None
        else:
            assert not isinstance(next_task, IncomingMessageTask) or not isinstance(
                next_task.message, StubbedMessage
            ), f"StubbedMessage {next_task.message} is not recorded in static mode!"
            self.recorded_task_orders[state.current_step_func()][cur_step].append(next_task)
            self.recorded_task_idx_to_stubs[state.current_step_func()][cur_step].append(None)

    def _record_task(self, next_task):
        cur_step = self.step[state.current_step_func()]

        if cur_step == len(self.recorded_task_orders[state.current_step_func()]):
            self.recorded_task_orders[state.current_step_func()].append([])

        if not state.cfg.static_mode:
            self.recorded_task_orders[state.current_step_func()][cur_step].append(
                next_task.task_metadata
            )
        else:
            self.record_task_sequence_static(next_task, cur_step)

        if state.cfg.offload_activations and cur_step == self.record_step:
            self.load_scheduled[state.current_step_func()].append(False)

    def find_best_step(self):
        logger.debug(
            f"Step execution times: {self.recorded_execution_times[state.current_step_func()]}"
        )

        # choosing the max since we record negative durations
        step_times = self.recorded_execution_times[state.current_step_func()]

        # if fast mode is enabled, ignore the first step, since the communication pattern
        # will not be the same as the other steps
        if state.cfg.fast_mode:
            assert (
                len(step_times) > 1
            ), "When fast mode is enabled, record_step must be larger than 0"
            step_times = step_times[1:]
            return 1 + step_times.index(min(step_times))

        return step_times.index(min(step_times))

    def get_next_task(self):
        # if not yet partitioned, return the received next task immediately
        if not self.partitioned:
            return self.opportunistic_queue.get_next_task()

        trans_id = TransactionIdentifier(8080, False)

        if not self.task_order_set[state.current_step_func()]:
            # before self.record_step: task order is not yet determined
            # dp_rank 0 broadcasts the order of tasks it receives to its dp_group
            # other ranks follow this order

            if dp_rank() == 0:
                next_task = self.opportunistic_queue.get_next_task()

                if pp_rank() == 0:
                    if (
                        len(self.recorded_execution_times[state.current_step_func()])
                        == self.step[state.current_step_func()]
                    ):
                        self.recorded_execution_times[state.current_step_func()].append(time.time())

                if not isinstance(next_task, NoTask):
                    self.current_task += 1
                    if self.should_record():
                        self._record_task(next_task)

                    state.comm.async_bcast(next_task.task_metadata, trans_id, CommGroup.DP_GROUP)
                    state.comm.wait_transaction(trans_id)

                return next_task
            else:
                # get the metadata for the next task to be executed
                state.comm.async_recv_from(0, trans_id, RankType.DP_RANK)
                next_task_meta = state.comm.wait_recv(0, trans_id, RankType.DP_RANK)

                if next_task_meta.is_no_task:
                    return NoTask()

                next_task = self._get_next_task_with_meta(next_task_meta)
                if self.should_record():
                    self._record_task(next_task)
                if not isinstance(next_task, NoTask):
                    self.current_task += 1
                return next_task
        else:
            self.current_task += 1
            if state.cfg.static_mode:
                # directly return the task
                current_task = self.task_order[state.current_step_func()][self.current_task]
                if isinstance(current_task, IncomingMessageTask) and isinstance(
                    current_task.message, StubbedMessage
                ):
                    with state.serialization_manager.catch_and_raise_for_large_object(current_task):
                        current_task = copy.deepcopy(current_task)
                    destub_msg = self.comm.reconstruct_destubmessage(
                        current_task.message,
                        self.task_idx_to_stubs[state.current_step_func()][self.current_task],
                    )
                    current_task.message = destub_msg
            else:
                task_metadata = self.task_order[state.current_step_func()][self.current_task]
                current_task = self._get_next_task_with_meta(task_metadata)

            self.maybe_load_activations()
            return current_task

    def register_incoming_tensor_metadata(self):
        """Passing the cached tensor reception requests to the listener
           Each (src, link_id) will contain a queue of the tensor reception requests"""
        incoming_tensors_metadata = []
        incoming_tensors_metadata_ = []
        recorded_src_ids = set()
        for idx, task in enumerate(self.task_order[state.current_step_func()]):
            if isinstance(task, IncomingMessageTask) and isinstance(task.message, StubbedMessage):
                assert (
                    self.task_idx_to_stubs[state.current_step_func()][idx] != None
                ), f"StubbedMessage {task.message} and its stubs are not recorded in static mode!"
                stubs = self.task_idx_to_stubs[state.current_step_func()][idx]
                current_task_tensor_meta = []
                current_task_tensor_meta_ = []
                for stub in stubs:
                    # Let module P be on rank 0, and let two of its children C1 and C2 appear back to back, both on rank 1:
                    #     x = self.C1(x)
                    #     x = self.C2(x)
                    # In this case normal execution sends the output of C1 to P which is then sent back to C2, so a needless
                    # roundtrip between 0 and 1.
                    # Under fast mode, C1 just locally caches the tensor and does not send anything to P. P instead creates a
                    # dummy tensor locally, which is also not sent to C2. When execution reaches C2, the stub for the incoming
                    # tensor will not be dummy, but the src will be the rank itself, in which case it will just pick up the
                    # locally cached tensor. So in this example P should not make a preemptive recv call, and neither should C2,
                    # which covers the two cases in the condition below.
                    if not stub.is_dummy and not stub.src == rank():
                        src_id = (stub.src, stub.link_id)
                        assert (
                            src_id not in recorded_src_ids
                        ), f"Duplicated src_id pair {src_id} is not allowed!"
                        recorded_src_ids.add(src_id)
                        tensor_meta = smplib.TorchTensorMeta(
                            dims=len(stub.shape),
                            dummy_tensor=torch.empty(1, dtype=stub.dtype),
                            shape_vec=list(stub.shape),
                            source=src_id[0],
                            link_id=src_id[1],
                        )
                        tensor_meta_ = TorchTensorMeta_(
                            dims=len(stub.shape),
                            dummy_tensor=torch.empty(1, dtype=stub.dtype),
                            shape_vec=list(stub.shape),
                            source=src_id[0],
                            link_id=src_id[1],
                        )
                        current_task_tensor_meta_.append(tensor_meta_)
                        current_task_tensor_meta.append(tensor_meta)
                incoming_tensors_metadata.append(current_task_tensor_meta)
                incoming_tensors_metadata_.append(current_task_tensor_meta_)
        logger.debug(f"Recorded tensor metadata at rank {rank()}: {incoming_tensors_metadata_}")
        smplib.smp_torch_register_incoming_tensor_metadata(
            incoming_tensors_metadata, state.current_step_func().id
        )

    def maybe_load_activations(self):
        """ Preemptively load the activations for the next activation_loading_horizon tasks """

        if state.cfg.offload_activations and self.has_recorded_bwd_fwd_tasks:
            step_fn = state.current_step_func()

            for offset in range(state.cfg.activation_loading_horizon):
                task_metadata = self.task_order[step_fn][self.current_task + offset]
                if self._task_needs_activations(task_metadata, self.current_task + offset):
                    if self._is_ready_for_loading(offset):
                        self._load_for_task(task_metadata, self.current_task + offset)
                        self.load_scheduled[step_fn][self.current_task + offset] = True

                if self.current_task + offset == len(self.task_order[step_fn]) - 1:
                    break

    def _is_ready_for_loading(self, offset):
        task_id = self.current_task + offset
        step_fn = state.current_step_func()

        if self.load_scheduled[step_fn][task_id]:
            return False

        task_metadata = self.task_order[step_fn][task_id]

        if task_metadata not in self.bwd_task_to_fwd_tasks[step_fn]:
            return False

        fwd_tasks_and_indices = self.bwd_task_to_fwd_tasks[step_fn][task_metadata]

        for i in range(offset):
            upcoming_task_metadata = self.task_order[step_fn][self.current_task + i]

            # if the task depends on an upcoming task, do not schedule the load yet
            if (upcoming_task_metadata, self.current_task + i) in fwd_tasks_and_indices:
                return False

        return True

    def _task_needs_activations(self, task_metadata, task_id):
        step_fn = state.current_step_func()
        if task_metadata in self.bwd_task_to_fwd_tasks[step_fn]:
            if not task_metadata.is_incoming_message:
                mb_status = state.pipeline.get_status(task_metadata.mb)
                if mb_status == MbStatus.READY_FOR_BWD:
                    return True
            else:
                return True

        return False

    def _load_for_task(self, task_metadata, task_id):
        for fwd_task_and_index in self.bwd_task_to_fwd_tasks[state.current_step_func()][
            task_metadata
        ]:
            logger.debug(rmsg(f"Loading activations for task {fwd_task_and_index}"))
            state.current_offloader().load(fwd_task_and_index)

    def _get_next_task_with_meta(self, metadata):
        """ Returns a task whose metadata matches the given input. If there is no such task received earlier,
            waits until one is received. Places the tasks received in the meantime in self.received_tasks
            for later processing. """

        # check if the previously received tasks match the given metadata
        task = self.received_tasks.pop(metadata, None)
        if task is not None:
            self.task_artifact = self.received_tasks_artifact.pop(metadata, None)
            return task

        # if not, wait for the matching task from the opportunistic queue
        while True:
            next_task = self.opportunistic_queue.get_next_task()
            if next_task.matches(metadata):
                return next_task
            elif (
                not isinstance(next_task, NoTask)
                and next_task.task_metadata not in self.received_tasks
            ):
                if self.task_artifact != None:
                    self.received_tasks_artifact[next_task.task_metadata] = self.task_artifact
                    self.task_artifact = None
                self.received_tasks[next_task.task_metadata] = next_task

    def mark_step_done(self):
        super(DeterministicServerQueue, self).mark_step_done()

        cur_step = self.step[state.current_step_func()]

        if not self.task_order_set[state.current_step_func()] and pp_rank() == 0 and dp_rank() == 0:
            torch.cuda.synchronize()
            self.recorded_execution_times[state.current_step_func()][cur_step] -= time.time()

        # set the task order to the one that gave the lowest step time in the first
        # record_step steps
        if cur_step == self.record_step:
            if pp_rank() == 0 and dp_rank() == 0:
                assert rank() == 0, "pp_rank 0 and dp_rank 0 must be rank 0"
                self.best_step = self.find_best_step()
                state.comm.broadcast(self.best_step, CommGroup.WORLD)
            else:
                self.best_step = state.comm.recv_from(0, RankType.WORLD_RANK)

            if rank() == 0:
                logger.debug(f"Best step: {self.best_step}")

            step_fn = state.current_step_func()
            self.task_order[step_fn] = self.recorded_task_orders[step_fn][self.best_step]
            if state.cfg.static_mode:
                # update the recorded link ids for server communicator
                self.comm.msg_meta_to_link_id[step_fn] = self.comm.record_msg_meta_to_link_id[
                    step_fn
                ][self.best_step]
                self.task_idx_to_stubs[step_fn] = self.recorded_task_idx_to_stubs[step_fn][
                    self.best_step
                ]
                self.register_incoming_tensor_metadata()
            self.has_recorded_bwd_fwd_tasks = len(self.recorded_bwd_task_to_fwd_tasks[step_fn]) > 0
            if state.cfg.offload_activations and self.has_recorded_bwd_fwd_tasks:
                # recorded length might be zero if activation checkpointing is not enabled
                self.bwd_task_to_fwd_tasks[step_fn] = self.recorded_bwd_task_to_fwd_tasks[step_fn][
                    self.best_step
                ]

            self.task_order_set[state.current_step_func()] = True

        if state.cfg.offload_activations:
            size = len(self.load_scheduled[state.current_step_func()])
            self.load_scheduled[state.current_step_func()] = [False for _ in range(size)]

        self.current_task = -1
        self.task_artifact = None
        self.step[state.current_step_func()] += 1
        self.received_tasks.clear()
        self.received_tasks_artifact.clear()


class OpportunisticServerQueue(ServerQueue):
    """ A non-deterministic server queue that prioritizes the processing of an incoming message
        with respect to processing a next microbatch. Pipeline schedule will depend on the
        timing of incoming messages. """

    def should_record(self):
        return False

    def has_more_tasks(self):
        # always return True for pp_rank != 0 since this is not used as a stopping condition in this case
        return pp_rank() != 0 or self.pipeline.has_more_ticks()

    def get_next_task(self):
        if pp_rank() == 0:
            next_mb = self.pipeline.get_next_microbatch()

            # 1. if there is a backward pass to start, prioritize that first
            if (
                self.partitioned
                and next_mb is not None
                and self.pipeline.get_status(next_mb) == MbStatus.READY_FOR_BWD
            ):
                return NextMicrobatchTask(next_mb)

            # 2. if there is a request to respond to, prioritize that next
            elif self.comm.has_message() or not self.partitioned:
                msg = self.comm.get_next_message(block=True)
                return IncomingMessageTask(msg)

            # 3. if we have started all microbatches already, then wait for one of the above two cases to occur
            elif next_mb is None:
                return NoTask()

            # 4. if we are above active_microbatches limit, then wait, to keep memory use in check
            elif (
                self.pipeline.get_status(next_mb) == MbStatus.READY_FOR_FWD
                and self.in_flight_mbs >= state.cfg.active_microbatches
            ):
                # return dummy task if we are above the active_microbatch limit
                return NoTask()

            # 5. else: absolutely nothing else to do, and we are within active_microbatches budget: start new fwd pass
            else:
                return NextMicrobatchTask(next_mb)
        else:
            msg = self.comm.get_next_message(block=True)
            return IncomingMessageTask(msg)
