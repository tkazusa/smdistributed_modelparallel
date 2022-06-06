# Standard Library
from enum import Enum

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel import tensorflow as smp
from smdistributed.modelparallel.backend.collectives import CommGroup, RankType
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.tensorflow.attrs import (
    DUMMY_ATTR,
    LINK_ID_ATTR,
    MICROBATCH_ATTR,
    OP_ID_ATTR,
)
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.state_mod import state
from smdistributed.modelparallel.tensorflow.utils import (
    ALLGATHER_OP,
    ANY_OP,
    INPUT_OP,
    OUTPUT_OP,
    GraphBuildError,
    GraphTraverser,
    autograph_do_not_convert,
    get_attr,
    get_dummy_spec,
    get_graph_ops,
    get_op,
    get_op_type,
    get_true_spec,
    op_has_attr,
    raise_if_has_cross_microbatch_dependencies,
)


class OpBehavior(Enum):
    RECV = 0
    SEND = 1
    VALVE = 2
    MARKER = 3
    PASS_THROUGH = 4
    DUMMY = 5
    OUTPUT_RECV = 6
    BCAST = 7
    STOP_GRADIENT = 8

    def get_bwd_behavior(op_behavior):
        if op_behavior in [OpBehavior.PASS_THROUGH, OpBehavior.DUMMY, OpBehavior.STOP_GRADIENT]:
            return op_behavior
        if op_behavior == OpBehavior.VALVE:
            return OpBehavior.MARKER
        if op_behavior == OpBehavior.MARKER:
            raise ValueError("Forward op cannot have MARKER behavior!")
        if op_behavior == OpBehavior.RECV:
            return OpBehavior.SEND
        if op_behavior == OpBehavior.SEND:
            return OpBehavior.RECV
        if op_behavior == OpBehavior.OUTPUT_RECV:
            return OpBehavior.DUMMY
        if op_behavior == OpBehavior.BCAST:
            return OpBehavior.VALVE
        raise (f"Undefined op behavior {op_behavior}")

    def get_io(op_behavior):
        """Return whether the op with given behavior can occur as an input or an output of a subgraph"""
        if op_behavior in [OpBehavior.PASS_THROUGH, OpBehavior.DUMMY, OpBehavior.STOP_GRADIENT]:
            return None
        if op_behavior in [OpBehavior.RECV, OpBehavior.VALVE]:
            return 0  # Input
        else:
            return 1  # Output


class CompileStatus(Enum):
    MODEL_COMPILE = 0
    STEP_COMPILE = 1
    TRAIN = 2


class CompilerState:
    def __init__(self, op_attr, op_metadata, op_behavior, io_counts, recv_link_ids, compile_graph):
        self.op_attr = op_attr
        self.op_metadata = op_metadata
        self.op_behavior = op_behavior
        self.io_counts = io_counts
        self.recv_link_ids = recv_link_ids
        self.compile_graph = compile_graph


class CompileCache:
    def __init__(self):
        self._cache = {}

    def put(self, key, value):
        self._cache[self._standardize(key)] = value

    def contains(self, key):
        return self._standardize(key) in self._cache

    def get(self, key):
        return self._cache[self._standardize(key)]

    def clear(self):
        self._cache = {}

    def _standardize(self, key):
        return key


class Compiler:
    """ An object that analyzes the tentatively constructed compile_graph extracts attributes from it, and
        aids in constructing the actual graph that will be used in training.
    """

    def __init__(self, core, state=state):
        self.op_attr = [
            {} for _ in range(state.num_microbatches())
        ]  # op_id -> (tick, peer, link_id)
        self.op_metadata = [{} for _ in range(state.num_microbatches())]  # op_id -> (shape, dtype)
        self.op_behavior = [{} for _ in range(state.num_microbatches())]  # op_id -> OpBehavior
        self.io_counts = []
        self.recv_link_ids = []
        self.compile_graph = None

        self.compile_cache = CompileCache()
        self.current_key = None
        self.current_backend_key = None

        self.core = core
        self.state = state

    @autograph_do_not_convert
    def compile(self, graph, dev, mb, op_id_to_device, forward_only=False):
        """Compile the current compile_graph to extract the attributes. compile_graph must exist.
           Key is an integer associated with the current graph, and is used to determine whether the graph
           has changed since the last compile and a new compilation is required."""
        self.compile_graph = graph
        self.traverser = GraphTraverser(graph)

        self.op_attr[mb], self.op_behavior[mb], self.op_metadata[mb] = self.get_op_attributes(
            dev, mb, op_id_to_device, forward_only
        )

        self.correct_beh_if_op_not_exist(mb)

        if mb == 0:
            get_logger().debug(f"rank {self.core.pp_rank()} op_attr: {self.op_attr[mb]}")
            get_logger().debug(f"rank {self.core.pp_rank()} op_behavior: {self.op_behavior[mb]}")

    def correct_beh_if_op_not_exist(self, mb):
        """If one smp op does not exist in all ranks, reset that op's behavior to STOP_GRADIENT."""
        op_ids = set(self.op_behavior[mb].keys())
        if not self.state.tracking_model:
            all_op_ids = self.state.comm.allgather(op_ids, group=CommGroup.PP_GROUP)
            common_op_ids = op_ids.intersection(*all_op_ids)
            for op_id in op_ids:
                if op_id not in common_op_ids:
                    self.op_behavior[mb][op_id] = OpBehavior.STOP_GRADIENT

    def set_link_id(self, input_id, output_id, link_map):
        link_id = self.state.get_link_id(input_id, output_id)
        link_map[output_id] = link_id
        link_map[-output_id] = link_id
        link_map[input_id] = link_id
        link_map[-input_id] = link_id

    def should_set_link_id(self):
        return self.core.pp_rank() == 0

    def get_all_op_ids(self):
        if self.state.tf_version == 1:
            smp_ops = self.traverser.get_ops_of_type(
                [smp.INPUT_OP, smp.OUTPUT_OP], forward_only=False
            )
            return {get_attr(op, OP_ID_ATTR) for op in smp_ops}
        else:
            smp_ops_mb0 = self.traverser.get_ops_of_type(
                [smp.INPUT_OP, smp.OUTPUT_OP], forward_only=False
            )
            op_ids = set()
            for mb in range(self.state.num_microbatches()):
                op_ids = op_ids.union(
                    {
                        self.state.op_id_gen.transpose_op_id(get_attr(op, OP_ID_ATTR), mb)
                        for op in smp_ops_mb0
                    }
                )
            return op_ids

    def get_op_attributes(self, dev, mb, op_id_to_device, forward_only=False):
        """ Determine the op_behavior, op_metadata, and op_attr by traversing the graph and analyzing the operations."""

        # In TF2, every microbatch operates on the graph for mb=0, but we transpose op_ids as needed
        # In TF1, every microbatch operates on the complete graph, so we filter the ops for each mb
        mb_or_none = mb if self.state.tf_version == 1 else None
        input_ops = self.traverser.get_ops_of_type(INPUT_OP, microbatch=mb_or_none)

        op_behavior = {}
        link_id_map = {}
        peer_map = {}
        tick_map = {}
        subgraph_links = []
        metadata = {}
        all_op_ids = self.get_all_op_ids()

        def set_op(op, beh):
            op_id = get_attr(op, OP_ID_ATTR)
            op_id = self.state.op_id_gen.transpose_op_id(op_id, mb)

            op_behavior[op_id] = beh
            if beh != OpBehavior.SEND and beh != OpBehavior.DUMMY:
                metadata[op_id] = get_true_spec(op)
            else:
                metadata[op_id] = get_dummy_spec()

            # It is possible backward op might not exist in the graph (eg. no gradient exists).
            # So need to check if -op_id actually exists in the graph before setting its bwd behavior.
            if not forward_only and -op_id in all_op_ids:
                op_behavior[-op_id] = OpBehavior.get_bwd_behavior(beh)

        for input_op in input_ops:
            output_ops = self.traverser.backward_walk(input_op, stopping_op_type=OUTPUT_OP)
            if len(output_ops) > 1:
                input_op_list = [input_op]
                raise GraphBuildError(
                    f"Multiple outputs connect to an input! Input op: {input_op_list}, output ops: {output_ops}"
                )

            input_op_id = self.state.op_id_gen.transpose_op_id(get_attr(input_op, OP_ID_ATTR), mb)
            input_op_dev = op_id_to_device[input_op_id]

            if len(output_ops) == 0:
                if get_attr(input_op, DUMMY_ATTR):
                    set_op(input_op, OpBehavior.PASS_THROUGH)
                elif input_op_dev == dev:
                    set_op(input_op, OpBehavior.VALVE)
                else:
                    set_op(input_op, OpBehavior.DUMMY)
            else:
                output_op = output_ops[0]
                output_op_id = self.state.op_id_gen.transpose_op_id(
                    get_attr(output_op, OP_ID_ATTR), mb
                )
                output_op_dev = op_id_to_device[output_op_id]

                # we set the peer rank to just the device id (pp_rank) for now
                # while importing these will be translated into global ranks
                src = output_op_dev  # self.core.get_pp_group()[output_op_dev]
                dest = input_op_dev  # self.core.get_pp_group()[input_op_dev]

                dev_rank = dev  # self.core.get_pp_group()[dev]

                if src != dev_rank and dest != dev_rank:
                    set_op(input_op, OpBehavior.DUMMY)
                    set_op(output_op, OpBehavior.DUMMY)

                    # if they are a send/recv pair for other ranks
                    if src != dest and self.should_set_link_id():
                        self.set_link_id(input_op_id, output_op_id, link_id_map)

                elif src == dest:
                    set_op(input_op, OpBehavior.PASS_THROUGH)
                    set_op(output_op, OpBehavior.PASS_THROUGH)
                elif src == dev_rank:
                    set_op(input_op, OpBehavior.DUMMY)
                    set_op(output_op, OpBehavior.SEND)

                    if self.should_set_link_id():
                        self.set_link_id(input_op_id, output_op_id, link_id_map)

                    peer_map[output_op_id] = dest
                    if -output_op_id in all_op_ids:
                        peer_map[-output_op_id] = dest
                elif dest == dev_rank:
                    set_op(input_op, OpBehavior.RECV)
                    set_op(output_op, OpBehavior.DUMMY)

                    if self.should_set_link_id():
                        self.set_link_id(input_op_id, output_op_id, link_id_map)

                    peer_map[input_op_id] = src
                    if -input_op_id in all_op_ids:
                        peer_map[-input_op_id] = src

        # take care of the last output ops
        output_ops = self.traverser.get_ops_of_type(OUTPUT_OP, microbatch=mb_or_none)
        for output_op in output_ops:
            output_op_id = self.state.op_id_gen.transpose_op_id(get_attr(output_op, OP_ID_ATTR), mb)
            output_op_dev = op_id_to_device[output_op_id]
            if output_op_id not in op_behavior:
                if output_op_dev == dev:
                    # this can only be a marker op since it is not connected to any further input ops
                    set_op(output_op, OpBehavior.BCAST)
                else:
                    set_op(output_op, OpBehavior.OUTPUT_RECV)
                    peer_map[
                        output_op_id
                    ] = output_op_dev  # self.core.get_pp_group()[output_op_dev]

                if self.should_set_link_id():
                    self.set_link_id(output_op_id, output_op_id, link_id_map)

        # for each op get the tick based on graph topology and the chosen pipeline type
        tick_map = self.get_tick_map(mb, all_op_ids, forward_only=forward_only)

        # Create link id for allgather ops
        allgather_ops = self.traverser.get_ops_of_type(ALLGATHER_OP, forward_only=False)
        for op in allgather_ops:
            mb = get_attr(op, MICROBATCH_ATTR)
            if mb != 0:
                op_id = get_attr(op, OP_ID_ATTR)
                if self.should_set_link_id():
                    link_id = get_attr(op, LINK_ID_ATTR)
                    _, index, _ = self.state.get_att_from_link_id(link_id)
                    link_id_new = self.state.get_comm_link_id(mb, index)
                    link_id_map[op_id] = link_id_new
                op_behavior[op_id] = OpBehavior.DUMMY

        # Broadcast link ids
        if self.core.pp_rank() == 0 and not self.state.tracking_model:
            self.state.comm.broadcast(link_id_map, group=CommGroup.PP_GROUP)
        elif not self.state.tracking_model:
            link_id_map = self.state.comm.recv_from(0, RankType.PP_RANK)

        # merge op attributes into single object
        op_attr = {}
        for op_id, behavior in op_behavior.items():
            if behavior in [OpBehavior.VALVE, OpBehavior.MARKER]:
                op_attr[op_id] = tick_map[op_id], -1, -1
            elif behavior in [OpBehavior.BCAST]:
                op_attr[op_id] = tick_map[op_id], -1, link_id_map[op_id]
            elif behavior in [OpBehavior.RECV, OpBehavior.SEND, OpBehavior.OUTPUT_RECV]:
                op_attr[op_id] = tick_map[op_id], peer_map[op_id], link_id_map[op_id]
            else:
                op_attr[op_id] = (
                    tick_map[op_id],
                    -1,
                    (link_id_map[op_id] if op_id in link_id_map else -1),
                )  # dummy attributes required for pass through ops

        return op_attr, op_behavior, metadata

    def get_tick_map(self, mb, all_op_ids, forward_only=False):
        """ Create a mapping from op_id to the tick when that op will be executed.
        Ask the pipeline object what the tick should be for each op. """

        tick_map = {}
        self.state.pipeline.update(self.compile_graph, self.traverser, mb)

        # for each custom op, pass microbatch and subgraph to get the forward tick
        smp_ops = self.traverser.get_ops_of_type([smp.INPUT_OP, smp.OUTPUT_OP])
        for op in smp_ops:
            op_id = self.state.op_id_gen.transpose_op_id(get_attr(op, OP_ID_ATTR), mb)
            tick_map[op_id] = self.state.pipeline.get_tick(op_id, mb)
            if not forward_only and -op_id in all_op_ids:
                tick_map[-op_id] = self.state.pipeline.get_tick(-op_id, mb)
        return tick_map

    def save_state(self, key, outputs):
        if not self.compile_cache.contains(key):
            if state.tf_version == 1:
                # For TF1.x we create a new traverser, because the existing self.traverser is for
                # state.compile_graph, which is not the same as the default graph at this point
                traverser = GraphTraverser(tf.compat.v1.get_default_graph())
            else:
                traverser = self.traverser
            self.io_counts, self.recv_link_ids = self.compute_io_counts(outputs, traverser)
            comp_state = CompilerState(
                self.op_attr,
                self.op_metadata,
                self.op_behavior,
                self.io_counts,
                self.recv_link_ids,
                self.compile_graph,
            )
            self.compile_cache.put(key, comp_state)

    def load_state(self, key):
        if self.current_key != key:
            if not self.compile_cache.contains(key):
                raise ValueError(f"Invalid compiler key {key}.")

            comp_state = self.compile_cache.get(key)
            self.op_attr = comp_state.op_attr
            self.op_metadata = comp_state.op_metadata
            self.op_behavior = comp_state.op_behavior
            self.io_counts = comp_state.io_counts
            self.recv_link_ids = comp_state.recv_link_ids

            self.current_key = key

    def compute_io_counts(self, outputs, traverser):
        """Compute the number of SmpInput and SmpOutput ops of each behavior that will be encountered during this
           graph execution. Used by the backend to determine when a tick ends and the next one starts. Does not return
           the counts for PASS_THROUGH and DUMMY behaviors, since these ops do not communicate with the backend.

           Uses caching of the results so that if the set of fetches did not change, the counts will not be re-computed.

           In TensorFlow 1.x, this is called by SMPCompileHook at every step, since tf.Session.run() call can be made
           with different arguments, which can alter the set of SMP ops that will be encountered.

           In TensorFlow 2.x, it is called per tf.function trace.
        """

        outputs_list = outputs if isinstance(outputs, list) else [outputs]

        smp_ops = traverser.backward_walk(
            outputs_list[:], stopping_op_type=None, target_op_type=[OUTPUT_OP, INPUT_OP]
        )
        if len(smp_ops) == 0:
            dist_model_name = (
                "smp.DistributedModel" if state.tf_version == 2 else "smp.distributed_model()"
            )
            raise ValueError(
                f"The return value from smp.step function must depend on the {dist_model_name} output."
            )

        # For TF2.x, this is executed only for microbatch 0, and the counts for other microbatches
        # are inferred. This is because we trace only the first microbatch to save on start-up time.
        # For TF1.x, smp_ops above contains ops from all microbatches, so we compute io_counts for
        # all microbatches.

        ops_by_mb = [[] for _ in range(self.state.num_microbatches())]
        for op in smp_ops:
            ops_by_mb[get_attr(op, MICROBATCH_ATTR)].append(op)

        io_counts = [[] for _ in range(2)]
        recv_link_ids = []

        for mb in range(self.state.num_microbatches()):
            if self.state.tf_version == 2:
                self._update_io_counts(
                    ops_by_mb[0], mb, io_counts, recv_link_ids, transpose_op_ids=True
                )
            else:
                self._update_io_counts(
                    ops_by_mb[mb], mb, io_counts, recv_link_ids, transpose_op_ids=False
                )

        # equalize the length of the two io_counts lists by padding zeros
        num_ticks = max([len(x) for x in io_counts])
        for i in range(2):
            io_counts[i].extend([0 for _ in range(num_ticks - len(io_counts[i]))])

        get_logger().debug(f"pp_rank {smp.pp_rank()} has io counts: {io_counts}")
        return io_counts, recv_link_ids

    def _update_io_counts(self, ops, mb, io_counts, recv_link_ids, transpose_op_ids=True):
        for op in ops:
            if transpose_op_ids:
                op_id = self.state.op_id_gen.transpose_op_id(get_attr(op, OP_ID_ATTR), mb)
            else:
                op_id = get_attr(op, OP_ID_ATTR)
            behavior = self.op_behavior[mb][op_id]

            io = OpBehavior.get_io(behavior)
            if io is not None:
                tick = self.op_attr[mb][op_id][0]
                self._increment_list(io_counts[io], tick)
                get_logger().debug(
                    f"pp_rank {smp.pp_rank()} registering op {op.name} with tick {tick} and OpBehavior {behavior}"
                )

            if behavior == OpBehavior.RECV or behavior == OpBehavior.OUTPUT_RECV:
                self._add_to_list(recv_link_ids, tick, self.op_attr[mb][op_id][2])

    def _increment_list(self, lst, ind):
        if len(lst) <= ind:
            lst.extend([0 for _ in range(ind - len(lst) + 1)])
        lst[ind] += 1

    def _add_to_list(self, lst, ind, item):
        if len(lst) <= ind:
            lst.extend([[] for _ in range(ind - len(lst) + 1)])
        lst[ind].append(item)

    def get_op_behaviors(self):
        num_ops = 0
        op_ids, behaviors = [], []

        for mb in range(self.state.num_microbatches()):
            mb_op_ids, mb_behaviors = tuple(
                map(list, zip(*[(k, v.value) for k, v in self.op_behavior[mb].items()]))
            )
            op_ids.extend(mb_op_ids)
            behaviors.extend(mb_behaviors)
            num_ops += len(mb_op_ids)

        return num_ops, op_ids, behaviors

    def register_op_behavior(self):
        num_ops, op_ids, behaviors = self.get_op_behaviors()
        self.core.register_op_behavior(num_ops, op_ids, behaviors)

    def register_backend_state(self, key):
        # NOTE: this is only called by TF1

        if self.current_backend_key != key:
            self.core.register_recv_link_ids(self.recv_link_ids)

            num_ticks = len(self.io_counts[0])
            self.core.register_counts(num_ticks, *self.io_counts)

            self.register_op_behavior()
            self.current_backend_key = key
