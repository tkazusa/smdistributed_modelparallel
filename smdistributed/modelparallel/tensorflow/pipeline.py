# Standard Library
# First Party
from smdistributed.modelparallel.tensorflow import INPUT_OP, OUTPUT_OP
from smdistributed.modelparallel.tensorflow.attrs import OP_ID_ATTR
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.utils import get_attr, get_op_type


class UnsupportedGraphException(Exception):
    """ Raised if the specified graph is unsupported by the pipeline scheduler.
    """


class TFPipeline:
    """TF Pipeline object that implements the logic common to all pipeline types. A pipeline object effectively
       specifies a (not necessarily injective) mapping between a tick and a (subgraph, microbatch, fwd/bwd) triplet.
       A tick is a unit time slot in the execution schedule. At every tick, each device executes for the set of
       (subgraph, microbatch, fwd/bwd) triplets assigned to it for that tick by the pipeline object.
       Any new pipeline type must subclass TFPipeline and implement only three methods:
       get_fwd_tick, get_bwd_tick, is_forward. """

    def __init__(self, num_mb, state):
        self.op_id_to_depth = {}
        self.num_depths = 0
        self.num_fwd_depths = 0
        self.num_mb = num_mb
        self.state = state

        self._fwd_to_bwd_depth = {}
        self._bwd_to_fwd_depth = {}
        self._bwd_depths = set()

    def update(self, graph, traverser, mb):
        if graph is not None:
            for node in traverser.get_ops_of_type(
                [INPUT_OP, OUTPUT_OP], microbatch=0, forward_only=False
            ):
                self._set_depth_for_node(node, traverser, mb)

    def _add_depth_maps(self, op_id, depth):
        if op_id < 0:
            fwd_depth = self.op_id_to_depth[-op_id]
            self._fwd_to_bwd_depth[fwd_depth] = depth
            self._bwd_to_fwd_depth[depth] = fwd_depth
            self._bwd_depths.add(depth)

    def _set_depth_for_node(self, node, traverser, mb):
        op_id = self.state.op_id_gen.transpose_op_id(get_attr(node, OP_ID_ATTR), mb)

        # if we have already computed the depth, look it up
        if op_id in self.op_id_to_depth:
            return self.op_id_to_depth[op_id]

        # if not, we have to recursively compute it
        if get_op_type(node) == INPUT_OP:
            output_nodes = traverser.backward_walk(node, stopping_op_type=OUTPUT_OP)
            if len(output_nodes) == 0:
                self.op_id_to_depth[op_id] = 0
            else:
                depth = max([self._set_depth_for_node(n, traverser, mb) for n in output_nodes]) + 1
                self.op_id_to_depth[op_id] = depth
                self._add_depth_maps(op_id, depth)

        elif get_op_type(node) == OUTPUT_OP:
            input_nodes = traverser.backward_walk(node, stopping_op_type=INPUT_OP)
            depth = max([self._set_depth_for_node(n, traverser, mb) for n in input_nodes])
            self.op_id_to_depth[op_id] = depth
            self._add_depth_maps(op_id, depth)
        else:
            raise ValueError(f"Unsupported op type {node.op}")

        self.num_depths = max(self.num_depths, self.op_id_to_depth[op_id] + 1)
        get_logger().debug(f"Number of depths: {self.num_depths}")
        if op_id > 0:
            self.num_fwd_depths = max(self.num_fwd_depths, self.op_id_to_depth[op_id] + 1)

        return self.op_id_to_depth[op_id]

    def bwd_to_fwd_depth(self, depth):
        return self._bwd_to_fwd_depth[depth]

    def fwd_to_bwd_depth(self, depth):
        return self._fwd_to_bwd_depth[depth]

    def is_bwd_depth(self, depth):
        return depth in self._bwd_depths

    def get_tick(self, op_id, microbatch):
        raise NotImplementedError

    def num_ticks(self):
        raise NotImplementedError


class SimplePipeline(TFPipeline):
    """A pipeline which first computes all the forward passes for all microbatches,
       followed by all the backward passes. Example:

        Subgraph topology: SG0 -> SG1 -> SG2
        # microbatches: 4

        SG2:       F0 F1 F2 F3 B3 B2 B1 B0
        SG1:    F0 F1 F2 F3       B3 B2 B1 B0
        SG0: F0 F1 F2 F3             B3 B2 B1 B0
    """

    def num_ticks(self):
        return 2 * (self.num_fwd_depths + self.num_mb) - 2

    def get_tick(self, op_id, microbatch):
        if op_id not in self.op_id_to_depth:
            return -1
        if op_id > 0:
            return self.op_id_to_depth[op_id] + microbatch
        else:
            fwd_tick = self.get_tick(-op_id, microbatch)
            return 2 * (self.num_fwd_depths + self.num_mb) - 3 - fwd_tick


class InterleavedPipeline(TFPipeline):
    """A pipeline which alternates between forward and backward execution for all microbatches,
       for all subgraphs.
        Subgraph topology: SG0 -> SG1 -> SG2
        # microbatches: 6

        SG2:       F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5
        SG1:    F0    F1 B0 F2 B1 F3 B2 F4 B3 F5 B4    B5
        SG0: F0    F1    F2 B0 F3 B1 F4 B2 F5 B3    B4    B5
    """

    def num_ticks(self):
        return self.num_depths + 2 * self.num_mb - 2

    # TODO: refine/generalize this logic later
    def get_tick(self, op_id, microbatch):
        if op_id not in self.op_id_to_depth:
            return -1
        return self.op_id_to_depth[op_id] + 2 * microbatch
