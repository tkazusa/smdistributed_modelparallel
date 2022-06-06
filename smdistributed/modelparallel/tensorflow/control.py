# Standard Library

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel import tensorflow as smp
from smdistributed.modelparallel.tensorflow.ops import bundle
from smdistributed.modelparallel.tensorflow.state_mod import state


class ControlManager:
    """ Mainly exposes the 'maybe_reroute_control_inputs' function, which is used to establish control dependencies
        between subgraph replicas for different microbatches when XLA is enabled.

        When enabled, XLA can combine ops from two separate replicas of the same subgraph into a single op. Since
        the backend will not execute one replica without finishing the other, and the combined op will not execute
        until all its inputs are available, this causes a deadlock and results in a hang.

        ControlManager addresses this issue by creating artificial dependencies between the subgraph replicas, by
        adding the output of one replica as a control input to the input of the other. Since the two replicas are
        separated by custom SMP ops (which XLA does not know how to compile), this effectively prevents XLA from
        merging ops across subgraph replicas.

        Note that the specific control dependency structure depends on the chosen pipeline, and it is possible that a
        backward subgraph is a control dependency for a forward subgraph, and vice versa.
    """

    def __init__(self, enable):
        self.enable = enable

    def maybe_reroute_control_inputs(self):
        """If XLA is enabled, establish the control dependencies between subgraph replicas, for each subgraph."""

        if self.enable:
            input_ops = [
                op
                for op in tf.compat.v1.get_default_graph().get_operations()
                if op.type == smp.INPUT_OP
            ]
            output_ops = [
                op
                for op in tf.compat.v1.get_default_graph().get_operations()
                if op.type == smp.OUTPUT_OP
            ]

            for depth in range(state.pipeline.num_depths):
                if state.pipeline.is_bwd_depth(depth):
                    continue

                try:
                    bwd_depth = state.pipeline.fwd_to_bwd_depth(depth)
                    depths = [depth, bwd_depth]
                except KeyError:
                    depths = [depth]

                sg_input_ops = [
                    op
                    for op in input_ops
                    if state.pipeline.op_id_to_depth[op.get_attr("op_id")] in depths
                ]
                sg_output_ops = [
                    op
                    for op in output_ops
                    if state.pipeline.op_id_to_depth[op.get_attr("op_id")] in depths
                ]

                output_ticks = [[] for _ in range(state.pipeline.num_ticks())]
                input_ticks = [[] for _ in range(state.pipeline.num_ticks())]
                for op in sg_input_ops:
                    tick = state.pipeline.get_tick(op.get_attr("op_id"), op.get_attr("microbatch"))
                    input_ticks[tick].append(op)

                for op in sg_output_ops:
                    tick = state.pipeline.get_tick(op.get_attr("op_id"), op.get_attr("microbatch"))
                    output_ticks[tick].append(op)

                last_output_tick = -1
                for tick in range(state.pipeline.num_ticks()):
                    if len(input_ticks[tick]) > 0 and last_output_tick >= 0:
                        self._connect_ctrl_input(output_ticks[last_output_tick], input_ticks[tick])

                    if len(output_ticks[tick]) > 0:
                        last_output_tick = tick

    def _connect_ctrl_input(self, outputs, inputs):
        """Replace the placeholder tf.constant control input with the actual control input required by XLA. """
        bundled = bundle([op.inputs[0] for op in outputs])
        for inp in inputs:
            inp.inputs[1].op._update_input(1, bundled)
