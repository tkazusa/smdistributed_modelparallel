# Standard Library
import pickle
import sys
import traceback
from contextlib import contextmanager

# Third Party
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops

# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup, RankType
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.tensorflow import core, state
from smdistributed.modelparallel.tensorflow.attrs import (
    EXPECTED_SHAPE_ATTR,
    LINK_ID_ATTR,
    OP_ID_ATTR,
    OUT_TYPE_ATTR,
    OUTPUT_SHAPE_ATTR,
    PEER_ATTR,
    TICK_ATTR,
)
from smdistributed.modelparallel.tensorflow.graph_def_editor.util import python_type_to_attr_value
from smdistributed.modelparallel.tensorflow.graph_utils import (
    SMPCpuContext,
    get_tensor_size,
    make_tensor_shape,
)
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.utils import is_tensor_or_var_or_op, mainify_name

_TF_OBJECT_PREFIX = "__tf_obj__"


class BaseSerializedGraph:
    """
    A serialized representation of the model graph and other metadata useful for partitioning.
    A SerializedGraph object is created after tracing the model graph an extracting the relevant
    metadata. AutoPartitioner operates on the SerializedGraph, and the partitioning output is
    represented on the same SerializedGraph object.

    See SerializedGraphV1 and SerializedGraphV2 for specific implementations for TF1.x and TF2.x.
    """

    def __init__(self):
        # A dict mapping tensor names that are fed as arguments to smp.DistributedModel.__call__,
        # to their (shape, dtype) tuple
        self.model_inputs = {}

        # List of tensor names returned by smp.DistributedModel.__call__. Includes tensors returned
        # as part of a nested structure. Consists of nested lists indexed by device and microbatch.
        self.model_outputs = []

        # The nested structure output from smp.DistributedModel.__call__. Nested in lists indexed by
        # device and microbatch. TensorFlow objects are replaced by their names.
        self.structured_outputs = []

        # The GraphDef protobuf representing the model function body.
        self.graph_def = None

        # Mapping from variable to its assigned device. Persistent across re-partitioning passes. Reassignments are not allowed.
        self.var_to_device = {}

        # Initial constraints for op assignments. This is an *input* to the partitioning algorithm, not its output.
        self.op_to_device = {}

        # A mapping from SmpInput/SmpOutput op_id to the assigned device id.
        self.op_id_to_device = {}

        # OpId Generator object
        self.op_id_gen = None

        # A dict mapping op_ids to SMP op NodeDefs
        self.smp_ops = {}

        # A mapping from the variable names in the model to the corresponding placeholder name in the GraphDef. Only includes variables that are part of the current GraphDef - not persistent across re-partitioning passes.
        self.var_to_op = {}

        # Doubly nested lists of graph_pb2.GraphDef objects. self.partitioned_graphs[dev][mb]
        # represents the GraphDef assigned to device `dev` for microbatch `mb`
        self.partitioned_graphs = []

        # The partition_file to dump the SerializedGraph into
        self.partition_file = f"partition_file_0.pkl"

        # tensor shapes from profiling
        # For TF2 format will be {layer_name: {"inputs":inputs_shapes, "outputs":outputs_shapes}}
        # For TF1 format will be {tensor_name: shape}
        self.tensor_shapes = {}

        # The default tensorshape if the tensor can not be profiled
        self.default_tensor_size = 1

        # All internal profiling related members
        self.is_profiling = False
        self.raw_profiling_results = {}
        self.aggregation_method = None
        self.skip_profile = False
        self.has_profiled = False
        self.has_partitioned = False
        self.has_aggregated = False
        self.has_broadcasted = False

    def get_data(self):
        """ Return a tuple that represents the current state of the SerializedGraph."""
        self.op_id_gen = state.op_id_gen
        return (
            self.model_inputs,
            self.model_outputs,
            self.structured_outputs,
            self.graph_def,
            self.var_to_op,
            self.op_id_to_device,
            self.partitioned_graphs,
            self.var_to_device,
            self.op_to_device,
            self.smp_ops,
            self.op_id_gen,
        )

    def set_data(self, data):
        """ Set the current state of the SerializedGraph as the input tuple `data`."""
        self.model_inputs, self.model_outputs, self.structured_outputs, self.graph_def, self.var_to_op, self.op_id_to_device, self.partitioned_graphs, self.var_to_device, self.op_to_device, self.smp_ops, self.op_id_gen = (
            data
        )
        state.op_id_gen = self.op_id_gen

    def save(self):
        """ Dump the SerializedGraph into a file."""

        s = pickle.dumps(self.get_data())
        with open(self.partition_file, "wb") as f:
            f.write(s)

    def load(self):
        """ Load a SerializedGraph from a file. """

        with open(self.partition_file, "rb") as f:
            s = f.read()
        res = pickle.loads(s)
        self.set_data(res)

    def replace_with_name(self, x):
        """ If the input is a TensorFlow object, return its name"""
        if is_tensor_or_var_or_op(x):
            return _TF_OBJECT_PREFIX + mainify_name(x.name)
        else:
            return x

    def replace_with_tf_obj(self, elem, output_tensors, tf_output_names):
        """ Given `elem`, a TF-object name that is output by `self.replace_with_name`,
        return the TF object with the given name."""

        if isinstance(elem, str) and elem.startswith(_TF_OBJECT_PREFIX):
            ind = tf_output_names.index(elem[len(_TF_OBJECT_PREFIX) :])
            return output_tensors[ind]
        else:
            return elem

    def apply_compiled_attributes(self, graph_def, op_attr, op_metadata):
        """ Set the SMP op attributes gathered from compiler. """

        for node_def in graph_def.node:
            if OP_ID_ATTR in node_def.attr:
                op_id = node_def.attr[OP_ID_ATTR].i
                tick, peer, link_id = op_attr[op_id]
                node_def.attr[LINK_ID_ATTR].i = link_id
                if PEER_ATTR in node_def.attr:
                    shape, dtype = op_metadata[op_id]
                    shape_attr = python_type_to_attr_value([make_tensor_shape(shape)])

                    node_def.attr[TICK_ATTR].i = tick
                    node_def.attr[PEER_ATTR].i = peer

                    node_def.attr[OUT_TYPE_ATTR].CopyFrom(
                        attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
                    )
                    node_def.attr[EXPECTED_SHAPE_ATTR].CopyFrom(
                        python_type_to_attr_value(list(shape))
                    )
                    node_def.attr[OUTPUT_SHAPE_ATTR].CopyFrom(shape_attr)
        return graph_def

    def get_node_name_by_op_id(self, op_id):
        return self.smp_ops[op_id]

    @contextmanager
    def profiling_context(self, smd=None, graph=None):
        """
        The context used for profile:
        1. Decide if the profiling should be run
        2. Catch the error if there is any during the profiling
        3. Reset the smdebug hook back to user's hook for each profile
        4. Set all related flags
        """
        should_profile = (
            not self.skip_profile and not self.has_partitioned and smd != None and core.rank() == 0
        )
        try:
            if smd != None:
                # Users may specify their own hook, since smdebugger hook is a singleton,
                # backup this hook and restore once the profiling is over.
                user_hook = smd.get_hook("keras", create_if_not_exists=False)
            self.is_profiling = True
            yield should_profile
        except Exception as expt:
            exc_info = sys.exc_info()
            get_logger().warning(f"Sagemaker debugger fails with Exception:\n")
            traceback.print_exception(*exc_info)
            get_logger().warning(f"SMP will run without profiling")
            self.skip_profile = True
            if graph != None:
                ops._assert_same_graph = graph.orig_assert_same_graph
                ops._get_graph_from_inputs = graph.orig_get_graph_from_inputs
            if hasattr(SMPCpuContext, "org_device"):
                ops.device = SMPCpuContext.org_device
        finally:
            self.is_profiling = False
            self.has_profiled = True
            if self.skip_profile:
                get_logger().warning(
                    f"Profiling is skipped due to previous Sagemaker debugger failure"
                )
            elif self.has_partitioned:
                get_logger().warning(
                    f"Profiling is skipped due to model.profile() is called after regular model call and model is already partitioned"
                )
            elif smd == None:
                get_logger().warning(
                    "Sagemaker debugger is not installed, profiling is not supported"
                )
            if core.pp_rank() != 0:
                pass
            else:
                if smd != None:
                    smd.del_hook()
                    # Restore user's hook if there is any
                    if user_hook != None:
                        smd.set_hook(user_hook)

    def add_tensor_shape(self, tensor_name, shape, tag=None):
        if tag == None:
            self.tensor_shapes[tensor_name] = shape
        else:
            if tensor_name not in self.tensor_shapes:
                self.tensor_shapes[tensor_name] = {tag: shape}
            else:
                self.tensor_shapes[tensor_name][tag] = shape

    def should_aggregate(self):
        return core.rank() == 0 and not self.has_aggregated and self.has_profiled

    def broadcast_profile_result(self):
        if not self.has_broadcasted and self.has_profiled:
            if core.rank() == 0:
                state.comm.broadcast(
                    (self.tensor_shapes, self.default_tensor_size), group=CommGroup.WORLD
                )
            else:
                self.tensor_shapes, self.default_tensor_size = state.comm.recv_from(
                    0, RankType.WORLD_RANK
                )
            self.has_broadcasted = True

    def aggregate_profile_results(self):
        """Aggregate the tensor shapes from multiple steps, update the default tensor size"""
        tensor_size = []

        def _aggregate(tensor_shapes, axis=None, aggregation_method=None):
            if aggregation_method == None:
                aggregation_method = self.aggregation_method
            if aggregation_method == "median":
                return np.median(tensor_shapes, axis=axis)
            elif aggregation_method == "mean":
                return np.mean(tensor_shapes, axis=axis)
            elif aggregation_method == "max":
                return np.max(tensor_shapes, axis=axis)
            elif aggregation_method == "p95":
                return np.percentile(tensor_shapes, 95, axis=axis)
            else:
                raise ValueError("aggregation only support median, mean, max, p95")

        for name, shapes in self.raw_profiling_results.items():
            if state.tf_version == 2:
                input_shapes = shapes["inputs"]
                output_shapes = shapes["outputs"]
                if len(input_shapes) > 0:
                    agg_input = _aggregate(input_shapes, axis=0)
                    agg_input = agg_input.tolist()
                    for shape in agg_input:
                        tensor_size.append(get_tensor_size(shape))
                else:
                    agg_input = None
                if len(output_shapes) > 0:
                    agg_output = _aggregate(output_shapes, axis=0)
                    agg_output = agg_output.tolist()
                    for shape in agg_output:
                        tensor_size.append(get_tensor_size(shape))
                else:
                    agg_output = None
                self.add_tensor_shape(name, agg_input, tag="inputs")
                self.add_tensor_shape(name, agg_output, tag="outputs")
            else:
                agg_shape = _aggregate(shapes, axis=0)
                agg_shape = agg_shape.tolist()
                tensor_size.append(get_tensor_size(agg_shape))
                self.add_tensor_shape(name, agg_shape)

        # Update the default tensor size based on the all known tensor sizes
        if len(tensor_size) > 0:
            self.default_tensor_size = int(_aggregate(tensor_size, aggregation_method="median"))
            get_logger().info(f"Update default tensor size to {self.default_tensor_size}")
