# Standard Library
import copy
from contextlib import contextmanager

# Third Party
import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework.meta_graph import (
    create_meta_graph_def,
    import_scoped_meta_graph_with_return_elements,
)
from tensorflow.python.training.saver import export_meta_graph

# First Party
from smdistributed.modelparallel.tensorflow import core, state
from smdistributed.modelparallel.tensorflow.attrs import PEER_ATTR
from smdistributed.modelparallel.tensorflow.compile import CompileStatus
from smdistributed.modelparallel.tensorflow.graph_utils import _get_op_name
from smdistributed.modelparallel.tensorflow.serialization import BaseSerializedGraph
from smdistributed.modelparallel.tensorflow.utils import (
    is_ref_type,
    is_tensor_or_var,
    is_tensor_or_var_or_op,
)


class TraceGraph(tf.Graph):
    """
    A sub-class of tf.Graph. If an operation created in a smp.TraceGraph has is fed an
    input from another tf.Graph, a placeholder with the same signature as the input tensor
    is fed to the op instead, and the mapping between the original tensor and the placeholder
    is recorded.

    If an op is created inside smp.DistributedModel context which has an input tensor generated
    outside the smp.DistributedModel object, replaces the tensor with placeholder and maintains
    mapping from this placeholder to the original tensor.
    """

    def __init__(self):
        super(TraceGraph, self).__init__()
        self.step_input_map = {}
        self.model_input_map = {}
        self.model_ops = set()
        self.tensor_to_ph_mapping = {}

    def create_op(self, op_type, inputs, *args, **kwargs):

        ph_inputs = tf.nest.map_structure(self._convert_to_ph, inputs)

        # TODO while importing, we need to take into account the control_dependencies
        op = super(TraceGraph, self).create_op(op_type, ph_inputs, *args, **kwargs)

        if state.tracking_model:
            self.model_ops.add(op.name)

        return op

    def raise_error_if_ref_input(self, ts):
        if is_ref_type(ts):
            # this is not allowed because the placeholder that replaces it will have non-ref dtype, which
            # will not match the expected dtype of the ops that consume it.
            raise RuntimeError(
                f"RefVariable {ts} created outside smp.distributed_model is being accessed from within smp.distributed_model, which is not allowed. You can convert it to a ResourceVariable by feeding the option use_resource = True during variable creation."
            )

    def _convert_to_ph(self, ts):
        if is_tensor_or_var(ts) and ts.name in self.tensor_to_ph_mapping:
            return self.tensor_to_ph_mapping[ts.name]
        elif is_tensor_or_var(ts) and ts.op.graph != self:
            return self._get_placeholder(ts, self.step_input_map)
        elif is_tensor_or_var(ts) and state.tracking_model and ts.op.name not in self.model_ops:
            return self._get_placeholder(ts, self.model_input_map, add_to_model_ops=True)
        else:
            return ts

    def _get_placeholder(self, ts, input_map, add_to_model_ops=False):
        self.raise_error_if_ref_input(ts)
        ph = tf.placeholder(shape=ts.shape, dtype=ts.dtype)

        if add_to_model_ops:
            self.model_ops.add(ph.op.name)

        input_map[ph.name] = ts.name
        self.tensor_to_ph_mapping[ts.name] = ph

        return ph

    @contextmanager
    def trace(self):
        # prevent TF from complaining that the op inputs are not from the same graph
        # we will replace the tensors from other tf.Graphs with placeholders
        self.orig_assert_same_graph = ops._assert_same_graph
        self.orig_get_graph_from_inputs = ops._get_graph_from_inputs
        ops._assert_same_graph = lambda x, y: None
        ops._get_graph_from_inputs = self._get_graph_from_inputs
        with self.as_default():
            yield
        ops._assert_same_graph = self.orig_assert_same_graph
        ops._get_graph_from_inputs = self.orig_get_graph_from_inputs

    def _get_graph_from_inputs(self, op_input_list, graph=None):
        """
        Monkey-patch TF logic to determine which graph to place the operation in.
        By default, even when a default graph is given, TensorFlow places the operation
        in the graph that contains the input tensors. By returning self, we force each
        operation to be created in smp.TraceGraph, while inside smp.TraceGraph.trace() context.
        """
        return self


class SerializedGraphV1(BaseSerializedGraph):
    """
    TF1 implementation of SerializedGraph (see BaseSerializedGraph for details). Uses MetaGraphDef
    to keep track of the graph (GraphDef) and collections (CollectionDef). After the partition,
    updates the CollectionDefs based on the partitioning, by removing the unowned variables from
    the collections.
    """

    def __init__(self):
        super(SerializedGraphV1, self).__init__()
        self.meta_graph_def = None
        self.partitioned_meta_graph_def = None
        self.cond_to_device = {}  # TODO: fix

    def get_data(self):
        base_data = super(SerializedGraphV1, self).get_data()

        # serialize
        meta_graph_def_str = (
            self.meta_graph_def.SerializeToString() if self.meta_graph_def is not None else None
        )
        partitioned_meta_graph_def_str = (
            self.partitioned_meta_graph_def.SerializeToString()
            if self.partitioned_meta_graph_def is not None
            else None
        )

        return tuple(
            list(base_data)
            + [meta_graph_def_str, partitioned_meta_graph_def_str, self.cond_to_device]
        )

    def set_data(self, data):
        super(SerializedGraphV1, self).set_data(data[:-3])
        meta_graph_def_str, partitioned_meta_graph_def_str, self.cond_to_device = data[-3:]

        self.meta_graph_def = meta_graph_pb2.MetaGraphDef().ParseFromString(meta_graph_def_str)
        self.partitioned_meta_graph_def = meta_graph_pb2.MetaGraphDef().ParseFromString(
            partitioned_meta_graph_def_str
        )

    @contextmanager
    def track_graph(self):
        state.tracking_model = True
        with tf.variable_scope(f"SMPDistributedModel"):
            yield
        state.tracking_model = False

    def finalize(self, outputs):
        self.graph_def = self._filter_model(state.compile_graph.as_graph_def())

        self.structured_outputs = [
            [
                tf.nest.map_structure(self.replace_with_name, outputs)
                for mb in range(state.num_microbatches())
            ]
            for _ in range(state.cfg.pipeline_parallel_degree)
        ]

        flat_outputs = tf.nest.flatten(outputs)
        self.model_outputs = [
            [
                [x.name for x in flat_outputs if is_tensor_or_var_or_op(x)]
                for mb in range(state.num_microbatches())
            ]
            for _ in range(state.cfg.pipeline_parallel_degree)
        ]

        for ph, inp in state.compile_graph.model_input_map.items():
            inp_tensor = ops.get_default_graph().get_tensor_by_name(inp)
            self.model_inputs[ph] = (inp_tensor.shape.as_list(), inp_tensor.dtype)

        # reconstruct op assignment constraints
        self.op_to_device = {}

        # user-defined constraints
        for op_name, dev in state.op_to_device.items():
            if op_name in state.compile_graph.model_ops:
                self.op_to_device[op_name] = dev

        model_vars = [
            _get_op_name(v.name)
            for v in state.compile_graph.get_collection(tf.GraphKeys.VARIABLES)
            if _get_op_name(v.name) in state.compile_graph.model_ops
        ]

        # for compatibility of the SerializedGraph with TF2
        for var_name in model_vars:
            self.var_to_op[var_name] = var_name

        for var_name, dev in self.var_to_device.items():
            # if the variable is used in this graph
            if var_name in model_vars:
                # we do not need to explicitly place the auxiliary ops of variables (such as initializer etc.) because TF places a colocation constraint for those already, and the partitioner respects the colocation constraints in the GraphDef
                self.op_to_device[var_name] = dev

        self.meta_graph_def = export_meta_graph(graph=state.compile_graph)  # full graph

    def import_graph(self):
        input_map = copy.deepcopy(state.compile_graph.model_input_map)
        model_graph_def = copy.deepcopy(self.partitioned_graphs[core.pp_rank()][0])

        if state.compile_status == CompileStatus.TRAIN:
            model_graph_def = self.apply_compiled_attributes(
                model_graph_def, state.compiler.op_attr[0], state.compiler.op_metadata[0]
            )
        # translate peer attributes into global ranks
        for node in model_graph_def.node:
            if PEER_ATTR in node.attr and node.attr[PEER_ATTR].i != -1:
                node.attr[PEER_ATTR].i = core.get_pp_group()[node.attr[PEER_ATTR].i]

        self.partitioned_meta_graph_def = create_meta_graph_def(
            graph_def=model_graph_def,
            collection_list=[],
            saver_def=self.meta_graph_def.saver_def,
            meta_info_def=self.meta_graph_def.meta_info_def,
        )

        self.filter_collections()

        for ph, inp in input_map.items():
            if (
                state.compile_status == CompileStatus.TRAIN
                and inp in state.compile_graph.step_input_map
            ):
                mapped_inp = state.compile_graph.step_input_map[inp]
            else:
                mapped_inp = inp
            input_map[ph] = ops.get_default_graph().get_tensor_by_name(mapped_inp)

        # add dummy vars to input_map
        # if state.dummy_vars is None:
        state.dummy_vars = [
            tf.Variable(0.0, name=f"SMPDummy_{i}")
            for i in range(state.cfg.pipeline_parallel_degree)
        ]

        for dev in range(state.cfg.pipeline_parallel_degree):
            input_map[f"SMPDistributedModel/SMPDummy_{dev}"] = state.dummy_vars[dev]

        tf_output_names = self.model_outputs[core.pp_rank()][0]

        _, outputs = import_scoped_meta_graph_with_return_elements(
            self.partitioned_meta_graph_def,
            input_map=input_map,
            return_elements=tf_output_names,
            import_scope="import_0",
        )

        state.op_id_to_device = self.op_id_to_device

        return tf.nest.map_structure(
            lambda x: self.replace_with_tf_obj(x, outputs, tf_output_names),
            self.structured_outputs[core.pp_rank()][0],
        )

    def filter_collections(self):
        """ Go through collection_defs in full graph, remove variables/nodes/cond_contexts not assigned to this rank."""

        meta_graph_def = meta_graph_pb2.MetaGraphDef()
        meta_graph_def.CopyFrom(self.meta_graph_def)

        if ops.GraphKeys.GLOBAL_STEP in meta_graph_def.collection_def:
            # global_step will be available in the outer graph
            del meta_graph_def.collection_def[ops.GraphKeys.GLOBAL_STEP]

        for key in meta_graph_def.collection_def:
            col_def = meta_graph_def.collection_def[key]
            kind = col_def.WhichOneof("kind")

            if key == ops.GraphKeys.COND_CONTEXT:
                self._filter_cond_context(col_def.bytes_list.value)

            elif kind == "bytes_list":
                if key in ops.GraphKeys._VARIABLE_COLLECTIONS:
                    self._filter_variable_collection(key, col_def.bytes_list.value)
                else:
                    assert False, f"unknown type {key}, {col_def}"

            else:
                field = getattr(col_def, kind)
                if kind == "node_list":
                    self._filter_node_list(field.value)
                elif kind == "int64_list":
                    continue
                else:
                    assert False, f"unknown kind {kind}"

        self.partitioned_meta_graph_def.collection_def.MergeFrom(meta_graph_def.collection_def)

    def _filter_variable_collection(self, key, var_list):
        to_remove = []

        for i, value in enumerate(var_list):
            proto_type = ops.get_collection_proto_type(key)
            proto = proto_type()
            proto.ParseFromString(value)

            var_name = _get_op_name(proto.variable_name)

            if var_name not in self.var_to_device or self.var_to_device[var_name] != core.pp_rank():
                to_remove.append(i)

        self._remove_elements(var_list, to_remove)

    def _filter_node_list(self, node_list):
        to_remove = []
        op_set = set([n.name for n in self.partitioned_meta_graph_def.graph_def.node])

        for i, value in enumerate(node_list):
            if _get_op_name(value) not in op_set:
                to_remove.append(i)

        self._remove_elements(node_list, to_remove)

    def _filter_cond_context(self, context_list):
        to_remove = []

        for i, value in enumerate(context_list):
            proto_type = ops.get_collection_proto_type(ops.GraphKeys.COND_CONTEXT)
            proto = proto_type()
            proto.ParseFromString(value)
            if (
                proto.context_name not in self.cond_to_device
                or self.cond_to_device[proto.context_name] != core.pp_rank()
            ):
                to_remove.append(i)

        self._remove_elements(context_list, to_remove)

    def _filter_model(self, graph_def):
        to_remove = []
        for i, node_def in enumerate(graph_def.node):
            if node_def.name not in state.compile_graph.model_ops:
                to_remove.append(i)

        self._remove_elements(graph_def.node, to_remove)

        return graph_def

    def _remove_elements(self, lst, to_remove):
        for i in range(len(to_remove) - 1, -1, -1):
            lst.pop(to_remove[i])
