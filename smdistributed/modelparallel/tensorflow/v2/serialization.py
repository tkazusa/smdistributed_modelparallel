# Standard Library
import copy
from contextlib import contextmanager

# Third Party
import tensorflow as tf
from tensorflow.python.framework import ops

# First Party
from smdistributed.modelparallel.tensorflow import core, state
from smdistributed.modelparallel.tensorflow.attrs import PEER_ATTR
from smdistributed.modelparallel.tensorflow.compile import CompileStatus
from smdistributed.modelparallel.tensorflow.graph_utils import _get_op_name
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.pybind import inline_funclib
from smdistributed.modelparallel.tensorflow.serialization import BaseSerializedGraph
from smdistributed.modelparallel.tensorflow.utils import (
    cache_key,
    get_create_op,
    get_tensor_id,
    is_tensor_or_var,
    list_insert,
    mainify_name,
    set_create_op,
)
from smdistributed.modelparallel.tensorflow.v2.utils import get_layers


class PartitionCache:
    """
    A cache of partitioned SerializedGraph objects, keyed by tf.function cache keys. This
    prevents duplicate partition requests from the helper process.
    """

    def __init__(self):
        self._cache = {}

    def put(self, func, args, kwargs, serialized_graph, signature=None):
        key = self._get_key(func, args, kwargs, signature)

        if key in self._cache:
            raise ValueError("Key already exists in the partition cache.")

        self._cache[key] = state.serialized_graph.get_data()

    def has_key(self, func, args, kwargs, signature=None):
        key = self._get_key(func, args, kwargs, signature)
        return key in self._cache

    def load(self, func, args, kwargs, signature=None):
        if not self.has_key(func, args, kwargs, signature):
            raise ValueError("Key not found in partition cache.")

        key = self._get_key(func, args, kwargs, signature)

        _sg_data = self._cache[key]
        state.serialized_graph.set_data(_sg_data)

    def _get_key(self, func, args, kwargs, signature=None):
        _arg_prefix = [0, False, False]
        return cache_key(func, args, kwargs, signature, prefix=_arg_prefix)


class SerializedGraphV2(BaseSerializedGraph):
    """
    TF2.x implementation of SerializedGraph (see BaseSerializedGraph for details). Contains
    additional logic to handle Keras layers.
    """

    def __init__(self):
        super(SerializedGraphV2, self).__init__()

        # A mapping from the layer names in the model to input shape tuple, or None.
        # Before importing the graph, the layers assigned to this rank must be built
        # with these input shapes as an argument to Layer.build() function.
        self.layer_shapes = {}

        # Set of all layer names in the model
        self.layers = set()

        # List of sets of layers, where i'th set contains the layer names assigned to device i
        self.partitioned_layers = [set() for _ in range(state.cfg.pipeline_parallel_degree)]

        # A stack that maintains the nested layer scopes
        self.layer_stack = []

        # Mapping from op name to the name of the layer that contains the op
        self.op_to_layer = {}

        # A mapping from variable placeholder to corresponding variable shape
        self.var_shapes = {}

        # Layer names in the order they appear in get_layers(model)
        self.ordered_layer_names = []

        # Mapping from Python ID of layer in CPU model to layer name
        # This is not broadcast across ranks since it's only needed during
        # tracing, and the Python ID's will be local to the process
        self.id_to_name = {}

    def get_data(self):
        base_data = super(SerializedGraphV2, self).get_data()
        return tuple(
            list(base_data)
            + [
                self.layer_shapes,
                self.layers,
                self.partitioned_layers,
                self.layer_stack,
                self.op_to_layer,
                self.var_shapes,
                self.ordered_layer_names,
            ]
        )

    def set_data(self, data):
        super(SerializedGraphV2, self).set_data(data[:-7])
        self.layer_shapes, self.layers, self.partitioned_layers, self.layer_stack, self.op_to_layer, self.var_shapes, self.ordered_layer_names = data[
            -7:
        ]

    def _is_layer_stateless(self, layer):
        return len(layer.weights) == 0

    def _get_layer_name(self, layer):
        if self._is_layer_stateless(layer):
            return mainify_name(ops.get_name_scope())
        else:
            return self.id_to_name[id(layer)]

    @contextmanager
    def track_graph(self, model, args, kwargs):
        """Keep track of the layers built while constructing the model graph, and
        the input shapes they are built with. This will later be used to re-construct
        the model graph."""

        def wrap_maybe_build(maybe_build, layer):
            def wrapper(*args):
                # add only if we have not seen this layer before - this takes into account layer reuse
                # OR if the layer is stateless - we can treat reused stateless layers as separate layers -
                # doing otherwise seems to hurt performance
                if id(layer) not in self.id_to_name or self._is_layer_stateless(layer):
                    name = mainify_name(ops.get_name_scope())
                    list_insert(self.ordered_layer_names, state.layer_order_map[id(layer)], name)
                    self.id_to_name[id(layer)] = name
                    self.layers.add(name)
                return maybe_build(*args)

            return wrapper

        # wrap keras.Layer.build to keep track of the input_shapes
        def wrap_build(build, layer):
            def wrapper(*args):
                name = self._get_layer_name(layer)
                self.layer_shapes[name] = args[0]
                return build(*args)

            # TF internally checks this attribute to determine if the layer has overridden build method:
            # https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/engine/base_layer.py#L2411
            # We preserve this attribute in the wrapper to avoid misleading TF into believing that
            # the layer has an overridden build method
            if hasattr(layer.build, "_is_default"):
                wrapper._is_default = True

            return wrapper

        def wrap_call(call_fn, layer):
            def wrapper(*args, **kwargs):
                name = self._get_layer_name(layer)
                self.layer_stack.append(name)
                try:
                    out = call_fn(*args, **kwargs)
                finally:
                    self.layer_stack.pop()
                return out

            return wrapper

        model_layers = get_layers(model)
        orig_layer_info = []
        for layer in model_layers:
            # saving original layer
            orig_layer_info.append((layer, layer.build, layer._maybe_build, layer.call))

            layer.build = wrap_build(layer.build, layer)
            layer._maybe_build = wrap_maybe_build(layer._maybe_build, layer)
            layer.call = wrap_call(layer.call, layer)

        org_create_op = get_create_op()

        def create_op_in_layer(*args, **kwargs):
            op = org_create_op(*args, **kwargs)
            if len(self.layer_stack) > 0:
                self.op_to_layer[mainify_name(op.name)] = self.layer_stack[-1]
            return op

        set_create_op(create_op_in_layer)

        with tf.name_scope(f"SMPDistributedModel"):
            yield

        # in case the last layers in `model_layers` are never called during the model call
        self.ordered_layer_names.extend(
            [None for _ in range(len(model_layers) - len(self.ordered_layer_names))]
        )

        set_create_op(org_create_op)

        # restoring original layer info
        for layer_info in orig_layer_info:
            layer_info[0].build = layer_info[1]
            layer_info[0]._maybe_build = layer_info[2]
            layer_info[0].call = layer_info[3]

    def finalize(self, graph):
        """ Called by smp.DistributedModel.__call__ to finalize the SerializedGraph, after
        the model graph was built. Records the model GraphDef, model inputs and model outputs."""

        # graph inputs
        self.model_inputs = {}
        inputs = graph.inputs[: -len(graph.captures)] if len(graph.captures) > 0 else graph.inputs
        for inp in inputs:
            self.model_inputs[mainify_name(inp.name)] = (inp.shape, inp.dtype)

        # variable mapping
        self.var_to_op = {}
        self.var_sizes = {}
        for v in graph.variables:
            op_name = mainify_name(graph._captures[get_tensor_id(v.handle)][1].name)
            self.var_to_op[mainify_name(v.name)] = op_name
            self.var_shapes[_get_op_name(op_name)] = v.shape.as_list()

        # returned outputs
        self.model_outputs = [
            [
                [mainify_name(n.name) for n in graph.outputs]
                for mb in range(state.num_microbatches())
            ]
            for _ in range(state.cfg.pipeline_parallel_degree)
        ]
        self.structured_outputs = [
            [
                tf.nest.map_structure(self.replace_with_name, graph.structured_outputs)
                for mb in range(state.num_microbatches())
            ]
            for _ in range(state.cfg.pipeline_parallel_degree)
        ]

        # GraphDef
        self.graph_def = inline_funclib(graph.as_graph_def(add_shapes=True))

        # reconstruct op assignment constraints
        self.op_to_device = {}

        # user-defined constraints
        op_set = set([mainify_name(op.name) for op in graph.get_operations()])
        for op_name, dev in state.op_to_device.items():
            main_op_name = mainify_name(op_name)
            if main_op_name in op_set:
                self.op_to_device[main_op_name] = dev

        # Existing variable assignments. Required for 2 use cases.
        # 1. Comes after user-specified constraints, since if the user specification conflicts with this,
        # we have to override it (this conflict normally should not happen).
        # 2. Auto-partition constraints. This is typically required for re-partitioning (eg. eval step after train step)
        for var_name, dev in self.var_to_device.items():
            # if the variable is used in this graph
            if var_name in self.var_to_op:
                op_name = self.var_to_op[mainify_name(var_name)]
                self.op_to_device[op_name] = dev

    def import_graph(self, model, args, kwargs, microbatch):
        """ Import the model represented by this SerializedGraph into the graph the call is made from. First builds
        the layers assigned to this device, then imports the GraphDef, while attaching the inputs and variables to the
        appropriate placeholders in the GraphDef. """
        input_map = {}
        tf.nest.map_structure(lambda x: self.add_to_map(x, input_map), args)
        tf.nest.map_structure(lambda x: self.add_to_map(x, input_map), kwargs)

        # build the layers that are needed
        assigned_layers = state.serialized_graph.partitioned_layers[core.pp_rank()]
        for i, layer in enumerate(get_layers(model)):
            # get the layer name within the original enclosing scope
            name = self.ordered_layer_names[i]

            # if this layer was never called, name will be None
            if name is None:
                continue

            layer_assigned = name in assigned_layers
            layer_unbuilt = not layer.built
            build_defined = not hasattr(layer.build, "_is_default")

            if build_defined and layer_unbuilt and layer_assigned:
                layer_shape = state.serialized_graph.layer_shapes[name]

                with tf.name_scope(name):
                    if layer_shape is not None:
                        shape = tuple(layer_shape)

                        get_logger().debug(
                            f"{core.pp_rank()} building layer {name} with shape {shape}"
                        )
                        layer.build(shape)
                    else:
                        get_logger().debug(
                            f"{core.pp_rank()} building layer {name} with shape None"
                        )
                        layer.build(None)

        for layer in model.dummy_layers:
            if not layer.built:
                layer.build(None)

        # map variables - model.variables will only include the variables in the layers that were built
        model_graph_def = copy.deepcopy(self.partitioned_graphs[core.pp_rank()][microbatch])

        for var in model.variables:
            if "SMPDummy" not in var.name and var.name in self.var_to_op:
                input_map[self.var_to_op[var.name]] = tf.convert_to_tensor(var.handle)

        for dev in range(state.cfg.pipeline_parallel_degree):
            input_map[f"SMPDistributedModel/SMPDummy_{dev}"] = model.dummy_layers[dev](None)

        # get list of tf outputs
        tf_output_names = self.model_outputs[core.pp_rank()][microbatch]

        get_logger().debug(f"Input_map: {input_map}")
        get_logger().debug(f"Returning: {tf_output_names}")

        # set smp op attributes
        if state.compile_status == CompileStatus.TRAIN:
            model_graph_def = self.apply_compiled_attributes(
                model_graph_def,
                state.compiler.op_attr[microbatch],
                state.compiler.op_metadata[microbatch],
            )

        # translate peer attributes into global ranks
        for node in model_graph_def.node:
            if PEER_ATTR in node.attr and node.attr[PEER_ATTR].i != -1:
                node.attr[PEER_ATTR].i = core.get_pp_group()[node.attr[PEER_ATTR].i]

        outputs = tf.graph_util.import_graph_def(
            model_graph_def,
            input_map=input_map,
            return_elements=tf_output_names,
            name=f"import_{microbatch}",
        )
        structured_output = tf.nest.map_structure(
            lambda x: self.replace_with_tf_obj(x, outputs, tf_output_names),
            self.structured_outputs[core.pp_rank()][microbatch],
        )
        get_logger().debug(f"Structured output: {structured_output}")

        state.op_id_to_device = self.op_id_to_device

        return structured_output

    def add_to_map(self, elem, input_map):
        if is_tensor_or_var(elem):
            input_map[elem.name] = elem
