# Standard Library
import re
from contextlib import contextmanager
from enum import Enum
from functools import lru_cache

# Third Party
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, node_def_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework.c_api_util import ApiDefMap

# First Party
from smdistributed.modelparallel.tensorflow import graph_def_editor as gde
from smdistributed.modelparallel.tensorflow.attrs import DTYPE_ATTR, OUTPUT_SHAPE_ATTR, SHAPE_ATTR
from smdistributed.modelparallel.tensorflow.graph_def_editor.tensor import Tensor as gde_Tensor
from smdistributed.modelparallel.tensorflow.graph_def_editor.util import python_type_to_attr_value
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.state_mod import state

# TF1.x variable types
VARIABLE_TYPES = ["Variable", "VariableV2", "AutoReloadVariable", "VarHandleOp", "_VarHandlesOp"]


@lru_cache(maxsize=1)
def _get_var_consumer_ops():
    """ Automatically populate the ops that consume a Resource or Ref type using TF OpDefRegistry """
    api_def_map = ApiDefMap()

    consumer_ops = []
    for op_name in api_def_map.op_names():
        op_def = api_def_map.get_op_def(op_name)
        for arg in op_def.input_arg:
            if arg.type == tf.resource.as_datatype_enum or arg.is_ref:
                consumer_ops.append(op_name)
                break
    assert len(consumer_ops) > 0, "Missing variable consumer ops."

    return consumer_ops


def make_tensor_shape(shape):
    def _make_none(x):
        return None if x is None or x < 0 else x

    _shp = [_make_none(dim) for dim in shape]
    return tf.TensorShape(_shp)


def get_num_elements_in_var(shape):
    prod = 1
    for dim in shape:
        prod *= dim
    return prod


def get_tensor_shape(tensor, op_to_layer):
    """
    Getting tensor shape from 2 methods:
    1. Profiling result from smdebug
    2. Existing shape from the graph

    If can't find the tensor shape, return None
    """

    if state.tf_version == 2:

        def _exist(layer_name, tensor_type, index):
            """Determine whether a tensor exists in our profiling result"""
            return (
                layer_name in state.serialized_graph.tensor_shapes
                and state.serialized_graph.tensor_shapes[layer_name][tensor_type] != None
                and len(state.serialized_graph.tensor_shapes[layer_name][tensor_type]) > index
            )

        related_nodes = tensor.consumers()
        related_nodes.add(tensor.node)

        for node in related_nodes:
            if tensor in node.inputs:
                tensor_type = "inputs"
                index = node.inputs.index(tensor)
            else:
                tensor_type = "outputs"
                index = node.outputs.index(tensor)

            if node.name in op_to_layer:
                layer_name = op_to_layer[node.name]
                layer_name = layer_name[layer_name.rfind("/") + 1 :]
            else:
                layer_name = None

            if _exist(layer_name, tensor_type, index):
                tensor_shape = state.serialized_graph.tensor_shapes[layer_name][tensor_type][index]
                get_logger().info(
                    f"Getting tensor {tensor.name} shape {tensor_shape} as {tensor_type} from layer {layer_name}"
                )
                return tensor_shape
    else:
        tensor_name = tensor.name.replace("SMPDistributedModel/", "")
        if tensor_name in state.serialized_graph.tensor_shapes:
            shape = state.serialized_graph.tensor_shapes[tensor_name]
            get_logger().info(f"Getting tensor {tensor_name} shape {shape} from profile")
            return shape

    tensor_shape = tensor.shape.as_list()
    # Don't get tensor shape from graph if default_tensor_size is not updated from profiling
    if None in tensor_shape or state.serialized_graph.default_tensor_size == 1:
        tensor_shape = None

    return tensor_shape


def is_var_consumer(node):
    is_consumer_type = get_op_type(node) in _get_var_consumer_ops()
    is_var_read_op_v1 = get_op_type(node) == "Identity" and node.name.endswith("/read")
    is_initial_value_v1 = get_op_type(node) == "Const" and node.name.endswith("/initial_value")

    return is_consumer_type or is_var_read_op_v1 or is_initial_value_v1


def is_var(node):
    if isinstance(node, node_def_pb2.NodeDef):
        return node.op in VARIABLE_TYPES or (
            node.op == "Placeholder" and node.attr["dtype"].type == tf.resource.as_datatype_enum
        )
    if isinstance(node, gde.Node):
        return node.op_type in VARIABLE_TYPES or (
            node.op_type == "Placeholder" and node.outputs[0].dtype == tf.resource
        )
    if isinstance(node, tf.Operation):
        return node.type in VARIABLE_TYPES or (
            node.type == "Placeholder" and node.outputs[0].dtype == tf.resource
        )

    raise TypeError(f"{node} is an unrecognized type!")


def get_tensor_size(tensor_shape):
    if tensor_shape == None:
        return state.serialized_graph.default_tensor_size

    if isinstance(tensor_shape, tf.Tensor) or isinstance(tensor_shape, gde_Tensor):
        tensor_shape = tensor_shape.shape.as_list()

    val = 1
    for s in tensor_shape:
        val *= int(s)
    return val if val > 0 else 1  # in case size 0 tensor


def get_tensor(g, sg_tensor):
    node = g.get_node_by_name(sg_tensor.node)
    if sg_tensor.io == IO.INPUT:
        return node.inputs[sg_tensor.index]
    elif sg_tensor.io == IO.CONTROL_INPUT:
        return node.control_inputs[sg_tensor.index]
    else:
        return node.outputs[sg_tensor.index]


def get_op_type(x):
    if isinstance(x, node_def_pb2.NodeDef):
        return x.op
    if isinstance(x, gde.Node):
        return x.op_type
    if isinstance(x, tf.Operation):
        return x.type
    raise TypeError(f"Unsupported type {type(x)}.")


def is_placeholder(node):
    return node.op_type == "Placeholder"


def is_const(node):
    return node.op_type == "Const"


def node_consumers(g, node):
    cons = []
    for n in g.nodes:
        control_input_names = [c.name for c in n.control_inputs]
        if node.name in control_input_names:
            cons.append(n)
    return cons


class IO(Enum):
    INPUT = 0
    CONTROL_INPUT = 1
    OUTPUT = 2


def _get_op_name(tensor_name):
    _num_removed = re.sub(r"\d+$", "", tensor_name)
    if _num_removed.endswith(":"):
        return _num_removed[:-1]
    else:
        return tensor_name


def get_op(obj):
    return obj.node if isinstance(obj, gde_Tensor) else obj


def make_placeholder_graph_def(graph_def, name, shape, dtype):
    ph_node = graph_def.node.add()
    ph_node.name = _get_op_name(name)
    ph_node.op = "Placeholder"
    ph_node.attr[DTYPE_ATTR].CopyFrom(attr_value_pb2.AttrValue(type=dtype.as_datatype_enum))
    ph_node.attr[SHAPE_ATTR].CopyFrom(python_type_to_attr_value(shape))
    ph_node.attr[OUTPUT_SHAPE_ATTR].CopyFrom(python_type_to_attr_value(shape))


def update_consumers(g, node_name, new_node):
    node = g.get_node_by_name(node_name)
    for out_idx, out in enumerate(node.outputs):
        update_tensor_consumers(g, out, new_node, out_idx)


def update_tensor_consumers(g, tensor, new_node, out_idx):
    for cons in tensor.consumers():
        for idx, inp in enumerate(cons.inputs):
            if tensor.name == inp.name:
                cons_ref = g.get_node_by_name(cons.name)
                if new_node.name != cons_ref.name:
                    cons_ref.replace_input(idx, new_node.outputs[out_idx])
                    break


def get_item_in_graph(g, item):
    if isinstance(item, gde.Node):
        return g.get_node_by_name(item.name)
    else:
        op_name = item.node.name
        node = g.get_node_by_name(op_name)
        for out in node.outputs:
            if out.name == item.name:
                return out


class SMPCpuContext:
    """
    A class that contains contextmanagers that place the model on CPU
    - cpu_var_creator: Using this context to place all variables on CPU
    - cpu_context: Using this context to place the entire model on CPU
    """

    @staticmethod
    @contextmanager
    def cpu_var_creator():
        """ A context manager that ensures that all variables and ops constructed inside
        will be placed on CPU. """
        with tf.variable_creator_scope(SMPCpuContext._cpu_var_scope):
            yield

    @staticmethod
    @contextmanager
    def cpu_context():
        """ A context manager that ensures that all variables and ops constructed inside
        will be placed on CPU. """
        with SMPCpuContext._patch_device_context():
            with ops.device("/cpu:0"):
                yield

    @staticmethod
    def _cpu_var_scope(next_creator, **kwargs):
        with SMPCpuContext._patch_device_context():
            with ops.device("/cpu:0"):
                return next_creator(**kwargs)

    @staticmethod
    @contextmanager
    def _patch_device_context():
        """
        To ensure that TF will not internally override the
        tf.device('cpu') context given in variable_creator_scope, we patch tf.device
        to always choose CPU regardless of the argument.
        """
        SMPCpuContext.org_device = ops.device
        ops.device = SMPCpuContext._cpu_device
        yield
        ops.device = SMPCpuContext.org_device

    @staticmethod
    @contextmanager
    def _cpu_device(dev):
        with SMPCpuContext.org_device("/cpu:0"):
            yield
