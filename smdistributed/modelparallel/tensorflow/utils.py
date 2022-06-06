# Standard Library
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

# Third Party
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
from tensorflow.python.eager.function import CacheKey
from tensorflow.python.framework import load_library, ops, versions
from tensorflow.python.platform import resource_loader

# First Party
from smdistributed.modelparallel.backend.collectives import RankType, TransactionIdentifier
from smdistributed.modelparallel.tensorflow import core
from smdistributed.modelparallel.tensorflow import graph_def_editor as gde
from smdistributed.modelparallel.tensorflow.attrs import (
    XLA_COMPILE_ATTR,
    XLA_INTERNAL_SCOPE_ATTR,
    XLA_SCOPE_ATTR,
)
from smdistributed.modelparallel.tensorflow.graph_def_editor.tensor import Tensor as gde_Tensor
from smdistributed.modelparallel.tensorflow.graph_def_editor.util import attr_value_to_python_type
from smdistributed.modelparallel.tensorflow.graph_utils import is_var, is_var_consumer
from smdistributed.modelparallel.tensorflow.pybind import (
    backward_walk,
    get_nodes_by_names,
    get_ops_of_type,
    set_graph,
)
from smdistributed.modelparallel.tensorflow.state_mod import state

try:
    from tensorflow.python.eager.function import _make_input_signature_hashable
except ImportError:
    # this is a no_op in TF2.0
    _make_input_signature_hashable = lambda x: x


INPUT_OP = "SmpInput"
OUTPUT_OP = "SmpOutput"
ANY_OP = "SmpAnyOp"
ALLGATHER_OP = "SmpAllgather"

_TRACE_MODEL_PREFIX = "trace_smp_distributed_model"
_PROFILE_MODEL_PREFIX = "profile_smp_distributed_model"

DTYPE_TO_BYTES = {
    tf.float16: 2,
    tf.float32: 4,
    tf.float64: 8,
    tf.bfloat16: 2,
    tf.complex64: 8,
    tf.complex128: 16,
    tf.int8: 1,
    tf.uint8: 1,
    tf.uint16: 2,
    tf.uint32: 4,
    tf.uint64: 8,
    tf.int16: 2,
    tf.int32: 4,
    tf.int64: 8,
    tf.bool: 0.125,
    tf.qint8: 1,
    tf.quint8: 1,
    tf.qint16: 2,
    tf.quint16: 2,
    tf.qint32: 4,
}


def dtype_to_bytes(dtype):
    if dtype in DTYPE_TO_BYTES:
        return DTYPE_TO_BYTES[dtype]
    else:
        return 8


def override_xla():
    """ Patch Graph._create_op_helper method in TF to enable XLA support. This method is called by TF right after
    a new op is created, to adjust the attributes. If we are in XLA scope, we adjust the XLA-related attributes
    so that TF does not merge ops across microbatches. """
    org_create_op_helper = ops.Graph._create_op_helper

    def _create_op_helper_wrapper(self, op, compute_device=True):
        org_create_op_helper(self, op, compute_device)
        if state.in_xla_scope and not (is_var_consumer(op) or is_var(op)):
            # If two neighboring ops have different _XlaScope (or _XlaInternalScope, depending on how XLA was enabled) attributes,
            # XLA will not fuse ops along this edge. However, if there is any op that does not have an _XlaScope attribute that indirectly
            # connect these ops, this op can act as a bridge and XLA will still fuse everything.

            # To prevent XLA from fusing ops for different ticks, we assign a _XlaScope attribute
            # The tick will be added in the grappler optimizer afterwards
            op._set_attr(XLA_SCOPE_ATTR, attr_value_pb2.AttrValue(s=("jit_scope").encode()))
            op._set_attr(
                XLA_INTERNAL_SCOPE_ATTR, attr_value_pb2.AttrValue(s=("jit_scope").encode())
            )
        else:
            # if is_var_consumer(op) or is_var(op):
            # Since variables are shared across microbatches, we need to disable XLA compilation for ops that directly interact with
            # variables. Otherwise they act as "bridges" across microbatches and XLA can still fuse ops from different microbatches.
            op._set_attr(XLA_COMPILE_ATTR, attr_value_pb2.AttrValue(b=False))

    ops.Graph._create_op_helper = _create_op_helper_wrapper


@contextmanager
def xla_scope():
    state.in_xla_scope = True
    yield
    state.in_xla_scope = False


@contextmanager
def measure_time(name):
    t0 = time.time()
    yield
    print(f"[{name}]", time.time() - t0)


def autograph_do_not_convert(func):
    if state.tf_version == 2:
        return tf.autograph.experimental.do_not_convert(func)
    else:
        return func


class GraphTraverser:
    _id = 0

    def __init__(self, graph):
        self.graph = graph
        self.id = GraphTraverser._id
        GraphTraverser._id += 1

        if isinstance(self.graph, tf.Graph):
            set_graph(self.id, graph.as_graph_def())
        elif isinstance(self.graph, gde.Graph):
            set_graph(self.id, graph.to_graph_def())
        elif isinstance(self.graph, graph_pb2.GraphDef):
            set_graph(self.id, graph)
        else:
            raise TypeError(f"Unsupported graph type {type(graph)}")

    def get_nodes_by_names(self, names):
        if isinstance(self.graph, tf.Graph):
            return [self.graph.get_operation_by_name(name) for name in names]
        elif isinstance(self.graph, gde.Graph):
            return [self.graph.get_node_by_name(name) for name in names]
        elif isinstance(self.graph, graph_pb2.GraphDef):
            return get_nodes_by_names(self.id, names)
        else:
            raise TypeError(f"Unsupported graph type {type(graph)}")

    def backward_walk(self, tensors_or_ops, stopping_op_type=None, target_op_type=None):
        """
        Run DFS starting from `tensors_or_ops`, following inputs of ops, and return any encountered op
        whose type is among listed in `target_op_type`. We do not continue backward walk from any op
        whose type is listed in `stopping_op_type`.

        Inputs:
            tensors_or_ops:   A tf.Tensor, tf.Operation, or a list of tf.Tensor or tf.Operation objects
                              to start the backward walk from.
            stopping_op_type: A str or a list of str's specifying the op types to stop the backward
                              walk at. If None, backward walk will continue until the graph ends.
            target_op_type:   A str or a list of str's specifying the types of ops that will be returned.
                              If None, defaults to the stopping op_types. If ANY_OP, any op encountered
                              will be returned.

        Outputs:
            target_ops:       List of ops of the target type encountered during backward walk
        """

        tensors_or_ops_list = (
            tensors_or_ops if isinstance(tensors_or_ops, list) else [tensors_or_ops]
        )

        if stopping_op_type:
            stopping_ops = (
                stopping_op_type if isinstance(stopping_op_type, list) else [stopping_op_type]
            )
        else:
            stopping_ops = []

        if target_op_type == ANY_OP:
            target_types = []
        elif target_op_type is None or target_op_type == []:
            target_types = stopping_ops
        else:
            target_types = target_op_type if isinstance(target_op_type, list) else [target_op_type]

        ops_list = [(get_op(t) if is_tensor_type(t) else t) for t in tensors_or_ops_list]
        op_names = backward_walk(self.id, ops_list, stopping_ops, target_types)
        return self.get_nodes_by_names(op_names)

    def get_ops_of_type(self, op_types, microbatch=None, forward_only=True):
        """ Return the list of ops whose type is in `op_types`, microbatch attribute (if specified)
        matches `microbatch`, and `forward_only` attribute (if specified) matches `forward_only`."""

        list_of_types = op_types if isinstance(op_types, list) else [op_types]
        op_names = get_ops_of_type(self.id, list_of_types, microbatch, forward_only)
        return self.get_nodes_by_names(op_names)


def list_insert(lst, index, item):
    if len(lst) <= index:
        lst.extend([None for _ in range(index - len(lst) + 1)])
    lst[index] = item


def mainify_name(name, prefix=_TRACE_MODEL_PREFIX):
    """
    During CPU tracing/profiling, we add _TRACE_MODEL_PREFIX/_PROFILE_MODEL_PREFIX to all layer and op names. For the main
    execution, we need to strip off these to match the op names to the built layers.
    """
    mainified_name = name.replace(prefix + "__", "")
    return mainified_name.replace(prefix + "_", "")


def assert_compatible(spec_arg):
    if is_tensor_or_var(spec_arg.arg):
        if not spec_arg.spec.is_compatible_with(spec_arg.arg):
            raise TypeError(
                f"Input signature {spec_arg.spec} is not compatible with argument with shape {spec_arg.arg.shape}."
            )


class _SpecArg:
    def __init__(self, spec, arg):
        self.spec = spec
        self.arg = arg


class GraphBuildError(Exception):
    pass


def validate_input_signature(signature):
    def _validate_item(item):
        if not isinstance(item, tf.TensorSpec):
            raise TypeError(
                f"input_signature argument must consist of a possibly nested sequence of tf.TensorSpec objects. Found {item}"
            )

    tf.nest.map_structure(_validate_item, signature)


def cache_key(func, c_args, c_kwargs, signature=None, prefix=None):
    """Returns the TensorFlow-generated cache key associated with the graph, used to determine if the graph
    has changed and requires a recompilation. """

    if prefix == None:
        prefix = []

    if func._created_variables:
        fn = func._stateless_fn
    else:
        fn = func._stateful_fn

    if fn is None:
        return None

    tf_minor_version = _get_minor_tf_version()
    tf_major_version = _get_major_tf_version()

    if tf_major_version == 2 and tf_minor_version >= 4:
        _key = fn._cache_key(prefix + list(c_args), c_kwargs, fn._cache_key_context())
    else:
        _key = fn._cache_key(prefix + list(c_args), c_kwargs)

    if signature is not None:
        flat_input_signature = tuple(tf.nest.flatten((prefix, signature, c_kwargs)))
        # TF2.4 added variable_policy att into the CacheKey namedtuple
        if hasattr(_key, "variable_policy"):
            return CacheKey(
                input_signature=_make_input_signature_hashable(flat_input_signature),
                parent_graph=_key.parent_graph,
                device_functions=_key.device_functions,
                colocation_stack=_key.colocation_stack,
                in_cross_replica_context=_key.in_cross_replica_context,
                variable_policy=_key.variable_policy,
                xla_context_id=_key.xla_context_id,
            )
        else:
            return CacheKey(
                input_signature=_make_input_signature_hashable(flat_input_signature),
                parent_graph=_key.parent_graph,
                device_functions=_key.device_functions,
                colocation_stack=_key.colocation_stack,
                in_cross_replica_context=_key.in_cross_replica_context,
                xla_context_id=_key.xla_context_id,
            )
    else:
        return _key


def get_sinks(graph):
    """ Return the list of nodes with no outputs given a graph as a gde.Graph object"""
    sinks = []
    for node in graph.nodes:
        if len(node.outputs) == 0:
            sinks.append(node)
        else:
            consumers = []
            for out in node.outputs:
                consumers.extend(out.consumers())
            if len(consumers) == 0:
                sinks.append(node)
    return sinks


def get_attr(node, attr):
    if isinstance(node, (tf.Operation, gde.Node)):
        return node.get_attr(attr)
    else:
        return attr_value_to_python_type(node.attr[attr])


def get_op(obj):
    if isinstance(obj, (tf.Tensor, tf.Variable, tf.IndexedSlices)):
        return obj.op
    elif isinstance(obj, gde_Tensor):
        return obj.node
    elif isinstance(obj, (tf.Operation, gde.Node)):
        return obj
    else:
        raise TypeError(f"Invalid type {type(obj)}")


def is_tensor_type(obj):
    return isinstance(obj, tf.Tensor) or isinstance(obj, gde_Tensor)


def is_op_type(obj):
    return isinstance(obj, tf.Operation) or isinstance(obj, gde.Node)


def get_tensor_id(tensor):
    """ Given a tensor, fetch an id.
        Tensor id calculation changed after 2.0
    """
    version_str_list = versions.__version__.split(".")
    if int(version_str_list[0]) == 2 and int(version_str_list[1]) >= 1:
        return id(tensor)
    else:
        return ops.tensor_id(tensor)


def get_dummy_spec():
    """Return the shape and dtype for the dummy tf.Operations"""
    return [], tf.float32


def get_true_spec(op):
    if isinstance(op, tf.Operation):
        return op.get_attr("expected_shape"), op.get_attr("out_type")
    elif isinstance(op, node_def_pb2.NodeDef):
        shape_attr_value = op.attr["expected_shape"]
        dtype_attr_value = op.attr["out_type"]
    else:
        shape_attr_value = [p[1] for p in op._attributes if p[0] == "expected_shape"][0]
        dtype_attr_value = [p[1] for p in op._attributes if p[0] == "out_type"][0]

    shape = shape_attr_value if isinstance(shape_attr_value, list) else shape_attr_value.list.i
    dtype = (
        tf.DType(dtype_attr_value.type) if hasattr(dtype_attr_value, "type") else dtype_attr_value
    )
    return shape, dtype


def make_tf_compatible_name(name):
    """Converts a tensor name into TensorFlow-compatible form."""
    return re.sub("[^a-zA-Z0-9_]", "_", name)


def load_lib(name, op_list=None):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.
      op_list: A list of names of operators that the library should have. If None
          then the .so file's contents will not be verified.

    Raises:
      NameError if one of the required ops is missing.
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    try:
        for expected_op in op_list or []:
            for lib_op in library.OP_LIST.op:
                if lib_op.name == expected_op:
                    break
            else:
                raise NameError(
                    "Could not find operator %s in dynamic library %s" % (expected_op, name)
                )
    # For TF2.1, library is a python module that does not have OP_LIST attr
    except AttributeError:
        pass
    return library


def op_has_attr(op, attr):
    if isinstance(op, tf.Operation):
        return attr in op.node_def.attr
    elif isinstance(op, gde.Node):
        return op.has_attr(attr)
    else:
        return attr in op.attr


def get_op_type(op):
    if isinstance(op, tf.Operation):
        return op.type
    elif isinstance(op, gde.Node):
        return op.op_type
    else:
        return op.op


def get_graph_ops(graph):
    return graph.get_operations() if isinstance(graph, tf.Graph) else graph.nodes


def convert_tensor_to_spec(args, kwargs, input_signature=None):
    def _convert(_arg):
        return (
            tf.TensorSpec(shape=_arg.shape, dtype=_arg.dtype)
            if isinstance(_arg, tf.Tensor)
            else _arg
        )

    if input_signature is None:
        spec_args = tf.nest.map_structure(_convert, args)
        spec_kwargs = tf.nest.map_structure(_convert, kwargs)
    else:

        def _assert_not_tf(x):
            if is_tensor_or_var(x):
                raise TypeError(
                    "When input_signature is specified, all tf.Tensor and tf.Variable arguments must be positional arguments."
                )

        tf.nest.map_structure(_assert_not_tf, kwargs)

        spec_args = input_signature
        spec_kwargs = kwargs

    return spec_args, spec_kwargs


def is_tensor_or_var(input):
    return is_tensor(input) or isinstance(input, tf.Variable)


def is_ref_type(input):
    return input.dtype.name.endswith("_ref")


def is_tensor_or_var_or_op(input):
    return is_tensor_or_var(input) or isinstance(input, tf.Operation)


def is_tensor(input):
    return isinstance(input, tf.Tensor) or isinstance(input, tf.IndexedSlices)


def is_tensor_or_op(input):
    return is_tensor(input) or isinstance(input, tf.Operation)


def raise_if_invalid_non_split_inputs(ns_inputs):
    if ns_inputs is not None and not isinstance(ns_inputs, list):
        raise TypeError("Non-split inputs must be a list.")


def raise_if_invalid_input_split_axes(input_split_axes):
    if input_split_axes is not None and not isinstance(input_split_axes, dict):
        raise TypeError("input_split_axes must be a dict.")


def raise_if_has_cross_microbatch_dependencies(graph):
    """Check if dependencies exist between microbatches"""
    if state.cfg.pipeline == "interleaved":
        bwd_input_ops = [
            op
            for op in graph.get_operations()
            if (op.type == INPUT_OP and not op.get_attr("forward"))
        ]
        offending_ops = set()
        trav = GraphTraverser(graph)

        for input_op in bwd_input_ops:
            intermediate_ops = trav.backward_walk(input_op, stopping_op_type=OUTPUT_OP)
            for intermediate_op in intermediate_ops:
                if intermediate_op.get_attr("microbatch") != input_op.get_attr("microbatch"):
                    offending_ops.add(input_op)
        if len(offending_ops) > 0:
            raise GraphBuildError(
                f"Operations {[op.name for op in offending_ops]} depend on operations corresponding "
                + "to a different microbatch, which is not allowed. "
            )


def set_to_sorted_list(s):
    return sorted(list(s))


class UnionFind:
    def __init__(self, size):
        self._parent = [None] * size
        for i in range(size):
            self._parent[i] = i

    def find_parent(self, x):
        visited = set()
        while self._parent[x] != x:
            if x in visited:
                raise ValueError(
                    "Infinite Loop deteted in UnionFind while look for parent. This should not happen !!!"
                )
            visited.add(x)
            x = self._parent[x]
        return x

    def unify(self, x, y):
        parent_x = self.find_parent(x)
        parent_y = self.find_parent(y)

        if parent_x != parent_y:
            self._parent[parent_x] = self._parent[parent_y]


def create_and_write_to_file(root_dir, file_path, data):
    """
        Args:
            root_dir: directory to which we are writing the file. i.e dest/
            file_path: path including the subdir i.e rank_0/test.index, rank_1/test.index.
            data: data to be written to the file.
    """

    complete_file_path = os.path.join(root_dir, file_path)

    # Getting just the directory in which the file will reside
    dir_to_file = complete_file_path.rsplit("/", 1)[0]

    # creating all the directories if missing
    os.makedirs(dir_to_file, exist_ok=True)

    # remove existing file if present in the path.
    if os.path.exists(complete_file_path):
        os.remove(complete_file_path)

    # writing data to file
    with open(complete_file_path, "wb+") as newfile:
        newfile.write(data)


def get_transfer_chunk_size():
    # Setting default to 1GB, as ctypes string_buffer adds this limitation.
    GB = 1024 * 1024 * 1024
    return int(os.environ.get("SMP_CKPT_TRANSFER_CHUNK_SIZE", str(GB)))


def break_data(data):
    """ Break incoming data into chunk size, if needed."""

    chunk_size = get_transfer_chunk_size()
    start = 0
    end = chunk_size
    result = []

    number_of_pieces = math.ceil(len(data) / chunk_size)
    for p in range(1, number_of_pieces):
        result.append(data[start:end])
        start += chunk_size
        end += chunk_size

    result.append(data[start:])
    return result


def send_checkpoint_files(ckpt_path, dest_rank, sub_dir=None, rank_type=RankType.PP_RANK):
    """
        Send files in checkpoint path:
        - checkpoint file
        - multiple data files
        - .index file.
        - .meta file (TF 1.x)

        Args:
            ckpt_path: The dir to be send
            dest_rank: destination rank to send to.
            sub_dir: list of sub directories to be send. If None, everything in 'ckpt_path' dir is sent
    """

    file_list = []  # sorted(os.listdir(ckpt_path))
    number_of_messages = 0
    if ckpt_path:
        ckpt_path = os.path.realpath(ckpt_path)
        for root, directories, filenames in os.walk(ckpt_path):
            if sub_dir and not (root in sub_dir):
                # only include dir that are in sub_dir list
                continue
            index = root.find(ckpt_path)
            # creating root with just subdirectory name, i.e rank_1/filename.
            mod_root = root[index + len(ckpt_path) + 1 :]
            for filename in filenames:
                file_list.append(os.path.join(mod_root, filename))

        # Calculating total number of messages that will be send.
        # If a file is greater than transfer chuck size, break into multiple pieces.

        chunk_size = get_transfer_chunk_size()
        root_directory = Path(ckpt_path)

        number_of_messages = sum(
            math.ceil(os.stat(os.path.join(ckpt_path, f)).st_size / chunk_size) for f in file_list
        )

    # Sending total messages
    state.comm.send(number_of_messages, dest_rank, rank_type=rank_type)
    transaction_id = 0

    # Sending each file in a tuple (filename, list of data pieces)
    for filename in file_list:
        data = None
        with open(os.path.join(ckpt_path, filename), "rb") as f:
            data = f.read()
        number_of_pieces = math.ceil(len(data) / chunk_size)

        # Breaking the data (if needed) and send.
        broken_data = break_data(data)
        for id in range(number_of_pieces):
            state.comm.async_send(
                (filename, broken_data[id]),
                dest_rank,
                TransactionIdentifier(transaction_id, False),
                rank_type=rank_type,
            )
            transaction_id += 1
    state.comm.wait_all_transactions()


def receive_checkpoint_files(ckpt_path, source_rank, rank_type=RankType.PP_RANK):
    """
        Receive checkpoint files from  source_rank.
    """
    # Receiving total number of messages expected.

    num_messages = state.comm.recv_from(source_rank, rank_type=rank_type)
    # initiating receive.
    for transaction_id in range(0, num_messages):
        state.comm.async_recv_from(
            source_rank, TransactionIdentifier(transaction_id, False), rank_type=rank_type
        )

    # Storing the incoming message as dict filename->[split_pieces]

    message = {}
    for transaction_id in range(0, num_messages):
        filename, val = state.comm.wait_recv(
            source_rank, TransactionIdentifier(transaction_id, False), rank_type=rank_type
        )
        if filename in message.keys():
            message[filename].append(val)
        else:
            message[filename] = [val]

    # write files to output directory
    for filename, val in message.items():
        final_data = b"".join(val)
        create_and_write_to_file(ckpt_path, filename, final_data)


def gather_ckpt_files(send_ckpt_path, receive_ckpt_path, receiver_pp_rank):
    """ Gather files from all ranks to one rank """

    # Sync before sending and receving
    core.barrier()

    # Send checkpoints from ranks that are not in dest_rank.
    if core.is_in_same_instance(core.pp_rank_to_rank(receiver_pp_rank)) != True:
        send_checkpoint_files(send_ckpt_path, receiver_pp_rank)

    # Receive checkpoints from other nodes is multi node case
    if core.pp_rank() == receiver_pp_rank:
        all_ranks = core.get_pp_group()
        all_ranks_instance = [core.is_in_same_instance(rank) for rank in all_ranks]
        # if all the ranks_instance is not same, mp is across nodes.
        if all_ranks_instance.count(all_ranks_instance[0]) != len(all_ranks_instance):
            # Step 2: Fetch all checkpoints from ranks on other nodes.
            for rank in all_ranks:
                # receive from ranks not in the same instance.
                if core.is_in_same_instance(rank) != True:
                    receive_checkpoint_files(receive_ckpt_path, rank, rank_type=RankType.WORLD_RANK)

    # Sync after.
    core.barrier()


def bcast_ckpt_files(sender_ckpt_path, sender_pp_rank, recv_ckpt_path=None):
    """ bcast ckpt files from one rank to the all the rank in other node.
        Used for sending combined ckpt to ranks on other node.
    """

    # if recv_ckpt_path is None , use the sender path to receive
    recv_ckpt_path = recv_ckpt_path if recv_ckpt_path else sender_ckpt_path

    # Sync before sending and receving
    core.barrier()

    if core.pp_rank() == sender_pp_rank:
        for dest_global_rank in core.get_pp_group():
            if core.is_in_same_instance(dest_global_rank) != True:
                rank_dir = os.path.join(sender_ckpt_path, "mp_rank_" + str(dest_global_rank))
                if os.path.exists(rank_dir + "/checkpoint"):
                    # if a rakk dir has checkpoint file, then we are sending rank folder
                    send_checkpoint_files(sender_ckpt_path, dest_global_rank, sub_dir=[rank_dir])
                elif os.path.exists(os.path.join(sender_ckpt_path, "checkpoint")):
                    # if no ranks, then we are sending ckpt in this directory
                    send_checkpoint_files(
                        sender_ckpt_path, dest_global_rank, rank_type=RankType.WORLD_RANK
                    )
                else:
                    # if no ckpt , then just tell ranks nothing to send.
                    send_checkpoint_files(None, dest_global_rank, rank_type=RankType.WORLD_RANK)
    else:
        sender_global_rank = core.pp_rank_to_rank(sender_pp_rank)
        if core.is_in_same_instance(sender_global_rank) != True:
            receive_checkpoint_files(recv_ckpt_path, sender_pp_rank, rank_type=RankType.WORLD_RANK)

    # Sync after.
    core.barrier()


def is_setup_multinode():
    all_ranks = core.get_pp_group()
    all_ranks_instance = [core.is_in_same_instance(rank) for rank in all_ranks]
    if all_ranks_instance.count(all_ranks_instance[0]) == len(all_ranks_instance):
        return False
    return True


def get_create_op():
    if _get_major_tf_version() == 1 or _get_minor_tf_version() < 1:
        return ops.get_default_graph().create_op
    else:
        return ops.get_default_graph()._create_op_internal


def _get_major_tf_version():
    return int(tf.__version__.split(".")[0])


def _get_minor_tf_version():
    return int(tf.__version__.split(".")[1])


def set_create_op(func):
    if _get_major_tf_version() == 1 or _get_minor_tf_version() < 1:
        ops.get_default_graph().create_op = func
    else:
        ops.get_default_graph()._create_op_internal = func


@contextmanager
def partition(device_id):
    """
    Constraints the partitioning algorithm.
    Populates the op name to device mapping.
    """
    if state.cfg.auto_partition:
        yield
    else:
        if device_id >= state.cfg.pipeline_parallel_degree:
            raise ValueError(f"Device id provided must be less than the number of partitions.")

        create_op = get_create_op()

        def create_op_and_set_mapping(*args, **kwargs):
            op = create_op(*args, **kwargs)
            state.op_to_device[op.name] = device_id
            return op

        set_create_op(create_op_and_set_mapping)

        yield

        set_create_op(create_op)


def smdebug_name_to_layer_name(name):
    """
    TF2 only. Convert the smdebug tensor name into layer name and type(inputs or outputs)
    """
    mainified_name = mainify_name(name, prefix=_PROFILE_MODEL_PREFIX)
    end = mainified_name.rfind("/")
    if end == 0 or end == -1:
        raise ValueError(f"tensor name {name} does not follow smdebug naming rule!")

    return mainified_name[:end], mainified_name[end + 1 :]
