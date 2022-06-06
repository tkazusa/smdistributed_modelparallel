# Standard Library
from collections import OrderedDict
from contextlib import contextmanager

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.backend.state_mod import ModelParallelState
from smdistributed.modelparallel.tensorflow.logger import get_logger


class TFModelParallelState(ModelParallelState):
    """ TensorFlow-specific global objects accessible from every module."""

    def __init__(self):
        super(TFModelParallelState, self).__init__()

        # A mapping from SmpInput/SmpOutput op_ids to device id the op is assigned to
        self.op_id_to_device = {}

        # Whether we are constructing the graph for compilation, or for actual training
        self.compile_status = None

        # The tf.Graph object used for compilation
        self.compile_graph = None

        # The Compiler object
        self.compiler = None

        # The current microbatch for which we are constructing the graph.
        self.microbatch = 0

        # Whether we are constructing the backward graph or not
        self.backward = False

        # Whether we should override XLA-related attributes of newly created ops
        self.in_xla_scope = False

        # TensorFlow module version imported by the user (1 or 2)
        self.tf_version = 0

        # The ControlManager object
        self.ctrl_mgr = None

        # A counter used to assign a unique link_id to every collective call
        self.comm_link_id_counter = None

        # Model partitioner
        self.partitioner = None

        # Serialized graph to be communicated to/from the partitioner
        self.serialized_graph = None

        # User-specified (and re-partition) op assignment constraints
        self.op_to_device = {}

        # Cache of previously partitioned SerializedGraph objects
        self.partition_cache = None

        # Flag that represents whether we are inside a @smp.step
        self.inside_smp_step = False

        # Flag representing whether we are tracing the DistributedModel
        self.tracking_model = False

        # Op ID generator
        self.op_id_gen = None

        # List of dummy variables (TF1.x only)
        self.dummy_vars = None

        # List of created TraceGraph objects
        self._trace_graphs = []

        # Flag to indicate if we are in the first step
        self._is_first_step = True

        # Map of post partition hooks. Key is func name and value is tuple of (args, kwargs)
        self._post_partition_hooks = OrderedDict()

        # The CPU models used for tracing and profiling (TF2.x only)
        self._tracing_model = None
        self._profile_model = None

        # spec_args used for creating SavedModel
        self._spec_args = None

        # saving
        self.is_saving = False

        # Use dummy logic for auto partition
        self.partition_type = None

        # A container for the accumulation variables for smp.step outputs (TF2.x only)
        self.accum_vars = None

        # The accumulated index of the allgather ops
        self.allgather_idx = 0

        # Mapping from Python ID of CPU layers to their order in get_layers() list
        self.layer_order_map = {}

        # Singleton Graph Traverser object
        self.graph_traverser = None

        # Flag to enable sync ckpts across nodes
        self._sync_checkpoints = False

        # Flag to check first training step.
        self._first_step_done = False

    @property
    def first_step_done(self):
        return self._first_step_done

    @first_step_done.setter
    def first_step_done(self, flag):
        self._first_step_done = flag

    @property
    def sync_checkpoints(self):
        return self._sync_checkpoints

    @sync_checkpoints.setter
    def sync_checkpoints(self, flag):
        self._sync_checkpoints = flag

    @property
    def spec_args(self):
        return self._spec_args

    @spec_args.setter
    def spec_args(self, spec_args):
        self._spec_args = spec_args

    def get_bwd_id(self, op_id):
        return (-1) * op_id

    def horovod_init(self, core):
        import horovod.tensorflow as hvd

        super(TFModelParallelState, self).horovod_init(core)
        hvd.init(core.get_dp_group())

    def generate_trace_graph(self):
        """ We maintain a reference to all TraceGraphs so that id(TraceGraph()) is unique per object throughout execution."""
        from smdistributed.modelparallel.tensorflow.v1.serialization import TraceGraph

        self._trace_graphs.append(TraceGraph())
        self.compile_graph = self._trace_graphs[-1]

    def initialize(self, config, core):
        from smdistributed.modelparallel.tensorflow.compile import Compiler
        from smdistributed.modelparallel.tensorflow.auto import AutoPartitioner, PartitionType

        super(TFModelParallelState, self).initialize(config, core)

        if self.tf_version == 2:
            from smdistributed.modelparallel.tensorflow.v2.serialization import (
                SerializedGraphV2,
                PartitionCache,
            )

            self.serialized_graph = SerializedGraphV2()
            self.partition_cache = PartitionCache()
        else:
            from smdistributed.modelparallel.tensorflow.v1.serialization import SerializedGraphV1

            self.serialized_graph = SerializedGraphV1()

        self.create_pipeline()
        self.compiler = Compiler(self.core)
        self.op_id_gen = OpIdGenerator()
        self.accum_vars = AccumulationVariables()
        self.partitioner = AutoPartitioner()
        self.partition_type = PartitionType.METIS

        if self.tf_version == 2:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[core.local_rank()], "GPU")

    def create_pipeline(self):
        from smdistributed.modelparallel.tensorflow.pipeline import (
            SimplePipeline,
            InterleavedPipeline,
        )
        from smdistributed.modelparallel.tensorflow.control import ControlManager

        if self.cfg.pipeline == "simple":
            self.pipeline = SimplePipeline(self.num_microbatches(), self)
        elif self.cfg.pipeline == "interleaved":
            self.pipeline = InterleavedPipeline(self.num_microbatches(), self)
        else:
            raise ValueError("Pipeline not supported!")

        # currently unused
        self.ctrl_mgr = ControlManager(False)

    def disable_horovod_api(self):
        """ Monkey-patch Horovod API with no-ops for helper process."""

        import horovod.tensorflow as hvd

        get_logger().debug("Disabling Horovod API")

        hvd.init = lambda: None
        hvd.shutdown = lambda: None
        hvd.rank = lambda: 0
        hvd.local_rank = lambda: 0
        hvd.size = lambda: 1
        hvd.local_size = lambda: 1

    @contextmanager
    def reset_allgather_index(self, reset=True):
        _index = self.allgather_idx
        yield
        if reset:
            self.allgather_idx = _index


class AccumulationVariables:
    def __init__(self):
        self._vars = {}

    def get_variable(self, tensor):
        if tensor.name not in self._vars:
            if isinstance(tensor, tf.IndexedSlices):
                raise TypeError("Cannot create accumulation variable for tf.IndexedSlices!")
            else:
                # tf.Tensor or tf.Variable
                shape = tensor.shape

            var = tf.Variable(
                tf.zeros_initializer()(shape=shape, dtype=tensor.dtype),
                trainable=False,
                name=tensor.name.split(":")[0] + "/accum",
            )
            self._vars[tensor.name] = var

        return self._vars[tensor.name]


class OpIdGenerator:
    """ Generates a unique id for each SMP op."""

    def __init__(self):
        self.id_map = {}
        self.reverse_map = {}

    def get_op_id(self, mb, asg_id, count):
        """ Get an op_id for the argument tuple. If the tuple was encountered before,
        return the same op_id."""

        if (mb, asg_id, count) not in self.id_map:
            op_id = 1 + len(self.id_map)
            self.id_map[(mb, asg_id, count)] = op_id
            self.reverse_map[op_id] = (mb, asg_id, count)
        return self.id_map[(mb, asg_id, count)]

    def transpose_op_id(self, op_id, mb):
        if op_id > 0:
            _, asg_id, count = self.reverse_map[op_id]
            return self.get_op_id(mb, asg_id, count)
        else:
            _, asg_id, count = self.reverse_map[-op_id]
            return -self.get_op_id(mb, asg_id, count)


state = TFModelParallelState()
