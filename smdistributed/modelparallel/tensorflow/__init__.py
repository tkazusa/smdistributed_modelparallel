# First Party
from smdistributed.modelparallel.backend import *
from smdistributed.modelparallel.tensorflow.comm import *
from smdistributed.modelparallel.tensorflow.core_mod import *
from smdistributed.modelparallel.tensorflow.ops import *
from smdistributed.modelparallel.tensorflow.state_mod import state
from smdistributed.modelparallel.tensorflow.utils import override_xla, partition
from smdistributed.modelparallel.tensorflow.v2.checkpoint import CheckpointManager
from smdistributed.modelparallel.tensorflow.v2.model import DistributedModel
from smdistributed.modelparallel.tensorflow.v2.step_mod import register_post_partition_hook, step

state.tf_version = 2

__all__ = [
    "init",
    "step",
    "partition",
    "DistributedModel",
    "CheckpointManager",
    "num_microbatches",
    "rank",
    "size",
    "local_rank",
    "local_size",
    "dp_rank",
    "dp_size",
    "mp_rank",
    "mp_size",
    "pp_rank",
    "pp_size",
    "tp_rank",
    "tp_size",
    "rdp_rank",
    "rdp_size",
    "mp_barrier",
    "pp_barrier",
    "dp_barrier",
    "tp_barrier",
    "rdp_barrier",
    "broadcast",
    "send",
    "recv_from",
    "allgather",
    "barrier",
]


def init(config=None):
    """Initialize SMP.
          Inputs:
             config : a dict which specifies the user-defined configuration.
          config accepts the following keys:
             partitions (required) :
                 Number of model partitions.
             microbatches (optional) [defaults to 1]:
                 An integer specifying the number of microbatches. The batch size must be divisible by this number.
             pipeline (optional) [defaults to "interleaved"]:
                 A str specifying the pipeline type. Can be either "simple" or "interleaved"
             horovod (optional) [defaults to False]:
                 A bool specifying whether SMP is used with Horovod. If Horovod is used, must be set to True.
             contiguous (optional) [defaults to True]:
                 A bool specifying whether the partitions must be contiguous.
             placement_strategy (optional) [defaults to "cluster"]:
                 When hybrid model/data parallelism is used, “cluster” places a single model replica
                 in neighboring device IDs, whereas “spread” will place them as far as possible.
                 Example:
                     - 8 GPUs: [0, 1, 2, 3, 4, 5, 6, 7], 4-way model parallelism, 2-way data parallelism.
                       Two model replicas, each partitioned across 4 GPUs.
                     - "spread" will place the two model replicas in [0, 2, 4, 6] and [1, 3, 5, 7]
                     - "cluster" will place the two model replicas in [0, 1, 2, 3] and [4, 5, 6, 7]
             xla (optional) [defaults to False]:
                 A bool specifying whether XLA is enabled. If XLA is used in TensorFlow 1.x, must be set to True.
                 This is not required in TF2.x.
             optimize (optional) [defaults to "memory"]:
                 A str specifying whether SMP should optimize speed or memory. Must be one of "speed" or "memory".
    """
    if config is None:
        config = {}

    init_internal(
        os.path.dirname(__file__),
        config,
        core,
        state,
        true_if_pt_else_tf=False,
        start_pipeline_threads=True,
    )

    if state.tf_version == 2:
        init_xla()
        patch_loss_scale()
        set_multi_node_flags()


def set_multi_node_flags():
    if core.is_multi_node():
        # set the state.sync_checkpoints to True if there are pp ranks across nodes for the dp rank 0
        if any([r >= core.local_size() for r in core.get_mp_group(0)]):
            state.sync_checkpoints = True


def init_xla():
    """ XLA-related initialization. The changes here enable all ops to be created with
    correct XLA attributes, which ensures that XLA does not fuse ops across microbatches. """

    from smdistributed.modelparallel.tensorflow.pybind import register_xla_optimizer
    from tensorflow.python.eager import context

    # Patch TF Python-level op creation logic to set XLA-related attributes appropriately
    override_xla()

    # Register a custom optimizer with TF Grappler. This makes sure that the new ops created
    # by Grappler, which we do not have direct access in Python, also have correct XLA attributes.
    register_xla_optimizer()

    class SMPContext(context.context().__class__):
        @property
        def config(self):
            cfg = super(SMPContext, self).config
            cfg.graph_options.rewrite_options.custom_optimizers.add().name = "smp_xla_optimizer"
            return cfg

    # Patch TF Eager context to enable the SMP XLA Grappler optimizer
    context.context().__class__ = SMPContext


def patch_loss_scale():
    # The default loss scaler checks finiteness of all gradients.
    # In SMP, this check will apply on a per pp_rank basis, so
    # we need to allgather across ranks for correct loss scaling.
    # The below patch may need update with new TF versions.
    from tensorflow.python.training.experimental import loss_scale
    from smdistributed.modelparallel.tensorflow.ops import is_all_finite_distributed

    loss_scale._is_all_finite = is_all_finite_distributed


num_microbatches = state.num_microbatches
