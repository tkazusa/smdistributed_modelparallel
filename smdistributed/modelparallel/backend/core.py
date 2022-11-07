# Standard Library
import atexit
import ctypes
import sys
from enum import Enum
from functools import lru_cache

# First Party
from smdistributed.modelparallel.backend.exceptions import (
    InvalidEnvironmentError,
    NotInitializedError,
    WorkerSizeError,
)
from smdistributed.modelparallel.backend.logger import get_logger

logger = get_logger()

# TODO: share the enum across languages
# this is also set in cpp, need to ensure this is the same as cpp enum
class InitStatus(Enum):
    GOOD = 0
    BAD_ENV = 1
    LOW_DEVICE_COUNT = 2


class Ranker:
    """ Maintain rank assignments based on the chosen placement strategy. """

    def __init__(self, placement_strategy, rdp_size, pp_size, tp_size):
        self.ps = placement_strategy

        if self.ps == "cluster":
            self.ps = "DPT"
        elif self.ps == "spread":
            self.ps = "TPD"

        self.size_map = {"P": pp_size, "D": rdp_size, "T": tp_size}
        self.size = pp_size * tp_size * rdp_size

    def get_pp_rank(self, rank):
        return self._get_group_rank(rank, "P")

    def get_tp_rank(self, rank):
        return self._get_group_rank(rank, "T")

    def get_rdp_rank(self, rank):
        return self._get_group_rank(rank, "D")

    def get_dp_rank(self, rank):
        t_index = self.ps.index("T")
        d_index = self.ps.index("D")
        if t_index < d_index:
            return self.get_rdp_rank(rank) + self.size_map["D"] * self.get_tp_rank(rank)
        else:
            return self.get_tp_rank(rank) + self.size_map["T"] * self.get_rdp_rank(rank)

    def get_mp_rank(self, rank):
        t_index = self.ps.index("T")
        p_index = self.ps.index("P")
        if t_index < p_index:
            return self.get_pp_rank(rank) + self.size_map["P"] * self.get_tp_rank(rank)
        else:
            return self.get_tp_rank(rank) + self.size_map["T"] * self.get_pp_rank(rank)

    def get_pp_group(self, rank):
        return self._get_group(rank, "P")

    def get_tp_group(self, rank):
        return self._get_group(rank, "T")

    def get_rdp_group(self, rank):
        return self._get_group(rank, "D")

    def get_dp_group(self, rank):
        t_index = self.ps.index("T")
        d_index = self.ps.index("D")
        dp_group = []
        if t_index < d_index:
            for t in self.get_tp_group(rank):
                dp_group.extend(self.get_rdp_group(t))
        else:
            for d in self.get_rdp_group(rank):
                dp_group.extend(self.get_tp_group(d))
        return dp_group

    def get_mp_group(self, rank):
        t_index = self.ps.index("T")
        p_index = self.ps.index("P")
        mp_group = []
        if t_index < p_index:
            for t in self.get_tp_group(rank):
                mp_group.extend(self.get_pp_group(t))
        else:
            for p in self.get_pp_group(rank):
                mp_group.extend(self.get_tp_group(p))
        return mp_group

    def _get_group(self, rank, query_dim):
        gap = self._get_multiplier(query_dim)
        rank_map = {}
        for dim in self.ps:
            if dim != query_dim:
                rank_map[dim] = self._get_group_rank(rank, dim)
            else:
                rank_map[dim] = 0
        base_rank = self._translate(rank_map)

        return [base_rank + i * gap for i in range(0, self.size_map[query_dim])]

    def _get_group_rank(self, rank, query_dim):
        multiplier = self._get_multiplier(query_dim)

        return (rank // multiplier) % self.size_map[query_dim]

    def _get_multiplier(self, query_dim):
        dim_index = self.ps.index(query_dim)
        multiplier = 1
        for dim in self.ps[dim_index + 1 :]:
            multiplier *= self.size_map[dim]
        return multiplier

    def _translate(self, rank_map):
        rank = 0
        for dim in self.ps:
            mult = self._get_multiplier(dim)
            rank += mult * rank_map[dim]
        return rank

    def translate(self, pp_rank, tp_rank, rdp_rank):
        return self._translate({"P": pp_rank, "D": rdp_rank, "T": tp_rank})

    def get_rdp_rank_from_dp_rank(self, dp_rank):
        t_index = self.ps.index("T")
        d_index = self.ps.index("D")
        if t_index < d_index:
            return dp_rank % self.size_map["D"]
        else:
            return dp_rank // self.size_map["T"]

    def get_tp_rank_from_dp_rank(self, dp_rank):
        t_index = self.ps.index("T")
        d_index = self.ps.index("D")
        if t_index < d_index:
            return dp_rank // self.size_map["D"]
        else:
            return dp_rank % self.size_map["T"]

    def get_pp_rank_from_mp_rank(self, mp_rank):
        t_index = self.ps.index("T")
        p_index = self.ps.index("P")
        if t_index < p_index:
            return mp_rank % self.size_map["P"]
        else:
            return mp_rank // self.size_map["T"]

    def get_tp_rank_from_mp_rank(self, mp_rank):
        t_index = self.ps.index("T")
        p_index = self.ps.index("P")
        if t_index < p_index:
            return mp_rank // self.size_map["P"]
        else:
            return mp_rank % self.size_map["T"]


class ExitHook(object):
    """
    A hook to be executed before sys.exit. Mainly used to notify helper and
    main processes to notify each other while shutting down, so the other can
    shutdown with the same exit code.
    """

    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        self._orig_excepthook = sys.excepthook
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc
        self._orig_excepthook(exc_type, exc, *args)


class ModelParallelCore:
    def __init__(self):
        self.lib = None
        self.cfg = None
        self.exit_hook = None
        self.exit_signal_received = False

    @property
    def initialized(self):
        if self.lib is None:
            return False
        else:
            return self.lib.smp_initialized()

    def attach_exit_hook(self):
        self.exit_hook = ExitHook()
        self.exit_hook.hook()

    def _validate_worker_size(self):
        world_size = self.size()
        pp_size = self.pp_size()
        tp_size = self.tp_size()
        if pp_size * tp_size > world_size:
            raise WorkerSizeError(
                f"The number of processes must be at least the product of the specified pipeline parallelism and tensor parallelism degrees ({pp_size * tp_size})."
            )
        if world_size % (pp_size * tp_size) != 0:
            raise WorkerSizeError(
                f"The number of processes must be an integer multiple of the product of the specified pipeline parallelism and tensor parallelism degrees ({pp_size * tp_size})."
            )
        if world_size // (pp_size * tp_size) > 1 and not (self.cfg.horovod or self.cfg.ddp):
            raise WorkerSizeError(
                f"If the number of processes is larger than the the product of the specified pipeline parallelism and tensor parallelism degrees ({pp_size * tp_size}), "
                + f"then 'horovod':True (for TensorFlow) or 'ddp':True (for PyTorch) must be specified. Passed configurations are {self.cfg._input_config}."
            )

    def shutdown(self):
        """Shutdown SMP. Performs cleanup at the backend. Automatically called
           upon termination of the script or error."""
        logger.debug(f"{self.rank()}  shutting down...")
        success = not self.exit_hook.exit_code and self.exit_hook.exception is None
        self.lib.smp_shutdown(ctypes.c_bool(success))

    def initialize(self, lib_path, cfg, true_if_pt_else_tf, start_pipeline_threads=True):
        """Create the DLL, and initialize the backend threads."""
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self.cfg = cfg

        # disable D2D if pipeline_parallel_degree is 1 so that we do not waste space
        # for D2D recv buffer - the below flag will not be enough to enable D2D if
        # SMP_DISABLE_D2D=1
        initialize_d2d = self.cfg.pipeline_parallel_degree > 1

        previously_inited = self.initialized
        status_code = self.lib.smp_init(
            ctypes.c_int(self.cfg.pipeline_parallel_degree),
            ctypes.c_int(self.cfg.microbatches),
            ctypes.c_bool(self.cfg.optimize == "memory"),
            ctypes.c_bool(true_if_pt_else_tf),
            ctypes.c_bool(start_pipeline_threads),
            ctypes.c_bool(self.cfg.horovod),
            ctypes.c_bool(initialize_d2d),
        )

        if not previously_inited:
            # NOTE: exceptions should not be raised between `smp_init` call up to this point,
            # as the shutdown clean up will not take place until the shutdown hook is registered.
            self.attach_exit_hook()
            atexit.register(self.shutdown)

        if status_code == InitStatus.BAD_ENV.value:
            raise InvalidEnvironmentError("SageMaker environment not found")
        elif status_code == InitStatus.LOW_DEVICE_COUNT.value:
            raise WorkerSizeError(
                f"The number of node-local MPI processes {self.lib.smp_local_size()} "
                f"cannot be larger than the device count {self.lib.smp_device_count()}"
            )
        elif status_code != InitStatus.GOOD.value:
            # NOTE for adding future exceptions:
            # To support catching this exception and retrying init for some reason
            # backend needs to be modified so that initialization of MPI and/or threads
            # does not happen multiple times.
            # This does not matter for the current errors as it does not make sense to retry them.
            raise InitializationErrors("Initialization failed")

        self.ranker = Ranker(
            self.cfg.placement_strategy, self.rdp_size(), self.pp_size(), self.tp_size()
        )

        self.cfg.construct_zero2d_config_dict(self)

        # needs to run after shutdown hook is registered
        self._validate_worker_size()

        # needs to run after setting initialized flag to True
        self.lib.smp_set_pp_group(
            ctypes.c_int(self.cfg.pipeline_parallel_degree),
            (ctypes.c_int * self.cfg.pipeline_parallel_degree)(*self.get_pp_group()),
        )
        self.lib.smp_create_timeline()

    def _validate_initialized(self):
        if not self.initialized:
            raise NotInitializedError

    @lru_cache(maxsize=1)
    def rank(self):
        """Global rank of the process."""
        self._validate_initialized()
        return self.lib.smp_rank()

    @lru_cache(maxsize=1)
    def size(self):
        """Global number of processes."""
        self._validate_initialized()
        return self.lib.smp_size()

    @lru_cache(maxsize=1)
    def local_rank(self):
        """Rank of the process in the machine."""
        self._validate_initialized()
        return self.lib.smp_local_rank()

    @lru_cache(maxsize=1)
    def local_size(self):
        """Number of processes in the machine."""
        self._validate_initialized()
        return self.lib.smp_local_size()

    @lru_cache(maxsize=1)
    def pp_rank(self):
        """Rank of the pipeline-parallel model partition assigned to the current process."""
        self._validate_initialized()
        return self.ranker.get_pp_rank(self.rank())

    @lru_cache(maxsize=1)
    def mp_rank(self):
        """ Alias for pp_rank() """
        if self.tp_size() > 1:
            logger.warning(
                f"Using mp_rank(). When tensor-parallel degree is more than 1, mp_rank() refers to the rank across all model-parallel processes, including pipeline parallelism and tensor parallelism. If you would like to get the rank among pipeline-parallel processes only, use pp_rank()."
            )
        return self.ranker.get_mp_rank(self.rank())

    @lru_cache(maxsize=1)
    def dp_rank(self):
        """Rank of the model replica assigned to the current process."""
        self._validate_initialized()
        return self.ranker.get_dp_rank(self.rank())

    @lru_cache(maxsize=1)
    def tp_rank(self):
        """ Rank within the tensor-parallelism group """
        self._validate_initialized()
        return self.ranker.get_tp_rank(self.rank())

    @lru_cache(maxsize=1)
    def rdp_rank(self):
        """ Rank within the reduced data-parallelism group """
        self._validate_initialized()
        return self.ranker.get_rdp_rank(self.rank())

    @lru_cache(maxsize=1)
    def dp_size(self):
        """The number of model replicas."""
        return self.size() // self.pp_size() if self.initialized else 1

    @lru_cache(maxsize=1)
    def pp_size(self):
        """The pipeline parallelism degree."""
        return self.cfg.pipeline_parallel_degree if self.initialized else 1

    @lru_cache(maxsize=1)
    def tp_size(self):
        """ Tensor-parallelism degree """
        return self.cfg.tensor_parallel_degree if self.initialized else 1

    @lru_cache(maxsize=1)
    def rdp_size(self):
        """ Reduced data-parallel degree """
        return self.dp_size() // self.tp_size()

    @lru_cache(maxsize=1)
    def mp_size(self):
        """ Model parallelism degree (tensor-parallelism x pipeline-parallelism) """
        if self.tp_size() > 1:
            logger.warning(
                f"Using mp_size(). When tensor-parallel degree is more than 1, mp_size() refers to the total model parallelism degree across pipeline parallelism and tensor parallelism. If you would like to get the pipeline parallelism degree, use pp_size()."
            )
        return self.pp_size() * self.tp_size()

    @lru_cache(maxsize=1024)
    def get_dp_group(self, pp_rank=None):
        """The group of ranks that hold the same model partition"""
        self._validate_initialized()
        if pp_rank is None:
            pp_rank = self.pp_rank()
        base_rank = self.ranker.translate(pp_rank=pp_rank, tp_rank=0, rdp_rank=0)
        return self.ranker.get_dp_group(base_rank)

    @lru_cache(maxsize=1024)
    def get_pp_group(self, dp_rank=None):
        """The group of ranks that jointly perform pipeline parallelism."""
        self._validate_initialized()
        if dp_rank is None:
            dp_rank = self.dp_rank()
        tp_rank = self.ranker.get_tp_rank_from_dp_rank(dp_rank)
        rdp_rank = self.ranker.get_rdp_rank_from_dp_rank(dp_rank)
        base_rank = self.ranker.translate(pp_rank=0, tp_rank=tp_rank, rdp_rank=rdp_rank)
        return self.ranker.get_pp_group(base_rank)

    @lru_cache(maxsize=1024)
    def get_mp_group(self, rdp_rank=None):
        """ Model parallelism group (across pipeline and tensor parallelism) """

        self._validate_initialized()
        if rdp_rank is None:
            rdp_rank = self.rdp_rank()
        base_rank = self.ranker.translate(pp_rank=0, tp_rank=0, rdp_rank=rdp_rank)
        return self.ranker.get_mp_group(base_rank)

    @lru_cache(maxsize=1024)
    def get_tp_group(self, pp_rank=None, rdp_rank=None):
        """ The group of ranks that jointly perform tensor parallelism. """
        self._validate_initialized()
        if pp_rank is None:
            pp_rank = self.pp_rank()
        if rdp_rank is None:
            rdp_rank = self.rdp_rank()
        rank = self.ranker.translate(pp_rank=pp_rank, tp_rank=0, rdp_rank=rdp_rank)
        return self.ranker.get_tp_group(rank)

    @lru_cache(maxsize=1024)
    def get_rdp_group(self, pp_rank=None, tp_rank=None):
        """ The group of ranks that hold the exact same model partitions after pipeline and tensor parallelism. """
        self._validate_initialized()
        if pp_rank is None:
            pp_rank = self.pp_rank()
        if tp_rank is None:
            tp_rank = self.tp_rank()
        rank = self.ranker.translate(pp_rank=pp_rank, tp_rank=tp_rank, rdp_rank=0)
        return self.ranker.get_rdp_group(rank)

    def barrier(self):
        """Hangs the computation of the current process until all process reach the barrier."""
        self._validate_initialized()
        return self.lib.smp_barrier()

    @lru_cache(maxsize=1024)
    def pp_rank_to_rank(self, pp_rank):
        """ Convert pp_rank to rank for the dp group the current process is in."""
        self._validate_initialized()
        return self.ranker.translate(
            pp_rank=pp_rank, tp_rank=self.tp_rank(), rdp_rank=self.rdp_rank()
        )

    @lru_cache(maxsize=1024)
    def dp_rank_to_rank(self, dp_rank):
        """ Convert dp_rank to rank for the pp_group the current process is in."""
        self._validate_initialized()
        tp_rank = self.ranker.get_tp_rank_from_dp_rank(dp_rank)
        rdp_rank = self.ranker.get_rdp_rank_from_dp_rank(dp_rank)
        return self.ranker.translate(tp_rank=tp_rank, rdp_rank=rdp_rank, pp_rank=self.pp_rank())

    @lru_cache(maxsize=1024)
    def rdp_rank_to_rank(self, rdp_rank):
        """ Convert rdp_rank to rank for the pp_group and tp_group the current process is in."""
        self._validate_initialized()
        return self.ranker.translate(
            tp_rank=self.tp_rank(), rdp_rank=rdp_rank, pp_rank=self.pp_rank()
        )

    @lru_cache(maxsize=1024)
    def tp_rank_to_rank(self, tp_rank):
        """ Convert tp_rank to rank for the pp_group and rdp_group the current process is in."""
        self._validate_initialized()
        return self.ranker.translate(
            tp_rank=tp_rank, rdp_rank=self.rdp_rank(), pp_rank=self.pp_rank()
        )

    @lru_cache(maxsize=1024)
    def mp_rank_to_rank(self, mp_rank):
        """ Convert mp_rank to rank for the rdp_group the current process is in."""
        self._validate_initialized()
        tp_rank = self.ranker.get_tp_rank_from_mp_rank(mp_rank)
        pp_rank = self.ranker.get_pp_rank_from_mp_rank(mp_rank)
        return self.ranker.translate(tp_rank=tp_rank, rdp_rank=self.rdp_rank(), pp_rank=pp_rank)

    def is_in_same_instance(self, rank):
        """ Return whether the given rank is in the same instance as the current rank. """
        return self.lib.smp_is_in_same_instance(rank) == 1

    def is_multi_node(self):
        """ Return whether running with multi-node. """
        return self.size() > self.local_size()

    def instance_id(self):
        """ Return the instance id the current rank is in. """
        return self.lib.smp_instance_id()

    def register_counts(self, size, input_count, output_count):
        """Register the input and output counts with the backend"""
        self._validate_initialized()
        return self.lib.smp_register_counts(
            ctypes.c_int(size),
            (ctypes.c_int * size)(*input_count),
            (ctypes.c_int * size)(*output_count),
        )

    def register_op_behavior(self, num_ops, op_ids, behaviors):
        """Register the op behaviors with the backend"""
        self._validate_initialized()
        return self.lib.smp_register_op_behavior(
            ctypes.c_int(num_ops),
            (ctypes.c_int * num_ops)(*op_ids),
            (ctypes.c_int * num_ops)(*behaviors),
        )

    def register_recv_link_ids(self, recv_link_ids):
        """ Register the link_id's for RECV ops with the backend"""
        self._validate_initialized()

        from smdistributed.modelparallel.backend.utils import flatten

        buf = flatten(recv_link_ids)
        buf_len = len(buf)
        self.lib.smp_register_recv_link_ids(ctypes.c_int(buf_len), (ctypes.c_int * buf_len)(*buf))

    def proxy_wake_up(self):
        """ Wake up the communications proxy thread, which indicates that a new training step is starting."""
        self._validate_initialized()
        self.lib.smp_proxy_wake_up()

    def timeline_start_step(self):
        """clear some timeline for each step"""
        return self.lib.smp_timeline_start_step()

    def timeline_end_step(self):
        """clear some timeline for each step"""
        return self.lib.smp_timeline_end_step()

    def timeline_record_pipeline_event(self, microbatch, label):
        """Record a pipeline event. For Pytorch the label is the pipeline request info."""
        label = label.encode("utf-8")
        self.lib.smp_timeline_record_pipeline_event.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.smp_timeline_record_pipeline_event(microbatch, label)

    def get_and_reset_memory_stats(self):
        """Get D2D buffer memory metrics (peak allocation and reservation)"""

        class SmpMemoryStats(ctypes.Structure):
            _fields_ = [
                ("backend_d2d_peak_allocated_mb", ctypes.c_size_t),
                ("backend_d2d_peak_reserved_mb", ctypes.c_size_t),
                ("gpu_free_memory_mb", ctypes.c_size_t),
                ("gpu_total_memory_mb", ctypes.c_size_t),
            ]

        self.lib.smp_get_and_reset_memory_metrics.restype = SmpMemoryStats
        return self.lib.smp_get_and_reset_memory_metrics()

    def get_and_reset_alloc_stats(self):
        """Get D2D allocation metrics (success and failure counts)"""

        class SmpAllocStats(ctypes.Structure):
            _fields_ = [
                ("backend_d2d_success_allocated", ctypes.c_ulonglong),
                ("backend_d2d_failure_allocated", ctypes.c_ulonglong),
            ]

        self.lib.smp_get_and_reset_alloc_metrics.restype = SmpAllocStats
        return self.lib.smp_get_and_reset_alloc_metrics()
