# Standard Library
import copy
import os
import socket
from collections import defaultdict
from contextlib import contextmanager

# Third Party
import torch

# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup, RankType
from smdistributed.modelparallel.backend.state_mod import ModelParallelState
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.exceptions import CheckpointingError, SMPUnsupportedError
from smdistributed.modelparallel.torch.handle_manager import HandleManager
from smdistributed.modelparallel.torch.links import LinkManager
from smdistributed.modelparallel.torch.module_manager import ModuleManager
from smdistributed.modelparallel.torch.patch_manager import PatchManager
from smdistributed.modelparallel.torch.patches.checkpoint import (
    CheckpointConfig,
    CheckpointNodesCache,
)
from smdistributed.modelparallel.torch.random import RngManager
from smdistributed.modelparallel.torch.serialization import SerializationManager
from smdistributed.modelparallel.torch.throttler import NCCLThrottler
from smdistributed.modelparallel.torch.tp_registry import TensorParallelismRegistry
from smdistributed.modelparallel.torch.utils import rmsg


class PTModelParallelState(ModelParallelState):
    def __init__(self):
        super(PTModelParallelState, self).__init__()
        from smdistributed.modelparallel.torch.offload import TensorOffloader

        self.module_manager = ModuleManager()
        self.handle_manager = HandleManager()
        self.patch_manager = PatchManager()
        self.link_manager = LinkManager()
        self.serialization_manager = SerializationManager()
        self.tp_registry = TensorParallelismRegistry()
        self.nccl_throttler = NCCLThrottler()
        self.checkpoint_nodes_cache = CheckpointNodesCache()
        self.rng_manager = RngManager()
        self.offloaders = defaultdict(lambda: TensorOffloader())

        # todo: this needs to be moved into below func
        # but step wrapper is defined before init_torch func is called from smdistributed.modelparallel init
        # its okay if this gets copied across multiple inits as the step functions
        # would have different id, so all use different keys in this dict
        self.step_func = {}

        # moving some inits to below functions helps us reset and init smp multiple times during tests.
        # any state that depends on the model must be inside below function, or must be reset inside it.
        # as a general rule try to place any new var inside below fn
        self.init_model_specific_state()

    def _initialize_data_parallel(self, core):
        if self.cfg.horovod:
            self.horovod_init(core)
        elif self.cfg.ddp or self.cfg.herring:
            self.ddp_init(core)

    def initialize(self, config, core):
        super(PTModelParallelState, self).initialize(config, core)
        self.create_pipeline()
        self.module_manager.set_config(config)
        self.nccl_throttler.set_throttle_limit()
        self.rng_manager.set_state(config.tensor_parallel_seed)
        self.stream = torch.cuda.current_stream()  # torch.cuda.Stream(device=core.local_rank())

    def _initialize_comm(self):
        from smdistributed.modelparallel.torch.collectives import PTCollectiveCommunicator

        self.comm = PTCollectiveCommunicator(self.core)

    def horovod_init(self, core):
        import horovod.torch as hvd

        super(PTModelParallelState, self).horovod_init(core)
        hvd.init(core.get_dp_group())

    def ddp_init(self, core):
        import torch.distributed as dist
        from torch.distributed.distributed_c10d import _get_default_group

        if core.rank() == 0:
            if "MASTER_ADDR" in os.environ:
                ip = os.environ["MASTER_ADDR"]
            else:
                ip = socket.gethostbyname(socket.gethostname())
            self.comm.broadcast(ip, group=CommGroup.WORLD)
        else:
            ip = self.comm.recv_from(0, RankType.WORLD_RANK)

        if self.cfg.ddp_port is not None:
            main_port = self.cfg.ddp_port
        elif "MASTER_PORT" in os.environ:
            main_port = int(os.environ["MASTER_PORT"])
        else:
            main_port = 29760

        # main pg
        init_method = f"tcp://{ip}:{main_port}"
        dist.init_process_group(
            self.cfg.ddp_dist_backend,
            rank=core.rank(),
            world_size=core.size(),
            init_method=init_method,
        )

        self.world_process_group = _get_default_group()

        # torch.dist requires all ranks to participate in a new_group call, and in same order

        # dp pg
        dp_groups = []
        for pp_rank in range(core.pp_size()):
            ranks = core.get_dp_group(pp_rank)
            dp_groups.append(dist.new_group(ranks=ranks))

        self.dp_process_group = dp_groups[core.pp_rank()]

        # mp pg
        mp_groups = []
        for rdp_rank in range(core.rdp_size()):
            ranks = core.get_mp_group(rdp_rank)
            mp_groups.append(dist.new_group(ranks=ranks))

        self.mp_process_group = mp_groups[core.rdp_rank()]

        # pp pg
        pp_groups = []
        for dp_rank in range(core.dp_size()):
            ranks = core.get_pp_group(dp_rank)
            pp_groups.append(dist.new_group(ranks=ranks))

        self.pp_process_group = pp_groups[core.dp_rank()]

        # tp pg
        tp_groups = []
        for pp_rank in range(core.pp_size()):
            tp_grps = []
            for rdp_rank in range(core.rdp_size()):
                ranks = core.get_tp_group(pp_rank=pp_rank, rdp_rank=rdp_rank)
                tp_grps.append(dist.new_group(ranks=ranks))
            tp_groups.append(tp_grps)

        self.tp_process_group = tp_groups[core.pp_rank()][core.rdp_rank()]

        # rdp pg
        rdp_groups = []
        for pp_rank in range(core.pp_size()):
            rdp_grps = []
            for tp_rank in range(core.tp_size()):
                ranks = core.get_rdp_group(pp_rank=pp_rank, tp_rank=tp_rank)
                rdp_grps.append(dist.new_group(ranks=ranks))
            rdp_groups.append(rdp_grps)

        self.rdp_process_group = rdp_groups[core.pp_rank()][core.tp_rank()]

        self.logger.info(
            rmsg(
                f"Finished initializing torch distributed process groups. pp_rank: {core.pp_rank()}, tp_rank: {core.tp_rank()}, dp_rank: {core.dp_rank()}, rdp_rank: {core.rdp_rank()}"
            )
        )

    def create_pipeline(self):
        from smdistributed.modelparallel.torch.pipeline import (
            SimplePipeline,
            InterleavedPipeline,
            OnlyForwardPipeline,
        )

        if self.cfg.pipeline == "simple":
            self.pipeline = SimplePipeline()
        elif self.cfg.pipeline == "interleaved":
            self.pipeline = InterleavedPipeline()
        elif self.cfg.pipeline == "_only_forward":
            self.pipeline = OnlyForwardPipeline()
        else:
            raise SMPUnsupportedError("Pipeline not supported!")

    def init_model_specific_state(self):
        # The current microbatch for which we are constructing the graph, keyed by step_function id
        self.microbatch = 0
        self.exec_thread_id = None
        self.model = None
        self.optimizer = None
        self.current_step_func_id = None
        self.no_grad_context = False
        self._param_index_to_name_tp_group = None
        self._param_name_to_index_tp_group = None
        self.checkpoint_activations_config = CheckpointConfig(enabled=False)
        self.phantom_structures = {}
        self.is_in_fwd_on_checkpointed_fn = False
        self.loaded_model_state = None
        self.loaded_optimizer_state = None
        self.no_reinitialization = False  # internal usage for test purpose

        self.param_initializers = {}

        # Upload the metrics to studio only once
        self.has_uploaded_metrics = False

        # Record the number of hops between devices
        self.num_hops = 0

        # helpful for testing to reset state inside manager classes before new runs
        self.module_manager.reset()
        self.handle_manager.reset()
        self.patch_manager.reset()
        self.link_manager.reset()
        self.checkpoint_nodes_cache.reset()
        self.serialization_manager.reset()
        self.offloaders.clear()
        self.tp_registry.reset()

    def current_minibatch(self):
        return self.current_step_func().minibatch

    def current_step_func(self):
        if self.current_step_func_id is None:
            return None
        return self.step_func[self.current_step_func_id]

    def current_offloader(self):
        if self.current_step_func_id is None:
            return None
        return self.offloaders[self.current_step_func_id]

    @property
    def param_name_to_index_tp_group(self):
        from smdistributed.modelparallel.torch.comm import CommGroup, allgather

        if self._param_name_to_index_tp_group is None:
            param_name_to_index = state.optimizer.param_name_to_index()
            self._param_name_to_index_tp_group = allgather(param_name_to_index, CommGroup.TP_GROUP)
        return self._param_name_to_index_tp_group

    @property
    def param_index_to_name_tp_group(self):
        if self._param_index_to_name_tp_group is None:
            self._param_index_to_name_tp_group = []
            param_name_to_index_tp_group = self.param_name_to_index_tp_group
            for param_name_to_index_map in param_name_to_index_tp_group:
                param_index_to_name_map = {}
                for param_name, index in param_name_to_index_map.items():
                    param_index_to_name_map[index] = param_name
                self._param_index_to_name_tp_group.append(param_index_to_name_map)
        return self._param_index_to_name_tp_group

    @property
    def in_step_func(self):
        return self.current_step_func() is not None

    def switching_to_worker(self, worker_id, req):
        """
        Some bookkeeping when we switch to a worker
        """
        from smdistributed.modelparallel.torch.messages import (
            ModuleExecutionRequest,
            StepExecutionRequest,
            SequentialModulesExecutionRequest,
        )

        if isinstance(req, (SequentialModulesExecutionRequest, ModuleExecutionRequest)):
            self.module_manager.execution_stack = req.execution_stack
        elif isinstance(req, StepExecutionRequest):
            # request for a new MB fwd/bwd pass
            self.module_manager.execution_stack = []
        self.exec_thread_id = worker_id

        self.microbatch = req.mb

    @property
    def current_worker(self) -> "WorkerHolder":
        # current executing worker
        if self.exec_thread_id is not None:
            return self.exec_server.get_worker(self.exec_thread_id)
        else:
            return None

    def is_tracing(self) -> bool:
        from smdistributed.modelparallel.torch.messages import TraceStepExecutionRequest

        cur_worker = self.current_worker
        if cur_worker is not None:
            return isinstance(cur_worker.req, TraceStepExecutionRequest)
        else:
            return False

    def should_record_metadata(self) -> bool:
        return self.cfg.static_mode and self.exec_server.server_queue.should_record()

    def skip_metadata_transmission(self) -> bool:
        """ Whether to skip metadata transmission, will return True under 2 conditions:
            - static_mode is set to True
            - The deterministic server queue has established the message sequence for the current step function
        """
        return (
            self.cfg.static_mode
            and self.exec_server.server_queue.task_order_set[self.current_step_func()]
        )

    def register_minibatch_link_ids(self):
        """Register the current minibatch's link_ids used to send tensors"""
        msg_meta_to_link_id = self.exec_server.comm.msg_meta_to_link_id[self.current_step_func()]
        with self.serialization_manager.catch_and_raise_for_large_object(msg_meta_to_link_id):
            self.exec_server.comm.minibatch_msg_meta_to_link_id = copy.deepcopy(msg_meta_to_link_id)

    def clear_minibatch_state(self):
        self.phantom_structures = {}
        self.module_manager.clear_minibatch_state()
        self.serialization_manager.clear_minibatch_state()
        if not self.cfg.zero2d_enabled():
            self.model.grad_counter.clear_minibatch_state()
        self.checkpoint_nodes_cache.reset()
        self.link_manager.reset()
        if self.current_offloader() is not None:
            self.current_offloader().reset()

        if self.skip_metadata_transmission():
            smplib.smp_torch_clear_minibatch_preemptive_receptions()

    @contextmanager
    def enable_activation_checkpoints(self, config):
        # what's inside this fn can't be interrupted as any computation here is all local
        # so when this fn returns we can reset it
        if config is None:
            yield
        else:
            if self.checkpoint_activations_config.enabled:
                # prev config is already set
                raise CheckpointingError(
                    f"Checkpointing module {config.module_name} inside a checkpointed module {self.checkpoint_activations_config.module_name} is not supported."
                )

            prev_config = self.checkpoint_activations_config
            self.checkpoint_activations_config = config
            try:
                yield
            finally:
                self.checkpoint_activations_config = prev_config

    @contextmanager
    def rerunning_fwd_on_checkpointed_fn(self,):
        self.is_in_fwd_on_checkpointed_fn = True
        try:
            yield
        finally:
            self.is_in_fwd_on_checkpointed_fn = False

    @contextmanager
    def fork_smp_rng_state(self, seed=None, enabled=True, reset_at_end=True):
        """
        Fork the seed of smp's RNG generator and reset at the end
        Args:
            seed: the seed used to set smp's RNG generator. If None the current seed will be used
            reset_at_end: if ``True`` the generator's seed will be set back to the original one at the end of this context manager
        """
        if not enabled:
            yield
            return

        orig_rng_state = self.rng_manager.get_state()
        if seed != None:
            self.rng_manager.set_state(seed)
        try:
            yield
        finally:
            if reset_at_end:
                self.rng_manager.set_state(orig_rng_state)

    @contextmanager
    def fork_torch_rng_state(self, device, enabled=True):
        """
        Similar as torch.random.fork_rng, used with pre-defined device
        Now it is only used for maintain the same RNG state before/after tracing
        Args:
            device: the device this context manager is running with, cpu or cuda
        """
        if not enabled:
            yield
            return

        orig_rng_state = torch.get_rng_state()
        if device != torch.device("cpu"):
            with torch.cuda.device(device):
                orig_cuda_rng_state = torch.cuda.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(orig_rng_state)
            if device != torch.device("cpu"):
                with torch.cuda.device(device):
                    torch.cuda.set_rng_state(orig_cuda_rng_state)

    def get_model_parallel_rank(self):
        return self.core.mp_rank()

    def get_model_parallel_group(self):
        return self.mp_process_group

    def get_model_parallel_world_size(self):
        return self.core.mp_size()

    def get_data_parallel_rank(self):
        return self.core.rdp_rank()

    def get_data_parallel_group(self):
        return self.rdp_process_group

    def get_data_parallel_world_size(self):
        return self.core.rdp_size()


state = PTModelParallelState()
