# Third Party
# Standard Library
import warnings
from contextlib import contextmanager
from distutils.version import LooseVersion
from enum import Enum

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel as OriginalDistributedDataParallel

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.allreduce.ddp import DdpNonOverlappingAllreducer
from smdistributed.modelparallel.torch.core import core, dp_size, local_rank, rdp_size, tp_size
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import check_env_var_truthy, rmsg

logger = get_logger()

_pt_16 = LooseVersion("1.7.0") > LooseVersion(torch.__version__) >= LooseVersion("1.6.0")
_pt_17_or_newer = LooseVersion(torch.__version__) >= LooseVersion("1.7.0")
_pt_18_or_newer = LooseVersion(torch.__version__) >= LooseVersion("1.8.0")
_pt_19_or_newer = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")
_pt_110_or_newer = LooseVersion(torch.__version__) >= LooseVersion("1.10.0")

if _pt_18_or_newer and not _pt_110_or_newer:
    from torch.nn.parallel.distributed import _DDPUnevenInputsConfig


class ReducerType(Enum):
    SCALED_BATCH = 0
    DEFAULT = 1


class DistributedDataParallel(OriginalDistributedDataParallel):
    def __init__(
        self,
        module,
        broadcast_buffers=True,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
    ):
        # Ensure all relevant attributes from original constructor are set here
        # we can't call the constructor as is, because it expects all params on same device
        # and also the creation of reducer in _ddp_init_helper is a problem, we have our own reducer
        nn.Module.__init__(self)
        self.module = module

        self.reducer = None
        self.process_group = None
        self.modules_buffers = None

        self.scaled_batch_reducer = None
        self.default_reducer = None
        self._module_copies = [self.module]
        self.is_multi_device_module = False
        self.device = torch.device("cuda", local_rank())
        self.device_type = self.device.type
        self.device_ids = [local_rank()]
        self.output_device = self.device_ids[0]
        self.default_process_group = (
            state.dp_process_group
        )  # we do not name state.default_process_group because ddp explicitly uses this variable
        self.reducer_types = []
        self.scaled_batch_process_group = state.rdp_process_group
        self.default_reducer_modules_buffers = None
        self.scaled_batch_reducer_modules_buffers = None
        self.dim = 0
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.bucket_cap_mb = bucket_cap_mb
        self.broadcast_bucket_size = int(250 * 1024 * 1024)
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.bucket_rebuild_enabled = not check_env_var_truthy("SMP_DISABLE_BUCKET_REBUILD", "0")
        if _pt_110_or_newer:
            pass
        elif _pt_19_or_newer:
            self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
                ddp_join_enabled=False,
                ddp_join_divide_by_initial_world_size=False,
                ddp_join_throw_on_early_termination=False,
            )
        elif _pt_18_or_newer:
            self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
                ddp_join_enabled=False, ddp_join_divide_by_initial_world_size=False
            )
        elif _pt_17_or_newer:
            self.ddp_join_enabled = False
            self.ddp_join_divide_by_initial_world_size = False
        if _pt_110_or_newer:
            self._passing_sync_batchnorm_handle(self.module)
        else:
            self._passing_sync_batchnorm_handle(self._module_copies)
        if hasattr(self, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = self._ddp_params_and_buffers_to_ignore
        else:
            self.parameters_to_ignore = []

    def _create_ddp_reducer(
        self,
        grad_counter,
        average_grads_across_microbatches,
        overlapping_allreduce,
        scaled_batch_reducer_params,
        default_reducer_params,
        scaled_batch_reducer_buffers,
        default_reducer_buffers,
    ):
        scaled_batch_reducer_params_req_grads = [
            x.requires_grad for x in scaled_batch_reducer_params.values()
        ]
        default_reducer_params_req_grads = [
            x.requires_grad for x in default_reducer_params.values()
        ]

        named_default_reducer_buffers = [
            [(buffer, buffer_name) for buffer_name, buffer in default_reducer_buffers.items()]
        ]
        named_scaled_batch_reducer_buffers = [
            [(buffer, buffer_name) for buffer_name, buffer in scaled_batch_reducer_buffers.items()]
        ]
        self.default_reducer_modules_buffers = [
            [
                buffer
                for (buffer, buffer_name) in module_buffers
                if buffer_name not in self.parameters_to_ignore
            ]
            for module_buffers in named_default_reducer_buffers
        ]
        self.scaled_batch_reducer_modules_buffers = [
            [
                buffer
                for (buffer, buffer_name) in module_buffers
                if buffer_name not in self.parameters_to_ignore
            ]
            for module_buffers in named_scaled_batch_reducer_buffers
        ]

        if (
            any(default_reducer_params_req_grads) or any(scaled_batch_reducer_params_req_grads)
        ) and overlapping_allreduce:
            self._create_ddp_overlapping_reducer(
                grad_counter,
                average_grads_across_microbatches,
                any(default_reducer_params_req_grads),
                any(scaled_batch_reducer_params_req_grads),
            )
        else:
            state.model.overlapping_allreduce = False

            if len(scaled_batch_reducer_params) > 0:
                self.scaled_batch_reducer = DdpNonOverlappingAllreducer(
                    scaled_batch_reducer_params,
                    grad_counter,
                    average_grads_across_microbatches=average_grads_across_microbatches,
                    num_microbatches=state.num_microbatches(),
                    scaled_batch=True,
                    tp_size=tp_size(),
                    process_group=state.rdp_process_group,
                )
                self.reducer_types.append(ReducerType.SCALED_BATCH)

            if len(default_reducer_params) > 0:
                self.default_reducer = DdpNonOverlappingAllreducer(
                    default_reducer_params,
                    grad_counter,
                    average_grads_across_microbatches=average_grads_across_microbatches,
                    num_microbatches=state.num_microbatches(),
                    scaled_batch=False,
                    tp_size=tp_size(),
                    process_group=state.dp_process_group,
                )
                self.reducer_types.append(ReducerType.DEFAULT)

        if core.cfg._fp32_grad_accumulation:
            # For FP16, params are in FP16, but the grads in the bucket to be allreduced
            # are in FP32. Need to convert to FP16 before allreduce to reduce the communication cost
            # check torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py
            def allreduce_hook(
                process_group: object,
                bucket: dist._GradBucket if not _pt_19_or_newer else dist.GradBucket,
            ):
                # All processing on grad tensors including scaling should be handled
                # in the communication hook
                div_factor = (
                    tp_size() * process_group.size()
                    if process_group == state.rdp_process_group
                    else process_group.size()
                )
                if average_grads_across_microbatches:
                    div_factor = div_factor * state.num_microbatches()
                # Assumes that single device process for DDP, which is present throughout Rubik
                if _pt_110_or_newer:
                    compressed_tensor = bucket.buffer().to(torch.float16)
                elif _pt_19_or_newer:
                    compressed_tensor = bucket.get_tensor().to(torch.float16)
                else:
                    compressed_tensor = bucket.get_tensors()[0].to(torch.float16)
                compressed_tensor.div_(div_factor)
                fut = dist.all_reduce(
                    compressed_tensor, group=process_group, async_op=True
                ).get_future()

                def decompress(fut):
                    if _pt_110_or_newer:
                        decompressed_tensor = bucket.buffer()
                    elif _pt_19_or_newer:
                        decompressed_tensor = bucket.get_tensor()
                    else:
                        decompressed_tensor = bucket.get_tensors()[0]
                    # Decompress in place to reduce the peak memory.
                    # See: https://github.com/pytorch/pytorch/issues/45968
                    decompressed_tensor.copy_(fut.value()[0])
                    return [decompressed_tensor] if not _pt_19_or_newer else decompressed_tensor

                return fut.then(decompress)

            self.register_comm_hook(None, allreduce_hook)

    def _create_ddp_overlapping_reducer(
        self,
        grad_counter,
        average_grads_across_microbatches,
        create_default_reducer=True,
        create_scaled_batch_reducer=True,
    ):
        scaled_batch_condition = lambda m, p: state.model.is_scaled_batch_parameter(p)
        default_condition = lambda m, p: (not scaled_batch_condition(m, p))

        if create_scaled_batch_reducer:
            self.scaled_batch_reducer, self.scaled_batch_params = self._create_ddp_overlapping_reducer_with_condition(
                grad_counter,
                average_grads_across_microbatches,
                scaled_batch_condition,
                state.rdp_process_group,
                scaled_batch=True,
            )
            if _pt_18_or_newer and not _pt_19_or_newer:
                smplib._set_construction_logging_data(
                    self.scaled_batch_reducer,
                    self.module.__class__.__name__,
                    [] if self.device_ids is None else self.device_ids,
                    -1 if self.output_device is None else self.output_device,
                    self.broadcast_buffers,
                )
            self.reducer_types.append(ReducerType.SCALED_BATCH)
        if create_default_reducer:
            self.default_reducer, self.default_params = self._create_ddp_overlapping_reducer_with_condition(
                grad_counter,
                average_grads_across_microbatches,
                default_condition,
                state.dp_process_group,
                scaled_batch=False,
            )
            if _pt_18_or_newer and not _pt_19_or_newer:
                smplib._set_construction_logging_data(
                    self.default_reducer,
                    self.module.__class__.__name__,
                    [] if self.device_ids is None else self.device_ids,
                    -1 if self.output_device is None else self.output_device,
                    self.broadcast_buffers,
                )
            self.reducer_types.append(ReducerType.DEFAULT)

    def _create_ddp_overlapping_reducer_with_condition(
        self, grad_counter, average_grads_across_microbatches, condition, group, scaled_batch
    ):
        """
        Most code in this function is copied from DDP's init helper
        Some changes:
        - Tracking param_names and passing them to reducer
        - Using our reducer
        """
        # set helps ensure we add a param only once, if there are duplicate in list given to reducer
        # hook will be added multiple times
        params_set = set()
        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = []
        parameters = []
        param_names = []
        param_map = {}
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if (
                    param.requires_grad
                    and f"{module_name}.{param_name}" not in self.parameters_to_ignore
                    and param not in params_set
                ):
                    if condition(module, param):
                        params_set.add(param)
                        parameters.append(param)
                        param_names.append(state.model.get_param_name(param))
                        modules_and_parameters.append((module, param))

        # DDP reducer works with replica as first index, but we only have a single replica always
        # so increase level of nesting
        modules_and_parameters = [modules_and_parameters]
        for i in range(len(param_names)):
            param_map[i] = param_names[i]
        parameters = [parameters]
        param_names = [param_names]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, torch.nn.Embedding):
                return module.sparse
            if isinstance(module, torch.nn.EmbeddingBag):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            list(produces_sparse_gradient(module) for module, _ in replica)
            for replica in modules_and_parameters
        ]

        # The bucket size limit is specified in the constructor.
        # Additionally, we allow for a single small bucket for parameters
        # that are defined first, such that their gradients don't spill into
        # a much larger bucket, adding unnecessary latency after gradient
        # computation finishes. Experiments showed 1MB is a reasonable value.
        bucket_indices = smplib.compute_bucket_assignment_by_size(
            parameters[0],
            [smplib._DEFAULT_FIRST_BUCKET_BYTES, self.bucket_bytes_cap],
            expect_sparse_gradient[0],
            [],
        )

        args = [
            parameters,
            list(reversed(bucket_indices)),
            group,
            expect_sparse_gradient,
            self.bucket_bytes_cap,
        ]
        if _pt_16:
            pass
        elif _pt_17_or_newer:
            args.append(self.find_unused_parameters)
            args.append(self.gradient_as_bucket_view)
        else:
            raise RuntimeError("Unsupported Torch version")
        if _pt_19_or_newer:
            args.append(param_map)
        args.extend(
            [
                grad_counter,
                average_grads_across_microbatches,
                param_names,  # in same order as parameters
                state.num_microbatches(),
                scaled_batch,
                tp_size(),
                rdp_size() if scaled_batch else dp_size(),
                core.cfg._fp32_grad_accumulation,
            ]
        )
        if _pt_17_or_newer:
            args.insert(-1, core.cfg.shard_optimizer_state)

        return smplib.Reducer(*args), parameters

    @contextmanager
    def _reducer_type(self, reducer_type):
        existing_reducer = self.reducer
        existing_pg = self.process_group
        existing_modbuf = self.modules_buffers

        if reducer_type == ReducerType.SCALED_BATCH:
            self.reducer = self.scaled_batch_reducer
            self.process_group = self.scaled_batch_process_group
            self.modules_buffers = self.scaled_batch_reducer_modules_buffers
        else:
            self.reducer = self.default_reducer
            self.process_group = self.default_process_group
            self.modules_buffers = self.default_reducer_modules_buffers

        try:
            yield
        finally:
            self.reducer = existing_reducer
            self.process_group = existing_pg
            self.modules_buffers = existing_modbuf

    def _get_reducer_type(self, reducer):
        if reducer == self.scaled_batch_reducer:
            return ReducerType.SCALED_BATCH
        elif reducer == self.default_reducer:
            return ReducerType.DEFAULT
        else:
            raise RuntimeError("Unknown reducer type.")

    def _check_default_group(self):
        with self._reducer_type(ReducerType.DEFAULT):
            super(DistributedDataParallel, self)._check_default_group()

    def _broadcast_params_and_buffers(self):
        for reducer_type in [ReducerType.SCALED_BATCH, ReducerType.DEFAULT]:
            self._sync_params_and_buffers(reducer_type)

    def _sync_params_and_buffers(self, reducer_type, authoritative_rank=0):
        module_states = []
        scaled_batch_names = state.model._local_state_dict_nobool(
            cast_to_cpu=False, keep_vars=True, scaled_batch_only=True
        )

        for name, param in state.model._local_state_dict_nobool(cast_to_cpu=False).items():
            if name not in self.parameters_to_ignore:
                is_scaled_batch = name in scaled_batch_names
                is_scaled_batch_reducer = reducer_type == ReducerType.SCALED_BATCH

                if (is_scaled_batch and is_scaled_batch_reducer) or (
                    not is_scaled_batch and not is_scaled_batch_reducer
                ):
                    module_states.append(param.detach())

        if len(module_states) > 0:
            with self._reducer_type(reducer_type):
                args = [module_states, self.broadcast_bucket_size]
                if _pt_17_or_newer:
                    args.append(authoritative_rank)
                self._distributed_broadcast_coalesced(*args)

    # -----
    # Redirecting these methods so the extra wrapper is all hidden from the user
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.module.named_buffers(*args, **kwargs)

    def buffers(self, *args, **kwargs):
        return self.module.buffers(*args, **kwargs)

    def modules(self, *args, **kwargs):
        return self.module.modules(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    # -----

    # -----
    def _iterate_reducer_and_pg(self):
        for reducer, pg, modbuf in self._iterate_reducer_pg_buffers():
            yield reducer, pg

    def _iterate_reducer_pg_buffers(self):
        reducers = [self.scaled_batch_reducer, self.default_reducer]
        process_groups = [self.scaled_batch_process_group, self.default_process_group]
        modules_buffers = [
            self.scaled_batch_reducer_modules_buffers,
            self.default_reducer_modules_buffers,
        ]

        for reducer, pg, modbuf in zip(reducers, process_groups, modules_buffers):
            if reducer is not None:
                yield reducer, pg, modbuf

    # Overwriting below methods to make DDP code work with our non overlapping reducer
    def _match_unused_params_allreduce(self):
        if state.model.overlapping_allreduce:
            super(DistributedDataParallel, self)._match_unused_params_allreduce()

    def _match_all_reduce_for_bwd_pass(self):
        if state.model.overlapping_allreduce:
            super(DistributedDataParallel, self)._match_all_reduce_for_bwd_pass()
        else:
            self.reducer._match_all_reduce_for_bwd_pass()

    # -----

    def _sync_final_model(self, is_last_joiner):
        # Agree upon the process that will be the authoritative model copy.
        # The current rank is a candidate for being the authoritative copy if
        # is_last_joiner=True. We break ties via picking the larger rank.

        for reducer_type in self.reducer_types:
            with self._reducer_type(reducer_type):
                my_rank = dist.get_rank(self.process_group)
                self._authoritative_rank = self._find_common_rank(my_rank, is_last_joiner)
                self._sync_params_and_buffers(
                    reducer_type, authoritative_rank=self._authoritative_rank
                )

    # ---------------------------------------------
    # Same as in DDP 1.7, but redefined here so its available for PT1.6
    # ------
    def will_sync_module_buffers(self):
        return (
            self.require_forward_param_sync
            and self.broadcast_buffers
            and len(self.modules_buffers[0]) > 0
        )

    def _maybe_sync_buffers(self):
        for reducer_type in self.reducer_types:
            with self._reducer_type(reducer_type):
                if self.will_sync_module_buffers():
                    with torch.no_grad():
                        args = [self.modules_buffers[0], self.broadcast_bucket_size]
                        if _pt_17_or_newer:
                            if self._get_ddp_join_enabled():
                                if _pt_19_or_newer:
                                    authoritative_rank = self._find_common_rank(
                                        self._distributed_rank(), True
                                    )
                                else:
                                    authoritative_rank = self._find_common_rank(
                                        dist.get_rank(), True
                                    )
                            else:
                                # The process with rank 0 is considered the authoritative copy.
                                authoritative_rank = 0
                            args.append(authoritative_rank)

                        self._distributed_broadcast_coalesced(*args)

    # ---------------------------------------------

    # ---------------------------------------------
    # Redefining to link it to our register_comm_hook fn which accepts smp::torch::Reducer
    # ------
    # Technically this method has _ prefix in PT 1.6 and 1.7 but no prefix in PT 1.8
    # For simplicity dropping the prefix entirely. We can document this only for 1.8
    def register_comm_hook(self, state: object, hook: callable):
        self._check_comm_hook(hook)
        for reducer, pg in self._iterate_reducer_and_pg():
            curr_state = pg if not state else state
            smplib._register_comm_hook(reducer, curr_state, hook)

    # only in PT 1.8
    def _register_builtin_comm_hook(self, comm_hook_type):
        if _pt_18_or_newer:
            smplib._register_builtin_comm_hook(self.reducer, comm_hook_type)
        else:
            raise NotImplementedError(
                "_register_builtin_comm_hook is only supported with PyTorch 1.8 or newer"
            )

    def get_ddp_logging_data(self, reducer_type):
        if _pt_18_or_newer and not _pt_19_or_newer:
            if reducer_type == "default":
                return smplib._get_ddp_logging_data(self.default_reducer)
            elif reducer_type == "scaled_batch":
                return smplib._get_ddp_logging_data(self.scaled_batch_reducer)
            else:
                raise ValueError(
                    "reducer type can only be default or scaled_batch, but {} given".format(
                        reducer_type
                    )
                )
        else:
            raise NotImplementedError("_get_ddp_logging_data is only supported with PyTorch 1.8")

    # ---------------------------------------------

    # ---------------------------------------------
    # Added in PT 1.9
    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)

    # ---------------------------------------------

    # ---------------------------------------------
    # New methods used only by SMP
    # ------
    def _get_ddp_join_enabled(self):
        if _pt_110_or_newer:
            return False
        elif _pt_18_or_newer :
            return self.ddp_uneven_inputs_config.ddp_join_enabled
        elif _pt_17_or_newer:
            return self.ddp_join_enabled
        else:
            raise NotImplementedError

    def _get_ddp_join_divide_by_initial_world_size(self):
        if _pt_18_or_newer and not _pt_110_or_newer:
            return self.ddp_uneven_inputs_config.ddp_join_divide_by_initial_world_size
        elif _pt_17_or_newer:
            return self.ddp_join_divide_by_initial_world_size
        else:
            raise NotImplementedError

    def _pre_ddp_step(self, require_backward_grad_sync):
        self.require_backward_grad_sync = require_backward_grad_sync
        self._maybe_sync_buffers()

        if _pt_17_or_newer:
            if not _pt_110_or_newer and self._get_ddp_join_enabled():
                ones = [
                    torch.ones(1, device=torch.device("cuda", local_rank()))
                    for _ in self.reducer_types
                ]
                for i, (reducer, process_group) in enumerate(self._iterate_reducer_and_pg()):
                    work = dist.all_reduce(ones[i], group=process_group, async_op=True)
                    reducer._set_forward_pass_work_handle(
                        work, self._get_ddp_join_divide_by_initial_world_size()
                    )

                for reducer_type in self.reducer_types:
                    with self._reducer_type(reducer_type):
                        # Notify joined ranks whether they should sync in backwards pass or not.
                        self._check_global_requires_backward_grad_sync(is_joined_rank=False)

    def _post_ddp_step(self, require_forward_param_sync, require_backward_grad_sync):
        self.require_backward_grad_sync = require_backward_grad_sync
        self.require_forward_param_sync = require_forward_param_sync
        if self.bucket_rebuild_enabled:
            for reducer, _ in self._iterate_reducer_and_pg():
                if reducer._rebuild_buckets():
                    logger.info(rmsg("Reducer buckets have been rebuilt in this iteration."))

    # When running in join mode, schedules an allreduce to match the one in the
    # forward pass to determine the no. of currently active processes and whether
    # all processes have joined.
    def _schedule_shadow_all_reduce_for_fwd_pass(self):
        if len(self.reducer_types) == 0:
            # if there are no reducers (parameters), assume no active processes
            return 0

        all_active_procs = [torch.zeros(1, device=self.device) for r in self.reducer_types]
        for i, (_, process_group) in enumerate(self._iterate_reducer_and_pg()):
            dist.all_reduce(all_active_procs[i], group=process_group)
        return max([procs.item() for procs in all_active_procs])

    def _check_and_sync_module_buffers(self):
        for reducer_type in self.reducer_types:
            with self._reducer_type(reducer_type):
                super(DistributedDataParallel, self)._check_and_sync_module_buffers()

    if _pt_110_or_newer:

        @contextmanager
        def join(self, *args, **kwargs):
            raise NotImplementedError("join is currently only supported with PT 1.7-1.9")

    else:

        @contextmanager
        def join(
            self, divide_by_initial_world_size=True, enable=True, throw_on_early_termination=False
        ):
            """ Taken from ddp join() method, adapted for multi-reducer case."""
            if core.cfg._fp32_grad_accumulation:
                raise ValueError("join not supported when _fp32_grad_accumulation is enabled")
            try:
                if self.device_ids and len(self.device_ids) > 1:
                    raise ValueError(
                        """DDP join() API does not support Single-Process Multi-GPU
                        mode training. The recommended approach for DDP training is
                        to spawn a single process that works on a single GPU."""
                    )
                has_error = False

                # Todo: pt1.8 doesn't has different api for these two(i.)
                if _pt_19_or_newer:
                    self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
                        ddp_join_enabled=enable,
                        ddp_join_divide_by_initial_world_size=divide_by_initial_world_size,
                        ddp_join_throw_on_early_termination=throw_on_early_termination,
                    )
                elif _pt_18_or_newer:
                    self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
                        ddp_join_enabled=enable,
                        ddp_join_divide_by_initial_world_size=divide_by_initial_world_size,
                    )
                elif _pt_17_or_newer:
                    self.ddp_join_enabled = enable
                    self.ddp_join_divide_by_initial_world_size = divide_by_initial_world_size
                yield
            except Exception as e:
                # Set to skip any processing in the finally block.
                has_error = True
                raise e
            finally:
                # Skip any processing to let the exception immediately be raised if
                # there was one.
                if enable and not has_error:
                    all_procs_joined = False
                    is_last_joiner = True
                    i = 0
                    WARN_THRESHOLD = 1000
                    warnings.simplefilter("once")
                    while not all_procs_joined:
                        if i > WARN_THRESHOLD:
                            my_rank = dist.get_rank(self.default_process_group)
                            warnings.warn(
                                "Detected uneven input skew of greater "
                                f"than {WARN_THRESHOLD}. This means that rank {my_rank} "
                                f"has at least {WARN_THRESHOLD} fewer inputs than "
                                "other currently active ranks. This level of skew could "
                                "lead to performance degradation during training."
                            )
                        # Schedules allreduce to match fwd pass allreduce in non-joined procs
                        num_active_procs = self._schedule_shadow_all_reduce_for_fwd_pass()
                        if num_active_procs == 0:
                            all_procs_joined = True
                        else:
                            # Some DDP process still needs to be joined.
                            if is_last_joiner:
                                is_last_joiner = False
                            # It will rebuild buckets only once during training period
                            for reducer, _ in self._iterate_reducer_and_pg():
                                reducer._rebuild_buckets()
                            # Schedule a corresponding broadcast if we are syncing module
                            # buffers in the forward pass.

                            self._check_and_sync_module_buffers()

                            should_sync_tensors = []
                            works = []
                            for redtype in self.reducer_types:
                                with self._reducer_type(redtype):
                                    (
                                        work,
                                        should_sync_backwards_tensor,
                                    ) = self._check_global_requires_backward_grad_sync(
                                        is_joined_rank=True
                                    )
                                    works.append(work)
                                    should_sync_tensors.append(should_sync_backwards_tensor)
                            for work in works:
                                work.wait()

                            # If nonzero, then we should sync in the bwd pass.
                            should_sync_backwards = (
                                sum([tensor.item() for tensor in should_sync_tensors]) != 0
                            )

                            # Forward param sync is disabled in the next iteration
                            # if we are skipping grad sync this iteration. Hence, we
                            # set require_forward_param_sync appropriately here.
                            self.require_forward_param_sync = should_sync_backwards
                            if not should_sync_backwards:
                                continue
                            # Schedules one allreduce per gradient bucket to match
                            # the backwards pass allreduce.
                            for redtype in self.reducer_types:
                                with self._reducer_type(redtype):
                                    self._match_all_reduce_for_bwd_pass()
                            # Check if we need to allreduce locally unused params.
                            if self.find_unused_parameters:
                                for redtype in self.reducer_types:
                                    with self._reducer_type(redtype):
                                        self._match_unused_params_allreduce()
                            # It will push rebuilt params only once during training period
                            for reducer, _ in self._iterate_reducer_and_pg():
                                reducer._push_all_rebuilt_params()
                            i += 1

                    # All procs joined. Agree on authoritative rank and broadcast the model.
                    self._sync_final_model(is_last_joiner)

    # -----
    def _reset_join_state(self):
        if _pt_110_or_newer:
            return
        if _pt_19_or_newer:
            self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
                ddp_join_enabled=False,
                ddp_join_divide_by_initial_world_size=True,
                ddp_join_throw_on_early_termination=False,
            )
        elif _pt_18_or_newer:
            self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
                ddp_join_enabled=False, ddp_join_divide_by_initial_world_size=True
            )
        elif _pt_17_or_newer:
            self.ddp_join_enabled = False
            self.ddp_join_divide_by_initial_world_size = True

    # ---------------------------------------------
