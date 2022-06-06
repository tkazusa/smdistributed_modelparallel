# Standard Library
import itertools
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Optional, Sequence, Set, Union

# Third Party
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.comm import allgather, barrier
from smdistributed.modelparallel.torch.core import (
    core,
    dp_rank,
    dp_size,
    local_rank,
    pp_rank,
    rank,
    rdp_rank,
    rdp_size,
    tp_rank,
    tp_size,
)
from smdistributed.modelparallel.torch.ddp_model import DistributedDataParallel
from smdistributed.modelparallel.torch.exceptions import NumParametersNotMatch
from smdistributed.modelparallel.torch.nn import DistributedModule
from smdistributed.modelparallel.torch.patches.execution import distributed_backward
from smdistributed.modelparallel.torch.server import ExecutionServer
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import (
    check_env_var_truthy,
    collect_and_merge,
    collect_and_merge_in_group,
    dtype_size,
    raise_ddp_overlapping_exception,
    rmsg,
    slice_tp_tensors_in_state_dict,
)

logger = get_logger()

_pt_17_or_18_or_110_or_newer = (
    LooseVersion(torch.__version__) >= LooseVersion("1.7.0")
    and LooseVersion(torch.__version__) < LooseVersion("1.9.0")
) or (LooseVersion(torch.__version__) >= LooseVersion("1.10.0"))


class DistributedModel(nn.Module):
    def __init__(
        self,
        module,
        trace_device="gpu",
        trace_execution_times=False,
        trace_memory_usage=False,
        overlapping_allreduce=True,
        backward_passes_per_step=1,
        average_grads_across_microbatches=True,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        broadcast_buffers=True,
        gradient_as_bucket_view=True,
    ):
        """
        :param module: torch.nn.Module
        Module to be distributed
        DistributedModel wrapper takes care of both model and data parallelism.
        Data parallelism engines can be configured by setting `horovod` or `ddp` in smp config.

        :param trace_device: str or None
        Can be "gpu", "cpu", or None if you want to disable trace.

        """
        super(DistributedModel, self).__init__()
        self._validate(
            trace_device,
            trace_execution_times,
            overlapping_allreduce,
            backward_passes_per_step,
            gradient_as_bucket_view,
        )

        if not gradient_as_bucket_view and core.cfg._fp32_grad_accumulation:
            gradient_as_bucket_view = True
            get_logger().warning(
                "gradient_as_bucket_view=False may not provide much benefit with _fp32_grad_accumulation=True."
                "Force setting gradient_as_bucket_view=True since _fp32_grad_accumulation is set to True."
            )

        state.module_manager.simplify_tensor_parallelism_modules(module)
        state.model = self
        self._partitioned = False

        if state.cfg.tensor_parallel_degree > 1:
            dist_module = self._replace_tp_counterparts(module)
            if dist_module is not None:
                # entire module got replaced
                module = dist_module

        self._register_distributed_modules(module)

        if state.cfg.ddp:
            self.module = DistributedDataParallel(
                module,
                broadcast_buffers=broadcast_buffers,
                find_unused_parameters=find_unused_parameters,
                bucket_cap_mb=bucket_cap_mb,
                gradient_as_bucket_view=gradient_as_bucket_view,
            )
        else:
            self.module = module

        self._size = self._compute_model_size(module)

        self.overlapping_allreduce = overlapping_allreduce
        self.backward_passes_per_step = backward_passes_per_step
        self.average_grads_across_microbatches = average_grads_across_microbatches
        self.scaled_batch_reducer = None
        self.default_reducer = None

        self.trace_device = trace_device
        self.current_trace_device = trace_device
        self.trace_execution_times = trace_execution_times if self.trace_device == "gpu" else False
        self.trace_memory_usage = trace_memory_usage if self.trace_device == "gpu" else False

        state.exec_server = ExecutionServer()

        self._backward_passes_finished = 0
        self._param_names = {p: n for n, p in self.named_parameters()}
        self._local_parameters = OrderedDict()
        self._virtual_parameters = OrderedDict()
        self._scaled_batch_reducer_parameters = OrderedDict()
        self._default_reducer_parameters = OrderedDict()
        self._local_buffers = OrderedDict()
        self._scaled_batch_reducer_buffers = OrderedDict()
        self._default_reducer_buffers = OrderedDict()
        self._scaled_batch_reducer_param_set = set()
        self._default_reducer_param_set = set()
        self._scaled_batch_reducer_buffer_set = set()
        self._default_reducer_buffer_set = set()
        self._partitions_assigned = False
        self._moved_to_contiguous_buffer = False
        self._called_post_step_hooks = False
        self._has_reducer_parameters = True

        state.module_manager.assign_partition(self, 0)
        state.patch_manager.save_original_methods_for_model(model=self)
        state.patch_manager.patch_forward(model=self)
        state.patch_manager.patch_to_and_moves(model=self)
        self.param_buffer = None
        self.dp_rank_to_count = {}
        self.reducer_type_to_count = {}

        state.tp_registry.reset()

        if not state.cfg.auto_partition:
            state.module_manager.assign_unassigned_modules(state.model)
            state.module_manager.name_modules_and_create_parent_map()
            self._partitions_assigned = True
        state.module_manager.clear_tensor_parallelism_modules()

    def _validate(
        self,
        trace_device,
        trace_execution_times,
        overlapping_allreduce,
        backward_passes_per_step,
        gradient_as_bucket_view,
    ):
        if state.model is not None:
            raise RuntimeError(
                "Using DistributedModel wrapper more than once in a process is not supported yet"
            )
        if trace_device not in ["cpu", "gpu"]:
            raise ValueError("Trace_device must be one of `cpu` or `gpu`")
        if overlapping_allreduce not in [True, False]:
            raise ValueError("overlapping_allreduce can only take one of `True` or `False`")
        if trace_device == "cpu" and trace_execution_times == True:
            get_logger().warning(
                "Tracing execution time is not supported when trace device is set to CPU."
            )
        assert isinstance(
            backward_passes_per_step, int
        ), "backward_passes_per_step needs to be an int"

        if not overlapping_allreduce and core.cfg._fp32_grad_accumulation:
            raise_ddp_overlapping_exception(
                "_fp32_grad_accumulation=True with DistributedModel.__init__"
            )
        if core.cfg.shard_optimizer_state and not gradient_as_bucket_view:
            raise ValueError(
                "gradient_as_bucket_view must be True when shard_optimizer_state is True."
            )

    def _replace_tp_counterparts(self, module, name=None, memo=None):
        """
        Replace modules that are supported and marked for tensor parallelism with their
        distributed counterparts. name and memo arguments are used internally so should not
        be supplied with anything in external calls. The original modules are garbage collected
        unless the user explicitly maintains a reference to them. Under module reuse, the same
        distributed module is reused for both usages of the module.
        """

        if name is None:
            name = "main"
        if memo is None:
            memo = {}

        skip_param_checking = check_env_var_truthy("SMP_SKIP_PARAMS_CHECKING", "1")

        # handle the case where the top level module can be distributed
        if state.module_manager.should_tensor_parallelize(module):
            prev_num_params = self._check_num_params(module, name)
            distributed_module = state.tp_registry.distribute(module)
            self._inherit_original_module_partition(module, distributed_module, name)
            post_num_params = self._check_num_params(distributed_module, name, True)
            if not skip_param_checking:
                if prev_num_params != post_num_params:
                    raise NumParametersNotMatch(prev_num_params, post_num_params)
            return distributed_module

        module_children = [c for c in module.named_children()]
        for child_name, child in module_children:
            full_name = ".".join([name, child_name])
            distributed_child = None
            if child in memo:
                # use previously distributed module to handle module reuse
                distributed_child = memo[child]
            elif state.module_manager.should_tensor_parallelize(child):
                prev_num_params = self._check_num_params(child, full_name)
                distributed_child = state.tp_registry.distribute(child)
                post_num_params = self._check_num_params(distributed_child, full_name, True)
                if not skip_param_checking:
                    if prev_num_params != post_num_params:
                        raise NumParametersNotMatch(prev_num_params, post_num_params)
                memo[child] = distributed_child

            if distributed_child is not None:
                self._inherit_original_module_partition(child, distributed_child, full_name)
                module._modules[child_name] = distributed_child
                setattr(module, child_name, distributed_child)
            else:
                self._replace_tp_counterparts(child, full_name, memo)

    def _compute_model_size(self, module):
        num_bytes = 0
        for p in module.parameters():
            num_bytes += np.prod(p.shape) * dtype_size(p.dtype)
        return num_bytes

    def _check_num_params(self, module, full_name, dist=False):
        num_params = 0.0
        if dist:
            for p in module.parameters():
                if self.is_distributed_parameter(p):
                    num_params += np.prod(p.shape)
                elif tp_rank() == 0:
                    num_params += np.prod(p.shape)
            x = allgather(num_params, group=CommGroup.TP_GROUP)
            res = 0
            for i in x:
                res += i
        else:
            for p in module.parameters():
                num_params += np.prod(p.shape)
            res = num_params
        return res

    def _register_distributed_modules(self, module):
        """ This should not happen during _replace_tp_counterparts call to handle the case where the DistributedModule
            is directly called in the training script. """

        for child in module.children():
            if isinstance(child, DistributedModule):
                state.module_manager.register_distributed(child)
            else:
                self._register_distributed_modules(child)

    def _inherit_original_module_partition(self, module, distributed_module, module_name):
        def _has_assigned_partition(mod):
            return (
                mod in state.module_manager._module_partitions
                and state.module_manager._module_partitions[mod] is not None
            )

        if state.cfg.auto_partition:
            return

        if _has_assigned_partition(module):
            partition = state.module_manager._module_partitions[module]
            for m in distributed_module.modules():
                state.module_manager.assign_partition(m, partition)
        else:
            for name, m in module.named_modules():
                if rank() == 0 and _has_assigned_partition(m):
                    submodule_name = ".".join([module_name, name])
                    logger.warning(
                        f"Ignoring the manual partition assignment for the module {submodule_name} since its outer module {module_name} is being distributed as part of tensor parallelism. You can use smp.set_partition API after smp.DistributedModel call to re-assign the sub-modules of {module_name} to a desired partition."
                    )

    def _create_reducer(self):
        self.grad_counter = smplib.GradCounter(
            [n for n, p in self.local_named_parameters()], state.cfg.microbatches
        )

        if state.cfg.horovod:
            from smdistributed.modelparallel.torch.allreduce.horovod import HorovodAllreducer

            if len(self._scaled_batch_reducer_parameters) > 0:
                self.scaled_batch_reducer = HorovodAllreducer(
                    self._scaled_batch_reducer_parameters,
                    self.grad_counter,
                    overlapping_allreduce=self.overlapping_allreduce,
                    average_grads_across_microbatches=self.average_grads_across_microbatches,
                    num_microbatches=state.num_microbatches(),
                    scaled_batch=True,
                    tp_size=tp_size(),
                )
            if len(self._default_reducer_parameters) > 0:
                self.default_reducer = HorovodAllreducer(
                    self._default_reducer_parameters,
                    self.grad_counter,
                    overlapping_allreduce=self.overlapping_allreduce,
                    average_grads_across_microbatches=self.average_grads_across_microbatches,
                    num_microbatches=state.num_microbatches(),
                    scaled_batch=False,
                    tp_size=tp_size(),
                )

        elif state.cfg.ddp:
            self.module._create_ddp_reducer(
                self.grad_counter,
                average_grads_across_microbatches=self.average_grads_across_microbatches,
                overlapping_allreduce=self.overlapping_allreduce,
                scaled_batch_reducer_params=self._scaled_batch_reducer_parameters,
                default_reducer_params=self._default_reducer_parameters,
                scaled_batch_reducer_buffers=self._scaled_batch_reducer_buffers,
                default_reducer_buffers=self._default_reducer_buffers,
            )
            self.scaled_batch_reducer = self.module.scaled_batch_reducer
            self.default_reducer = self.module.default_reducer
        elif state.cfg.herring:
            from smdistributed.modelparallel.torch.allreduce.herring import HerringAllreducer

            if len(self._scaled_batch_reducer_parameters) > 0:
                self.scaled_batch_reducer = HerringAllreducer(
                    self._scaled_batch_reducer_parameters,
                    self.grad_counter,
                    overlapping_allreduce=self.overlapping_allreduce,
                    average_grads_across_microbatches=self.average_grads_across_microbatches,
                    num_microbatches=state.num_microbatches(),
                    scaled_batch=True,
                    tp_size=tp_size(),
                )
            if len(self._default_reducer_parameters) > 0:
                self.default_reducer = HerringAllreducer(
                    self._default_reducer_parameters,
                    self.grad_counter,
                    overlapping_allreduce=self.overlapping_allreduce,
                    average_grads_across_microbatches=self.average_grads_across_microbatches,
                    num_microbatches=state.num_microbatches(),
                    scaled_batch=False,
                    tp_size=tp_size(),
                )

        self._has_reducer_parameters = not (
            self.scaled_batch_reducer is None and self.default_reducer is None
        )
        if not self._has_reducer_parameters:
            # for non DP case set this to False
            self.overlapping_allreduce = False
        if self.scaled_batch_reducer is None:
            from smdistributed.modelparallel.torch.allreduce.scaler import GradScaler

            self.scaled_batch_reducer = GradScaler(
                self._scaled_batch_reducer_parameters,
                self.grad_counter,
                overlapping_allreduce=False,
                average_grads_across_microbatches=self.average_grads_across_microbatches,
                num_microbatches=state.num_microbatches(),
                scaled_batch=True,
                tp_size=tp_size(),
            )
        if self.default_reducer is None:
            from smdistributed.modelparallel.torch.allreduce.scaler import GradScaler

            self.default_reducer = GradScaler(
                self._default_reducer_parameters,
                self.grad_counter,
                overlapping_allreduce=False,
                average_grads_across_microbatches=self.average_grads_across_microbatches,
                num_microbatches=state.num_microbatches(),
                scaled_batch=False,
                tp_size=tp_size(),
            )

    def _populate_local_parameters(self):
        local_params = set()
        for m in self.module.modules():
            if state.module_manager.get_partition(m) == pp_rank():
                for n, p in m.named_parameters(recurse=False):
                    local_params.add(p)
                    if state.module_manager.is_scaled_batch_parameter(p):
                        assert state.module_manager.is_distributed(
                            m
                        ), "Scaled batch parameter has to be in distributed module."
                        self._scaled_batch_reducer_param_set.add(p)
                    else:
                        self._default_reducer_param_set.add(p)

        for n, p in self.named_parameters():
            if p in local_params:
                assert n not in self._local_parameters
                self._local_parameters[n] = p
            if p in self._scaled_batch_reducer_param_set:
                self._scaled_batch_reducer_parameters[n] = p
            if p in self._default_reducer_param_set:
                self._default_reducer_parameters[n] = p

        if dp_rank() == 0:
            logger.info(
                f"Number of parameters on partition {pp_rank()} are {len(self._local_parameters)}. "
                f"{len([x for x in self._local_parameters.values() if x.requires_grad])} require grads"
            )

    def _populate_local_buffers(self):
        local_buffers = set()
        for m in self.module.modules():
            if state.module_manager.get_partition(m) == pp_rank():
                for n, b in m.named_buffers(recurse=False):
                    local_buffers.add(b)
                    if state.module_manager.is_scaled_batch_buffer(b):
                        assert state.module_manager.is_distributed(
                            m
                        ), "Scaled batch buffer has to be in distributed module."
                        self._scaled_batch_reducer_buffer_set.add(b)
                    else:
                        self._default_reducer_buffer_set.add(b)

        for n, b in self.named_buffers():
            if b in local_buffers:
                assert n not in self._local_buffers
                self._local_buffers[n] = b
            if b in self._scaled_batch_reducer_buffer_set:
                self._scaled_batch_reducer_buffers[n] = b
            if b in self._default_reducer_buffer_set:
                self._default_reducer_buffers[n] = b

        if dp_rank() == 0 and len(self._local_buffers):
            logger.info(
                f"Number of buffers on partition {pp_rank()} are {len(self._local_buffers)}. "
            )

    def _disable_grad_for_nonlocal(self):
        # set requires_grad=False for parameters assigned to other ranks
        # to prevent allreducing them
        for m in self.module.modules():
            if state.module_manager.get_partition(m) != pp_rank():
                for p in m.parameters(recurse=False):
                    p.requires_grad = False
                    p.grad = None

    def _compute_descendant_partitions(self, module):
        partitions = {}
        module_partitions = set()
        for c in module.children():
            child_partitions, all_partitions = self._compute_descendant_partitions(c)
            partitions.update(all_partitions)
            module_partitions = module_partitions.union(child_partitions)
        module_partitions.add(state.module_manager.get_partition(module))
        partitions[module] = module_partitions

        return module_partitions, partitions

    def size(self):
        """ Size of the full model in bytes """
        return self._size

    def display_partition(self):
        """ Display the truncated partition tree. This version only runs in manual partition. """

        _, descendant_partitions = self._compute_descendant_partitions(self)

        queue = deque()

        # bfs traversal until all the subtree is assigned to a single partition
        logger.info("Partition assignments:")
        queue.append(self)
        visited = set()
        while len(queue) > 0:
            m = queue.popleft()
            if m not in visited:
                visited.add(m)
                name = state.module_manager.get_module_name(m)
                partition = state.module_manager.get_partition(m)
                logger.info(f"{name}: {partition}")
                if len(descendant_partitions[m]) > 1:
                    queue.extend(m.children())

        if state.cfg.tensor_parallel_degree > 1:
            logger.info("Tensor-parallel distributed modules:")
            queue.append(self)
            visited = set()
            while len(queue) > 0:
                m = queue.popleft()
                if m not in visited:
                    visited.add(m)
                    if isinstance(m, DistributedModule):
                        name = state.module_manager.get_module_name(m)
                        logger.info(name)
                    else:
                        queue.extend(m.children())

    def post_partition(self):
        """
        Comes here for both auto and manual partitioning
        """
        self._partitioned = True

        if rank() == 0:
            self.display_partition()

        self._disable_grad_for_nonlocal()

        # move params to right device
        self.module.to(torch.device("cuda", local_rank()))

        self._populate_local_parameters()
        self._populate_local_buffers()

        with torch.no_grad():
            for p in self.local_parameters():
                p.data = p.data.clone()

                if p in state.param_initializers:
                    state.param_initializers[p](p.data)

            state.param_initializers = {}

        barrier()
        if rank() == 0:
            logger.info(f"Finished partitioning the model")

        # setting allreduce hooks as needed, we wanted to delay this till module assignment is done
        self._create_reducer()

        self._broadcast_parameters_and_buffers()
        if dp_size() > 1:
            if dp_rank() == 0:
                logger.info(f"Broadcasted parameters and buffers for partition {pp_rank()}")

        for hook in state.module_manager._post_partition_hooks.values():
            hook(self, state.optimizer)

    def iterate_reducer_types(self):
        assert state.cfg.ddp, "DDP must be enabled to iterate reducer types"

        for reducer, _ in self.module._iterate_reducer_and_pg():
            yield self.module._get_reducer_type(reducer)

    def move_local_modules_to_contiguous_buffer(self, param_name_to_offset):
        from smdistributed.modelparallel.torch.ddp_model import ReducerType

        self.group_size = defaultdict(int)
        self.param_name_to_offset = defaultdict(dict)
        self.group_offset = {}
        cum_offset = 0
        for red_type in self.iterate_reducer_types():
            self.group_offset[red_type] = cum_offset
            for _, p in self._local_unique_named_parameters(red_type):
                self.group_size[red_type] += p.numel()
            size = dp_size() if red_type == ReducerType.DEFAULT else rdp_size()
            self.group_size[red_type] = (self.group_size[red_type] + size - 1) // size * size
            cum_offset += self.group_size[red_type]

        dtype = next(self.module.parameters()).dtype

        # allocate contiguous buffer
        self.param_buffer = torch.empty(
            cum_offset, device=torch.device("cuda", local_rank()), dtype=dtype, requires_grad=True
        )

        # redefine the parameters to point to the contiguous buffer
        with torch.no_grad():
            for red_type in self.iterate_reducer_types():
                for n, p in self._local_unique_named_parameters(red_type):
                    param_offset = param_name_to_offset[red_type][n]
                    self.param_name_to_offset[red_type][n] = param_offset
                    buffer_offset = self.param_buffer.narrow(
                        0, self.group_offset[red_type] + param_offset, p.numel()
                    )
                    param_view = buffer_offset.view(*p.shape)
                    param_view.copy_(p.data)
                    p.data = param_view

    def _local_unique_named_parameters(self, reducer_type=None):
        from smdistributed.modelparallel.torch.ddp_model import ReducerType

        def is_correct_parameter_type(par):
            if reducer_type == None:
                return True

            if reducer_type == ReducerType.SCALED_BATCH and self.is_scaled_batch_parameter(par):
                return True

            if reducer_type == ReducerType.DEFAULT and self.is_scaled_batch_parameter(par) is False:
                return True

            return False

        seen = set()
        for n, p in self.local_named_parameters():
            if id(p) not in seen and is_correct_parameter_type(p):
                seen.add(id(p))
                yield n, p

    def load_partition(self, partitioning_and_trace_results=None):
        if partitioning_and_trace_results is not None:
            # if partitioning_and_trace_results is None then assumes that module assignment is done on that rank
            state.module_manager.load_partitioning_and_trace_results(partitioning_and_trace_results)

    def _broadcast_parameters_and_buffers(self):
        # we broadcast needed params and buffers ourselves so users dont end up broadcasting cpu params
        local_state = self._local_state_dict_nobool(cast_to_cpu=False)
        if state.cfg.horovod:
            import horovod.torch as hvd

            hvd.broadcast_parameters(local_state, root_rank=0)
        elif state.cfg.ddp:
            self.module._broadcast_params_and_buffers()
        elif state.cfg.herring:
            raise NotImplementedError

    def state_dict(
        self, destination=None, prefix="", keep_vars=False, cast_to_cpu=True, gather_to_rank0=False
    ):
        """Get the global state_dict"""
        if not self.partitioned:
            # use full state dict, but tp state will be 1/8 hence we need allgather at the end
            local_state_dict = self.module.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
            if cast_to_cpu:
                for name, param in local_state_dict.items():
                    if isinstance(param, torch.Tensor):
                        local_state_dict[name] = param.cpu()

            logger.warning(
                rmsg(
                    f"state_dict() was called without training for any step. This means model is still not partitioned."
                    f"If you just loaded a checkpoint, this is okay. But if you are calling this right after model initialization,"
                    f"and have different seeds for different pipeline parallel ranks, then note that the parameter returned here"
                    f"by the rank calling state_dict may not be the same as the parameter which will be used for training by the "
                    f"rank which owns this parameter after partitioning. "
                    f"Typically though, we set the same seed across pipeline parallel ranks."
                )
            )
        else:
            local_state_dict = self.local_state_dict(
                cast_to_cpu=cast_to_cpu, destination=destination, prefix=prefix, keep_vars=keep_vars
            )
            del local_state_dict["_smp_is_partial"]
            del local_state_dict["_smp_load_info"]

        msg = "rdp_rank()" if tp_size() > 1 else "dp_rank()"
        logger.info(
            rmsg(
                f"Gathering model state_dict during saving. To prevent hangs, please ensure that model.state_dict() "
                f"(where model is smp.DistributedModel wrapped model) is called on all the ranks with {msg} == 0"
            )
        )
        from smdistributed.modelparallel.torch.comm import allgather, gather

        collect_fn = gather if gather_to_rank0 else allgather
        if gather_to_rank0:
            logger.warning(
                rmsg(
                    "gather_to_rank0 is set to True. Full state_dict will only be saved to rank 0 only, other ranks will have empty dicts"
                )
            )
        full_state_dict = collect_and_merge(
            local_state_dict, model_partitioned=self.partitioned, collect_fn=collect_fn
        )
        return full_state_dict

    def load_state_dict(self, state_dict, strict=True, same_partition_load=False):
        """
        load the state_dict, if it is a local state_dict, allgather to get the globale state_dict
        This function does not load the state_dict into the model. The real loading happens
        after the partition is finished, so each rank knows which parameters to load.
        """
        if same_partition_load:
            if "_smp_load_info" in state_dict:
                tensor_parallel_degree = state_dict["_smp_load_info"]["tensor_parallel_degree"]
                pipeline_parallel_degree = state_dict["_smp_load_info"]["pipeline_parallel_degree"]
                same_partition_load = (
                    tensor_parallel_degree == state.cfg.tensor_parallel_degree
                    and pipeline_parallel_degree == state.cfg.pipeline_parallel_degree
                )
                if same_partition_load:
                    logger.warning(
                        rmsg(
                            f"same_partition_load is enabled, ensure that the model partitioning decision hasn't changed between the save and load runs."
                        )
                    )
                else:
                    logger.warning(
                        rmsg(
                            f"same_partition_load is disabled. To use same_partition_load, ensure that the tensor_parallel_degree and pipeline_parallel_degree are not changed between the save and load runs."
                        )
                    )
                smp_load_info = state_dict["_smp_load_info"]
                del state_dict["_smp_load_info"]
            else:
                logger.warning(
                    rmsg(
                        f"same_partition_load is enabled but smp load info is not saved, fall back to load with allgather. To use same_partition_load, make sure the checkpoint is saved with the most recent smp release"
                    )
                )
                same_partition_load = False
        else:
            if "_smp_load_info" in state_dict:
                logger.info(
                    rmsg(
                        f"same_partition_load is available which could skip the allgather call in loading. To enable, set same_partition_load=True in model.load_state_dict()"
                    )
                )
                del state_dict["_smp_load_info"]

        is_partial = "_smp_is_partial" in state_dict and state_dict["_smp_is_partial"]
        if "_smp_is_partial" in state_dict:
            del state_dict["_smp_is_partial"]

        if is_partial:
            if not same_partition_load:
                msg = "rdp_rank()" if tp_size() > 1 else "dp_rank()"
                logger.info(
                    rmsg(
                        f"Gathering model state_dict during loading. To prevent hangs, please ensure that model.load_state_dict() "
                        f"(where model is smp.DistributedModel wrapped model) is called on all the ranks with {msg} == 0"
                    )
                )
                state_dict = collect_and_merge(state_dict)
            else:
                # Collect the remote varibale/buffers
                full_state_dict = self.module.state_dict()
                for name in smp_load_info["remote_names"]:
                    if name in state_dict:
                        raise ValueError(
                            f"remote variable/buffer {name} should not exist in local dict!"
                        )
                    if name not in full_state_dict:
                        raise ValueError(
                            f"full state dict should contain the remote variable/buffer {name}!"
                        )
                    state_dict[name] = full_state_dict[name]

        def _load_callable(model, _):
            if tp_size() > 1 and not same_partition_load:
                tp_split_state_dict = slice_tp_tensors_in_state_dict(state_dict)
                if tp_rank() == 0:
                    model.module.load_state_dict(tp_split_state_dict, strict=strict)
                else:
                    # Pytorch returns missing_keys and unexpected_keys from load state dict
                    # Check if len(missing_keys) == 0. unexpected keys are fine, since
                    # there are params which are set only for tp_rank() == 0.
                    keys = model.module.load_state_dict(tp_split_state_dict, strict=False)
                    if keys:
                        missing_keys, unexpected_keys = keys
                        if len(missing_keys) > 0 and strict:
                            raise RuntimeError(
                                f"len of missing keys greater than 0, {missing_keys} when loading."
                                f"state_dict on {tp_rank()}, please check if keys are missing from the state_dict."
                                f"If this is intended, set strict to False when calling load_state_dict on the DistributedModel"
                            )
            else:
                model.module.load_state_dict(state_dict, strict=strict)
            log = rmsg("Loaded model state_dict")
            logger.info(log) if rank() == 0 else logger.debug(log)

        self._call_after_partitioning(_load_callable)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.module.named_buffers(*args, **kwargs)

    def buffers(self, *args, **kwargs):
        return self.module.buffers(*args, **kwargs)

    def _mark_backward_in_step(self, val=True):
        self._step_had_backward = val

    def backward(
        self,
        tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
        grad_tensors: Union[torch.Tensor, Sequence[torch.Tensor], None] = None,
    ) -> None:
        if state.is_tracing():
            return

        if not state.in_step_func:
            raise RuntimeError(
                "smp.DistributedModel's backward can only be run inside smp.step annotated function"
            )
        self._mark_backward_in_step()
        distributed_backward(self, tensors, grad_tensors)

    def _call_after_partitioning(self, callable_fn):
        if self.partitioned:
            callable_fn(self, state.optimizer)
        else:
            self.register_post_partition_hook(callable_fn)

    @contextmanager
    def _step(self):
        self._mark_backward_in_step(False)

        if (
            (self._backward_passes_finished + 1) % self.backward_passes_per_step
        ) == 0 and torch.is_grad_enabled():
            self.require_backward_grad_sync = True
        else:
            self.require_backward_grad_sync = False

        def _prep_backward(self, optim):
            if state.model.overlapping_allreduce and self.require_backward_grad_sync:
                if self.default_reducer is not None:
                    self.default_reducer.prepare_for_backward()
                if self.scaled_batch_reducer is not None:
                    self.scaled_batch_reducer.prepare_for_backward()

        try:
            if state.cfg.ddp:

                def _pre_ddp_step(self, optimizer):
                    self.module._pre_ddp_step(
                        require_backward_grad_sync=self.require_backward_grad_sync
                    )

                self._call_after_partitioning(_pre_ddp_step)

            self._call_after_partitioning(_prep_backward)

            if state.skip_metadata_transmission():
                smplib.smp_torch_register_minibatch_preemptive_receptions(
                    state.current_step_func().id
                )
                state.register_minibatch_link_ids()

            yield

        finally:
            # We want to synchronize and ensure all grad communication is finished
            # so users can check the grads after step method.
            # If there are any unused params, they are also allreduced with whatever (0?) grad they have, in line with hvd and ddp.
            # This could cause issues if there arise situations when we think backward is done but is not actually done,
            # then we allreduce some params which didn't have all their grads computed. Its okay if they were never going to come
            # because of partial backward, but is a problem if they were going to come later.
            if self._step_had_backward:
                self._backward_passes_finished += 1

            if self._step_had_backward and self.require_backward_grad_sync:
                if self.scaled_batch_reducer is not None:
                    self.scaled_batch_reducer.synchronize()
                if self.default_reducer is not None:
                    self.default_reducer.synchronize()
                self._backward_passes_finished = 0
                require_forward_param_sync = True
            else:
                require_forward_param_sync = False

            if state.cfg.ddp:
                self.module._post_ddp_step(
                    require_forward_param_sync=require_forward_param_sync,
                    require_backward_grad_sync=self.require_backward_grad_sync,
                )

            if state.cfg.shard_optimizer_state:
                self._handle_opt_state_sharding()

            if not self._called_post_step_hooks:
                for hook in state.module_manager._post_step_hooks.values():
                    hook(self, state.optimizer)
                self._called_post_step_hooks = True

    def _handle_opt_state_sharding(self):
        if not self._moved_to_contiguous_buffer:
            if not _pt_17_or_18_or_110_or_newer:
                raise RuntimeError(
                    "Optimizer state sharding is currently supported for PyTorch 1.7, 1.8 or >=1.10 only."
                )

            assert state.cfg.ddp, "DDP must be enabled when optimizer state sharding is enabled."

            assert (
                self.overlapping_allreduce
            ), "Optimizer state sharding is currently only supported for overlapping allreduce."

            dtype = None
            name = None
            for n, p in self.named_parameters():
                if dtype is None:
                    dtype = p.dtype
                    name = n
                else:
                    if p.dtype != dtype:
                        raise TypeError(
                            f"Currently shard_optimizer_state is supported only when all parameters in the model have the same dtype. Found {name} with {dtype}, and {n} with {p.dtype}"
                        )

            param_name_to_offset = {}
            for reducer, _ in self.module._iterate_reducer_and_pg():
                red_type = self.module._get_reducer_type(reducer)
                param_name_to_offset[red_type] = reducer.get_param_name_to_offset()

            self.move_local_modules_to_contiguous_buffer(param_name_to_offset)
            self.create_virtual_parameters()

            self._moved_to_contiguous_buffer = True

    def create_virtual_parameters(self):
        """ Create virtual parameters that share storage with the original parameters, but at each rank corresponds only to
        subset of the parameters that fall within the (optimizer state) shard of the current rank. These virtual parameters
        will be used in the optimizer. """

        from smdistributed.modelparallel.torch.ddp_model import ReducerType

        grad_buffer = {}
        for reducer, _ in self.module._iterate_reducer_and_pg():
            red_type = self.module._get_reducer_type(reducer)
            grad_buffer[red_type] = reducer.get_grad_buffer()

        group_params = defaultdict(list)
        # filter params
        for red_type in self.iterate_reducer_types():
            rank = dp_rank() if red_type == ReducerType.DEFAULT else rdp_rank()
            size = dp_size() if red_type == ReducerType.DEFAULT else rdp_size()
            group_offset = self.group_offset[red_type]
            group_size = self.group_size[red_type]

            shard_lower_bound = group_offset + group_size // size * rank
            shard_upper_bound = group_offset + group_size // size * (rank + 1)

            reducer_params = {n for n, _ in self._local_unique_named_parameters(red_type)}

            start_index = 0
            for group_index, group in enumerate(state.optimizer.param_groups):
                new_params = []
                for i, p in enumerate(group["params"]):
                    name = state.optimizer.index_to_name[start_index + i]
                    if name not in reducer_params:
                        continue

                    param_lower_bound = self.param_name_to_offset[red_type][name] + group_offset
                    param_upper_bound = (
                        p.numel() + self.param_name_to_offset[red_type][name] + group_offset
                    )

                    new_param_lower_bound = max(param_lower_bound, shard_lower_bound)
                    new_param_upper_bound = min(param_upper_bound, shard_upper_bound)
                    new_param_size = new_param_upper_bound - new_param_lower_bound

                    if new_param_size <= 0:
                        # this means the parameter lies completely outside of the shard of the current rank.
                        # we do not create a virtual parameter in this case.
                        continue

                    new_param = nn.Parameter(
                        self.param_buffer.narrow(0, new_param_lower_bound, new_param_size)
                    )
                    new_grad = grad_buffer[red_type].narrow(
                        0, new_param_lower_bound - group_offset, new_param_size
                    )
                    new_param.grad = new_grad.view(new_param.shape)
                    new_params.append(new_param)
                    self._virtual_parameters[name] = new_param

                    dist_axis = state.module_manager.get_parameter_distribution_axis(p)
                    if dist_axis is not None:
                        state.module_manager.add_distributed_parameter(new_param, dist_axis)

                start_index += len(group["params"])
                group_params[group_index].extend(new_params)

        for group_index, group in enumerate(state.optimizer.param_groups):
            group["params"] = group_params[group_index]
        state.optimizer.redefined_params = True

    def forward(self, *args, **kwargs):
        if state.cfg.pipeline_parallel_degree > 1 and not state.in_step_func:
            raise RuntimeError(
                "SMP DistributedModel forward can only be run from inside a smp.step annotated function when pipeline parallelism degree is more than 1."
            )
        return self.module(*args, **kwargs)

    @property
    def partitioned(self):
        return self._partitioned

    """
    Creating new methods below so its clear that these methods can only be called on DistributedModel
    """

    def _validate_partitioned(self):
        if not self.partitioned:
            raise RuntimeError(
                "Model has not been partitioned yet. You can call this method after first step when using autopartitioning."
            )

    def local_modules(self):
        self._validate_partitioned()
        for m in self.module.modules():
            if state.module_manager.get_partition(m) == pp_rank():
                yield m

    def local_named_modules(self, memo: Optional[Set[nn.Module]] = None, prefix: str = ""):
        self._validate_partitioned()
        for n, m in self.module.named_modules(memo=memo, prefix=prefix):
            if state.module_manager.get_partition(m) == pp_rank():
                yield n, m

    def virtual_named_parameters(self):
        for n, p in self._virtual_parameters.items():
            yield n, p

    def local_parameters(self, recurse: bool = True):
        for n, p in self.local_named_parameters(recurse=recurse):
            yield p

    def local_named_parameters(self, recurse: bool = True):
        self._validate_partitioned()
        for n, p in self._local_parameters.items():
            yield n, p

    def local_named_buffers(self, prefix: str = "", recurse: bool = True):
        self._validate_partitioned()
        for name, buf in self.named_buffers(prefix=prefix, recurse=recurse):
            if name in self._local_buffers:
                yield name, buf

    def local_buffers(self, recurse=True):
        for n, b in self.local_named_buffers(recurse=recurse):
            yield b

    def is_scaled_batch_buffer(self, buf):
        return buf in self._scaled_batch_reducer_buffer_set

    def is_distributed_buffer(self, buf):
        return buf in state.module_manager._distributed_buffers

    def is_scaled_batch_parameter(self, param):
        return param in self._scaled_batch_reducer_param_set

    def is_distributed_parameter(self, param):
        return param in state.module_manager._distributed_parameters

    def scaled_batch_reducer_named_parameters(self):
        for n, p in self._scaled_batch_reducer_parameters.items():
            yield n, p

    def default_reducer_named_parameters(self):
        for n, p in self._default_reducer_parameters.items():
            yield n, p

    def distributed_modules(self):
        for mod in self.modules():
            if state.module_manager.is_distributed(mod):
                yield mod

    def _local_state_dict_nobool(
        self,
        cast_to_cpu=True,
        destination=None,
        prefix="",
        keep_vars=False,
        scaled_batch_only=False,
    ):
        d = self.local_state_dict(
            cast_to_cpu=cast_to_cpu,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
            scaled_batch_only=scaled_batch_only,
        )
        del d["_smp_is_partial"]
        if "_smp_load_info" in d:
            del d["_smp_load_info"]
        return d

    def local_state_dict(
        self,
        cast_to_cpu=True,
        destination=None,
        prefix="",
        keep_vars=False,
        scaled_batch_only=False,
    ):
        """
        Get the local state_dict which only contains parameters belong to this device
        Non-parameter values will be saved on all ranks.
        """
        full_state_dict = self.module.state_dict(
            destination=destination, prefix=prefix, keep_vars=True
        )
        # Gather all local parameter names in case there are params that are shared across modules
        local_params_buff = set(itertools.chain(self.local_parameters(), self.local_buffers()))

        local_names = set()
        remote_names = set()
        for name, value in full_state_dict.items():
            if value in local_params_buff:
                local_names.add(name)
            else:
                remote_names.add(name)

        local_dict = OrderedDict()
        for name, value in full_state_dict.items():
            if isinstance(value, torch.Tensor):
                if name in local_names:
                    if (
                        not scaled_batch_only
                        or self.is_scaled_batch_parameter(value)
                        or self.is_scaled_batch_buffer(value)
                    ):
                        local_dict[name] = value if keep_vars else value.detach()
            else:
                local_dict[name] = value

        if cast_to_cpu:
            for name, param in local_dict.items():
                if isinstance(param, torch.Tensor):
                    local_dict[name] = param.cpu()

        # Save smp related info
        local_dict["_smp_is_partial"] = True
        local_dict["_smp_load_info"] = {}
        local_dict["_smp_load_info"]["tensor_parallel_degree"] = state.cfg.tensor_parallel_degree
        local_dict["_smp_load_info"][
            "pipeline_parallel_degree"
        ] = state.cfg.pipeline_parallel_degree
        local_dict["_smp_load_info"]["remote_names"] = remote_names
        return local_dict

    def cpu(self):
        if self.partitioned:
            full_state = self.state_dict(keep_vars=True, cast_to_cpu=True)
            self.load_state_dict(full_state)
        super(DistributedModel, self).cpu()

    def cuda(self, *ignore):
        if not self.partitioned:
            logger.warning("Model has not been partitioned yet, ignoring model.cuda() call")
            return self
        else:
            # to is the same as cuda, but cuda is older
            # https://discuss.pytorch.org/t/are-there-some-differences-between-cuda-and-to-torch-device-cuda/22383
            return self.to(device=torch.device("cuda", local_rank()))

    def get_param_name(self, param):
        return self._param_names[param]

    def register_post_partition_hook(self, hook) -> RemovableHandle:
        return state.module_manager.register_post_partition_hook(hook)

    def register_post_step_hook(self, hook) -> RemovableHandle:
        return state.module_manager.register_post_step_hook(hook)

    # ------------------------------------------
    # PUBLIC DDP specific methods
    # -------------
    @contextmanager
    def join(self, divide_by_initial_world_size=True, enable=True):
        if state.cfg.ddp:
            with self.module.join(
                enable=enable, divide_by_initial_world_size=divide_by_initial_world_size
            ):
                yield
            self.module._reset_join_state()
        else:
            raise NotImplementedError(
                "join is only supported when using DDP. Please set ddp=True in SMP config"
            )

    def _register_builtin_comm_hook(self, comm_hook_type):
        if core.cfg.ddp and self.overlapping_allreduce:
            self.module._register_builtin_comm_hook(comm_hook_type)
        elif self._has_reducer_parameters:
            raise_ddp_overlapping_exception("_register_builtin_comm_hook")

    # Technically this method has _ prefix in PT 1.7 but no prefix in PT 1.8
    # For simplicity dropping the prefix entirely. We can document this only for 1.8
    def register_comm_hook(self, state: object, hook: callable):
        if core.cfg.ddp and self.overlapping_allreduce:
            if core.cfg._fp32_grad_accumulation:
                raise ValueError(
                    "register_comm_hook is not supporteed with _fp32_grad_accumulation=True"
                )
            self.module.register_comm_hook(state, hook)

            if rank() == 0 and self.average_grads_across_microbatches:
                logger.info(
                    "Registered comm hook for gradients. "
                    "Please note that when you register a comm hook you have full control of how the gradients are processed. "
                    "When using only data parallelism with Torch DDP you are expected to average grads across data parallel replicas within the hook. "
                    "When using SMP with DistributedModel, apart from averaging grads across data parallel replicas you have to also average grads across microbatches within the hook."
                )
        elif self._has_reducer_parameters:
            raise_ddp_overlapping_exception("register_comm_hook")

    def get_ddp_logging_data(self):
        if core.cfg.ddp and self.overlapping_allreduce:
            return self.module.get_ddp_logging_data()
        elif self._has_reducer_parameters:
            raise_ddp_overlapping_exception("get_ddp_logging_data")

    # ------------------------------------------
