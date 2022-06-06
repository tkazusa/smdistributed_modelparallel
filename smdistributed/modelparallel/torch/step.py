# Standard Library
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from functools import lru_cache as memoized
from typing import Dict, List, Set

# Third Party
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.backend.split import StepOutput as BaseStepOutput
from smdistributed.modelparallel.backend.split import TensorSplitter
from smdistributed.modelparallel.backend.utils import (
    get_divisibility_error_str,
    upload_metrics_to_studio,
)
from smdistributed.modelparallel.torch.comm import pp_barrier
from smdistributed.modelparallel.torch.core import dp_rank, pp_rank, rank, tp_rank
from smdistributed.modelparallel.torch.module_manager import TensorModuleInfo
from smdistributed.modelparallel.torch.state_mod import state
from smdistributed.modelparallel.torch.utils import check_env_var_truthy, map_structure, rmsg

logger = get_logger()


class StepOutput(BaseStepOutput):
    def reduce_mean(self):
        return self.process_outputs(lambda x: sum(x) / len(x))

    def reduce_sum(self):
        return self.process_outputs(sum)

    def concat(self):
        return self.process_outputs(lambda x: torch.cat(x, dim=0))

    def stack(self):
        return self.process_outputs(torch.stack)


class PTTensorSplitter(TensorSplitter):
    def get_tensor_type(self):
        return torch.Tensor

    def map_structure(self, func, structure):
        return map_structure(func, structure)

    def slice(self, tensor, num_mb, mb, axis=0):
        dim_size = list(tensor.size())[axis]
        if dim_size % num_mb != 0:
            raise ValueError(get_divisibility_error_str("pytorch", dim_size, num_mb))

        split_size = dim_size // num_mb
        return tensor.narrow(axis, mb * split_size, split_size)


class StepMemoryMetricsCollector:
    """Class that collects and saves torch and d2d buffer metrics to file."""

    def __init__(self, function_name, step, timestamp=None):
        self._function_name = function_name
        self._step = step
        if timestamp is None:
            timestamp = datetime.now()
        self._timestamp = timestamp.strftime("%d_%m_%Y_%H_%M_%S")

    @property
    @memoized(maxsize=1)
    def _should_collect_memory_metrics(self):
        return check_env_var_truthy("SMP_WRITE_STEP_MEMORY_METRICS", "0")

    def maybe_collect_memory_metrics(self, minibatch, open_func=open):
        """Collect pytorch allocator metrics and D2D metrics (size, peak utilization) and save to file"""
        if not self._should_collect_memory_metrics:
            return

        import torch.cuda

        with open_func(
            f"{self._function_name}_memory_metrics_dp{dp_rank()}_pp{pp_rank()}_tp{tp_rank()}_{self._timestamp}",
            "a",
        ) as metrics_file:
            torch_peak_reserved = torch.cuda.max_memory_reserved()
            torch_peak_allocated = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            memory_stats = state.core.get_and_reset_memory_stats()
            smp_backend_d2d_peak_allocated_mb = memory_stats.backend_d2d_peak_allocated_mb
            smp_backend_d2d_peak_reserved_mb = memory_stats.backend_d2d_peak_reserved_mb
            gpu_free_memory_mb = memory_stats.gpu_free_memory_mb
            gpu_total_memory_mb = memory_stats.gpu_total_memory_mb

            output_metrics = (
                f"{self._function_name}:{self._step} "
                f"minibatch {minibatch} "
                f"torch_peak_allocated {torch_peak_allocated} "
                f"torch_peak_reserved {torch_peak_reserved} "
                f"gpu_free_memory_mb {gpu_free_memory_mb} "
                f"gpu_total_memory_mb {gpu_total_memory_mb} "
                f"smp_backend_d2d_peak_allocated_mb {smp_backend_d2d_peak_allocated_mb} "
                f"smp_backend_d2d_peak_reserved_mb {smp_backend_d2d_peak_reserved_mb} "
            )
            print(output_metrics, file=metrics_file)


class StepFunction:
    _id = 0

    def __init__(self, func, non_split_inputs, input_split_axes, detach_outputs):
        self.id = StepFunction._id
        self.func = func
        self.non_split_inputs = non_split_inputs
        self.input_split_axes = input_split_axes
        self.detach_outputs = detach_outputs

        types_to_warn = [optim.Optimizer, _LRScheduler, Dataset, DataLoader, Sampler]
        types_to_suppress_warn = [nn.Module, optim.Optimizer, GradScaler, Namespace, torch.device]
        self._splitter = PTTensorSplitter(
            self.func,
            self.non_split_inputs,
            self.input_split_axes,
            types_to_warn,
            types_to_suppress_warn,
        )
        self.step_memory_metrics = StepMemoryMetricsCollector(self.func.__name__, self.id)

        state.step_func[self.id] = self
        StepFunction._id += 1
        self.minibatch = 0

        # number of times the module was executed in this step so far
        self._fwd_module_execution_counts: Dict[int, Dict[nn.Module, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        self._bwd_module_execution_counts: Dict[int, Dict[nn.Module, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )

        # map from TensorModuleInfo to consumer (module_name, count) tuples
        self._direct_consumers: Dict[TensorModuleInfo, List] = defaultdict(list)

        # tensors previously excluded from being added into direct consumers. this is
        # used to prevent cases where the generator of an input tensor is the parent already
        # so a direct child-to-child transmission is not needed.
        self._excluded_tensors: Set[TensorModuleInfo] = set()

    def update_direct_fwd_consumers(self, args, module, exclude_partitions, start_index=None):
        """ Record direct forward consumers of modules. The map is keyed by
        TensorModuleInfo, and maps into (module_name, count) pairs, where
        count is how many times this module was executed before. """

        assert state.microbatch == 0

        module_name = state.module_manager.get_module_name(module)
        if start_index:
            # for sequential
            module_name = module_name + "/" + str(start_index)

        for arg in args:
            if isinstance(arg, torch.Tensor) and hasattr(arg, "_smp_module_info"):
                tensor_generator_module = state.module_manager.get_module(
                    arg._smp_module_info.module_name
                )
                if (
                    state.module_manager.get_partition(tensor_generator_module)
                    in exclude_partitions
                    or arg._smp_module_info in self._excluded_tensors
                ):
                    self._excluded_tensors.add(arg._smp_module_info)
                    continue

                count = self.get_fwd_module_execution_count(module, 0)
                if (module_name, count) not in self._direct_consumers[arg._smp_module_info]:
                    if state.current_minibatch() == 0:
                        self._direct_consumers[arg._smp_module_info].append((module_name, count))
                    elif state.cfg.fast_mode:
                        raise RuntimeError(
                            "A change in model graph is detected. Graph changes are not supported in fast mode, please set 'fast_mode' to False."
                        )

    def update_direct_bwd_consumers(self, grads, module):
        assert state.microbatch == 0

        for grad in grads:
            if grad is not None:
                assert isinstance(grad, torch.Tensor)
                if hasattr(grad, "_smp_module_info"):
                    module_name = state.module_manager.get_module_name(module)
                    count = self.get_bwd_module_execution_count(module, 0)
                    self._direct_consumers[grad._smp_module_info].append((module_name, count))

    def merge_direct_consumer_maps(self):
        """ Allgather consumer maps across pp_ranks """
        from smdistributed.modelparallel.backend.collectives import CommGroup

        all_maps = state.comm.allgather(self._direct_consumers, CommGroup.PP_GROUP)

        merged_consumers = defaultdict(list)
        for rank_consumers in all_maps:
            for k, v in rank_consumers.items():
                merged_consumers[k].extend(v)
        self._direct_consumers = merged_consumers

    def has_direct_consumers(self, module_info):
        return module_info in self._direct_consumers

    def get_direct_consumers(self, module_info):
        if self.has_direct_consumers(module_info):
            return self._direct_consumers[module_info]
        else:
            return []

    def get_direct_consumer_partitions(self, module_info):
        consumer_infos = self.get_direct_consumers(module_info)
        dests = []
        for info in consumer_infos:
            mod = state.module_manager.get_module(info[0])
            dests.append(state.module_manager.get_partition(mod))
        return dests

    def increment_fwd_module_execution_count(self, module, microbatch):
        self._fwd_module_execution_counts[microbatch][module] += 1

    def increment_bwd_module_execution_count(self, module, microbatch):
        self._bwd_module_execution_counts[microbatch][module] += 1

    def get_fwd_module_execution_count(self, module, microbatch):
        return self._fwd_module_execution_counts[microbatch][module]

    def get_bwd_module_execution_count(self, module, microbatch):
        return self._bwd_module_execution_counts[microbatch][module]

    def __call__(self, *args, **kwargs):
        state.current_step_func_id = self.id
        state.core.timeline_start_step()
        with state.model._step():
            if pp_rank() == 0:
                mb_args, mb_kwargs = self._splitter.preprocess_args_all_mbs(
                    args, kwargs, state.num_microbatches()
                )
                state.pipeline.init_step()
                state.exec_server.run_step_leader(mb_args, mb_kwargs, self.id)
            else:
                state.exec_server.run_step_follower()

        # ensures all ranks finish a step together so they dont interfere with each other
        pp_barrier()

        state.core.timeline_end_step()
        self._upload_metrics_once()
        self.step_memory_metrics.maybe_collect_memory_metrics(self.minibatch)

        if self.minibatch == 0:
            self.merge_direct_consumer_maps()
            logger.debug(rmsg(f"Step func: {self.id}, Direct consumers: {self._direct_consumers}"))

        state.clear_minibatch_state()
        self.clear_minibatch_state()
        logger.debug(rmsg(f"Step func {self.id} finished minibatch {self.minibatch}"))
        self.minibatch += 1
        outputs = state.exec_server.outputs
        state.current_step_func_id = None
        state.exec_server.outputs = None

        output_list = [None for _ in outputs]
        for mb, item in outputs.items():
            output_list[mb] = item

        return self.as_step_output(output_list)

    def clear_minibatch_state(self):
        self._fwd_module_execution_counts.clear()
        self._bwd_module_execution_counts.clear()

    def _upload_metrics_once(self):
        """
        Upload the metrics to Sagemaker studio
        """
        if not state.has_uploaded_metrics:
            num_hops = state.comm.allgather(state.num_hops, group=CommGroup.PP_GROUP)
            num_hops = sum(num_hops)
            if rank() == 0:
                var_size, module_fraction, comm_vol = state.module_manager.get_metrics()
                metrics = {}
                metrics["total_communication_volume(MB)"] = round(comm_vol, 2)
                metrics["num_hops_between_devices"] = num_hops
                for i in range(state.cfg.pipeline_parallel_degree):
                    metrics[f"parameter_count_on_dev_{i}"] = var_size[i]
                    metrics[f"module_fraction_on_dev_{i}"] = module_fraction[i]
                upload_metrics_to_studio(metrics)

            state.has_uploaded_metrics = True

    def as_step_output(self, outputs):
        num_mb = len(outputs)
        if num_mb == 0:
            return

        if isinstance(outputs[0], list):
            return [
                self.as_step_output([outputs[mb][i] for mb in range(num_mb)])
                for i in range(len(outputs[0]))
            ]
        elif isinstance(outputs[0], tuple):
            return tuple(
                self.as_step_output([outputs[mb][i] for mb in range(num_mb)])
                for i in range(len(outputs[0]))
            )
        elif isinstance(outputs[0], dict):
            return {
                k: self.as_step_output([outputs[mb][k] for mb in range(num_mb)])
                for k, v in outputs[0].items()
            }
        elif isinstance(outputs[0], torch.Tensor):
            # outputs are detached in server if necessary
            return StepOutput(outputs)
        else:
            return outputs


def step(func=None, non_split_inputs=None, input_split_axes=None, detach_outputs=True):
    """
    Wrapper for the main step function which executes forward and backward pass in pytorch
    """

    def decorated(inner_func):
        return StepFunction(
            inner_func,
            non_split_inputs=non_split_inputs,
            input_split_axes=input_split_axes,
            detach_outputs=detach_outputs,
        )

    if func is None:
        return decorated
    else:
        return decorated(func)
