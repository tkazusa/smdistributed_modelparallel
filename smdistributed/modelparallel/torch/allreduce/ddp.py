# Third Party
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.allreduce.reducer import GradReducer
from smdistributed.modelparallel.torch.exceptions import DDPConfigError

logger = get_logger()


class DdpNonOverlappingAllreducer(GradReducer):
    """
    This is used when overlapping_allreduce is False with DDP.
    This is similar to Megatron's Local DDP.
    It buckets grads together and performs allreduce on the bucket at once.
    """

    def __init__(
        self,
        named_parameters,
        grad_counter,
        average_grads_across_microbatches,
        num_microbatches,
        scaled_batch,
        tp_size,
        process_group,
        ddp_config=None,
    ):
        super(DdpNonOverlappingAllreducer, self).__init__(
            named_parameters,
            grad_counter,
            overlapping_allreduce=False,
            average_grads_across_microbatches=average_grads_across_microbatches,
            num_microbatches=num_microbatches,
            scaled_batch=scaled_batch,
            tp_size=tp_size,
        )
        if ddp_config is None:
            ddp_config = {}
        self._validate_config(ddp_config)

        # whether to do the scaling of dp_size before or after allreduce
        # just kept this param same as what Megatron does
        self.reduce_after = ddp_config.get("reduce_after", True)

        # casts to fp32 for allreduce, also an option consistent with Megatron
        self.fp32_allreduce = ddp_config.get("fp32_allreduce", False)
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False
        self.process_group = process_group
        self.fwd_pass_work_handle = None
        self.use_static_world_size = True

    def _validate_config(self, ddp_config):
        supported_configs = ["reduce_after", "fp32_allreduce"]
        if any([k not in supported_configs for k in ddp_config.keys()]):
            raise DDPConfigError(
                f"Only the following ddp_config keys are supported {supported_configs}"
            )
        if "reduce_after" in ddp_config and ddp_config["reduce_after"] not in [False, True]:
            raise DDPConfigError("reduce_after in ddp_config can only be True or False")
        if "fp32_allreduce" in ddp_config and ddp_config["fp32_allreduce"] not in [False, True]:
            raise DDPConfigError("fp32_allreduce in ddp_config can only be True or False")

    def _set_forward_pass_work_handle(self, work, use_static_world_size):
        self.fwd_pass_work_handle = work
        self.use_static_world_size = use_static_world_size

    def _rebuild_buckets(self):
        # method added to maintain compatibility between this class and the CPP DDP reducer
        return False

    def _push_all_rebuilt_params(self):
        return

    def _match_all_reduce_for_bwd_pass(self):
        self._synchronize_internal(set_grad_zero=True)

    def _synchronize_internal(self, set_grad_zero=False):
        """
        Bucketing logic is similar to megatron https://github.com/jarednielsen/t5/blob/master/megatron/model/distributed.py#L35
        """
        buckets = {}
        for name, param in self.named_parameters.items():
            if param.requires_grad and param.grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                # scale grad to account for different microbatches
                # do it here just before allreduce
                if self.average_grads_across_microbatches:
                    if self.scaled_batch:
                        param.grad /= float(self.num_microbatches * self.tp_size)
                    else:
                        param.grad /= float(self.num_microbatches)

        if self.warn_on_half:
            if torch.cuda.HalfTensor in buckets:
                logger.warning(
                    "WARNING: gloo dist backend for half parameters may be extremely slow."
                    + " It is recommended to use the NCCL backend in this case."
                )
                self.warn_on_half = False

        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = _flatten_dense_tensors(grads)
            if set_grad_zero:
                coalesced = torch.zeros_like(coalesced)
            if self.fp32_allreduce:
                coalesced = coalesced.float()

            if self.fwd_pass_work_handle is not None and self.use_static_world_size is False:
                self.fwd_pass_work_handle.wait()
                result_tensor = self.fwd_pass_work_handle.result()[0]
                div_factor = result_tensor.item()
                self.fwd_pass_work_handle = None
            else:
                div_factor = dist.get_world_size(group=self.process_group)

            if not self.reduce_after:
                # reduce before
                coalesced /= div_factor
            dist.all_reduce(coalesced, group=self.process_group)
            torch.cuda.synchronize()

            if not set_grad_zero:
                # if this rank actually participated in the allreduce
                if self.reduce_after:
                    coalesced /= div_factor
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    def _hook_internal(self, name, param, grads):
        pass
