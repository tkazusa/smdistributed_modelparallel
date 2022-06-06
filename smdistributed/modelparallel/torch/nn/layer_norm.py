# Third Party
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import init

# First Party
from smdistributed.modelparallel.torch.core import local_rank, tp_size
from smdistributed.modelparallel.torch.nn.utils import get_local_channels
from smdistributed.modelparallel.torch.smp_torch_cuda_lib import (
    backward_affine_finish,
    backward_affine_local_sums,
    forward_affine_mean_var,
)

apex_is_available = True
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as ApexFusedLayerNorm
    from apex.normalization.fused_layer_norm import MixedFusedLayerNorm as ApexMixedFusedLayerNorm
except ImportError:
    apex_is_available = False


class DistributedLayerNormForTP(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, normalized_shape, orig_last_dim, eps):
        from smdistributed.modelparallel.torch.comm import get_tp_process_group
        from smdistributed.modelparallel.torch import is_tracing

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.dtype = input.dtype
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()

        mean = torch.sum(input_, dim=-1) / float(orig_last_dim)

        if not is_tracing():
            torch.distributed.all_reduce(mean, group=get_tp_process_group())

        mean = mean.unsqueeze(-1)
        shifted = input_ - mean

        var = torch.sum(shifted * shifted, dim=-1).unsqueeze(-1) / float(orig_last_dim)

        if not is_tracing():
            torch.distributed.all_reduce(var, group=get_tp_process_group())

        # TODO add fp16 support - not working for some reason
        input32 = input_.to(torch.float32)
        weight32 = weight_.to(torch.float32)
        bias32 = bias_.to(torch.float32)
        mean32 = mean.to(torch.float32)
        var32 = var.to(torch.float32)

        output, invvar = forward_affine_mean_var(
            input32, mean32, var32, ctx.normalized_shape, weight32, bias32, ctx.eps
        )
        ctx.save_for_backward(input32, weight32, bias32, mean32, invvar)
        return output.to(ctx.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        from smdistributed.modelparallel.torch.comm import get_tp_process_group
        from smdistributed.modelparallel.torch import is_tracing

        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        cast_grad_output = grad_output.contiguous().to(torch.float32)
        _, grad_weight, grad_bias, local_sum1, local_sum2 = backward_affine_local_sums(
            cast_grad_output, mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )

        if not is_tracing():
            torch.distributed.all_reduce(local_sum1, group=get_tp_process_group())
            torch.distributed.all_reduce(local_sum2, group=get_tp_process_group())

        grad_input, _, _ = backward_affine_finish(
            cast_grad_output,
            mean,
            invvar,
            local_sum1,
            local_sum2,
            input_,
            ctx.normalized_shape,
            weight_,
            bias_,
            ctx.eps,
            tp_size(),
        )

        return (
            grad_input.to(ctx.dtype),
            grad_weight.to(ctx.dtype),
            grad_bias.to(ctx.dtype),
            None,
            None,
            None,
        )


class DistributedLayerNorm(nn.Module):
    """ Distributed tensor-parallel implementation of layer normalization. Not intended for use by itself - should only be
        used inside other distributed modules, as it does not use parameter_creation_scope contexts. """

    def __init__(self, normalized_shape, eps=1e-5):
        super(DistributedLayerNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        local_normalized_shape = list(normalized_shape)
        local_normalized_shape[-1] = get_local_channels(local_normalized_shape[-1])

        self.normalized_shape = torch.Size(local_normalized_shape)
        self.orig_last_dim = list(normalized_shape)[-1]
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(*local_normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(*local_normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        if not input.is_cuda:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

        return DistributedLayerNormForTP.apply(
            input, self.weight, self.bias, self.normalized_shape, self.orig_last_dim, self.eps
        )


if apex_is_available:

    class FusedLayerNorm(ApexFusedLayerNorm):
        def forward(self, x):
            torch.cuda.set_device(local_rank())
            return super(FusedLayerNorm, self).forward(x)

    # From apex: Why "mixed"?
    # MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
    # as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
    # See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"
    class MixedFusedLayerNorm(ApexMixedFusedLayerNorm):
        def forward(self, x):
            torch.cuda.set_device(local_rank())
            return super(MixedFusedLayerNorm, self).forward(x)
