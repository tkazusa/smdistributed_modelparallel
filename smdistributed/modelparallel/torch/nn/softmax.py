# Third Party
import torch
import torch.nn as nn

# First Party
from smdistributed.modelparallel.torch.smp_torch_cuda_lib import (
    scaled_masked_softmax_backward,
    scaled_masked_softmax_forward,
    scaled_upper_triang_softmax_backward,
    scaled_upper_triang_softmax_forward,
)


class ScaledMaskedSoftmax(nn.Module):
    """ Fused kernel for scaling, masking, and softmax for self-attention blocks."""

    def __init__(self, scale):
        super(ScaledMaskedSoftmax, self).__init__()
        self.scale = scale

    def forward(self, inp, mask):
        if inp.dtype not in [torch.float16, torch.bfloat16]:
            raise TypeError(
                "Fused softmax kernel can only be used with float16 or bfloat16 dtypes."
            )

        return FusedScaledMaskedSoftmax.apply(inp, mask, self.scale)


class ScaledCausalMaskedSoftmax(nn.Module):
    """ Fused kernel for scaling, causal masking, and softmax for self-attention blocks."""

    def __init__(self, scale):
        super(ScaledCausalMaskedSoftmax, self).__init__()
        self.scale = scale

    def forward(self, inp):
        if inp.dtype not in [torch.float16, torch.bfloat16]:
            raise TypeError(
                "Fused softmax kernel can only be used with float16 or bfloat16 dtypes."
            )

        b, np, sq, sk = inp.size()
        inp = inp.view(-1, sq, sk)
        probs = FusedScaledUpperTriangMaskedSoftmax.apply(inp, self.scale)
        return probs.view(b, np, sq, sk)


class FusedScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the input attention mask
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_forward(inputs, mask, scale_t[0])

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_backward(output_grads, softmax_results, scale_t[0])

        return input_grads, None, None


class FusedScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_softmax_forward(inputs, scale_t[0])

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None
