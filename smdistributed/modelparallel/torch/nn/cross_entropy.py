# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: Taken from NVIDIA Megatron-LM repo with minor changes to adapt
# to smp terminology and structures

# Third Party
import torch
import torch.nn as nn

# First Party
from smdistributed.modelparallel.torch.core import tp_size
from smdistributed.modelparallel.torch.nn.utils import fwd_allreduce_for_tp


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, vocab_range):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        logits_max = fwd_allreduce_for_tp(logits_max, op=torch.distributed.ReduceOp.MAX)

        # Subtract the maximum value.
        ## vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))
        vocab_parallel_logits = vocab_parallel_logits.sub(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indices
        vocab_start_index, vocab_end_index = vocab_range
        partition_vocab_size = vocab_end_index - vocab_start_index

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        predicted_logits = fwd_allreduce_for_tp(predicted_logits)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        sum_exp_logits = fwd_allreduce_for_tp(sum_exp_logits)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        ## exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        exp_logits = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        ## grad_input.mul_(grad_output.unsqueeze(dim=-1) * tp_size())
        grad_input = grad_input.mul(grad_output.unsqueeze(dim=-1) * tp_size())

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, vocab_range):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, vocab_range)


class DistributedCrossEntropy(nn.Module):
    def __init__(self, vocab_range):
        super(DistributedCrossEntropy, self).__init__()
        self.vocab_range = vocab_range

    def forward(self, logits, targets):
        return vocab_parallel_cross_entropy(logits, targets, self.vocab_range)
