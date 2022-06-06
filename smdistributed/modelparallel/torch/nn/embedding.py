# Third Party
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd

# First Party
from smdistributed.modelparallel.torch.core import core
from smdistributed.modelparallel.torch.nn.dist_module import DistributedModule
from smdistributed.modelparallel.torch.nn.utils import (
    allgather_for_tp,
    fused_allgather_for_tp,
    get_local_channels,
    get_merge_shapes,
    initialize_with_output_partition,
    parameter_creation_scope,
    scatter_and_merge_for_tp,
)


class DistributedEmbedding(nn.Embedding, DistributedModule):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        initializer_range=0.02,
        _skip_allgather=False,
        _skip_scatter_and_merge=False,
    ):
        super(nn.Embedding, self).__init__()

        self.num_embeddings = num_embeddings
        self._skip_allgather = _skip_allgather
        self._skip_scatter_and_merge = _skip_scatter_and_merge

        self.embedding_dim = get_local_channels(embedding_dim)
        self.original_embedding_dim = embedding_dim

        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.sparse = sparse
        if self.max_norm is not None:
            raise ValueError(
                "Passing max_norm is not supported for DistributedEmbedding module. Please set max_norm=None when using DistributedEmbedding"
            )

        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.fp16_params = core.cfg.fp16_params
        dtype = torch.float16 if self.fp16_params else None

        if _weight is not None:
            raise ValueError("Passing not None _weight is not supported with DistributedEmbedding")

        with parameter_creation_scope(
            self,
            scaled_batch=True,
            dtype=dtype,
            use_normal=True,
            initializer_range=initializer_range,
        ):
            with initialize_with_output_partition(self, axis=1):
                self.weight = nn.Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
            self.reset_parameters()

    @staticmethod
    def can_distribute(*args, **kwargs):
        max_norm = None
        weight = None
        if "max_norm" in kwargs:
            max_norm = kwargs["max_norm"]

        if "_weight" in kwargs:
            weight = kwargs["_weight"]

        if "embedding_dim" in kwargs:
            embedding_dim = kwargs["embedding_dim"]

        if not kwargs:
            if len(args) > 1:
                embedding_dim = args[1]
            if len(args) > 3:
                max_norm = args[3]
            if len(args) > 7:
                weight = args[7]

        if max_norm:
            logger.info(
                f"Skipping distribution of nn.Embedding because distribution is not supported with max_norm not None"
            )
            return False
        elif weight:
            logger.info(
                f"Skipping distribution of nn.Embedding because distribution is not supported when weight is provided"
            )
            return False

        return True

    def forward(self, inp):
        batch_dim = 0
        embedding_dim = -1
        full_inputs = allgather_for_tp(inp, batch_dim) if not self._skip_allgather else inp
        output_embedding = F.embedding(
            full_inputs,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # Without this, the merge shapes arent correct for embedding_dim on some ranks
        merge_shapes = get_merge_shapes(self.original_embedding_dim)
        result = (
            scatter_and_merge_for_tp(
                output_embedding,
                split_dim=batch_dim,
                merge_dim=embedding_dim,
                merge_shapes=merge_shapes,
            )
            if not self._skip_scatter_and_merge
            else output_embedding
        )
        return result


# Currently unused
# To use this, in the __init__ divide num_embeddings by tp_rank
# and create weight
class DistVocabSplitFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        inp,
        weight,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
        slice_size,
        vocab_start_idx,
        vocab_end_idx,
    ):
        from smdistributed.modelparallel.torch.state_mod import state
        from smdistributed.modelparallel.torch.comm import get_tp_process_group

        batch_dim = 0
        embedding_dim = -1
        local_bs = inp.size(batch_dim)
        weight_requires_grad = weight.requires_grad
        ctx.slice_size = slice_size
        ctx.vocab_start_idx = vocab_start_idx
        ctx.vocab_end_idx = vocab_end_idx

        if not state.is_tracing():
            inp = inp.contiguous()
            # Allgather inputs from all TP ranks
            # After this step, all tp_ranks in the same tp_group should have the same inputs
            full_inputs = allgather_for_tp(inp, batch_dim)

            # Uncommenting and running with renorm on the dist module test crashes
            # Revisit this if max_norm needs to be supported later
            # if max_norm is not None:
            #    F._no_grad_embedding_renorm_(weight, full_inputs, max_norm, norm_type)

            with torch.enable_grad():
                # Build the mask, result is a boolean tensor with False for inputs outside index range
                # for tp_rank and True otherwise
                input_mask = (full_inputs < ctx.vocab_start_idx) | (
                    full_inputs >= ctx.vocab_end_idx
                )

                # Mask the input
                masked_input = full_inputs.clone() - ctx.vocab_start_idx
                masked_input[input_mask] = 0

                # Execute the partial embedding lookup on the full set of inputs
                detached_weight = weight.detach()
                detached_weight.requires_grad_(weight_requires_grad)
                output_embedding = F.embedding(
                    masked_input,
                    detached_weight,
                    padding_idx,
                    max_norm,
                    norm_type,
                    scale_grad_by_freq,
                    sparse,
                )
                output_embedding[input_mask, :] = 0.0
            ctx.saved_weight = detached_weight
            ctx.save_for_backward(output_embedding)
            result = output_embedding.clone()

            # All reduce the output_embedding
            dist.all_reduce(result, group=get_tp_process_group())
            result.detach_()
            return result
        else:
            # emulate the behavior on different tp ranks for tracing, but without collectives
            # same tensor will be concatenated batch_dim / inp.size(0) times to generate full inputs
            # this will be only run for forward
            full_inputs = scale_batch_for_tracing(inp, batch_dim)
            output = F.embedding(
                full_inputs, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
            )
            return narrow_batch_for_tracing(output, batch_dim)

    @staticmethod
    @custom_fwd
    def backward(ctx, grad_out):
        batch_dim = 0
        embedding_dim = -1
        output_embedding, = ctx.saved_tensors
        weight = ctx.saved_weight
        slice_size = ctx.slice_size
        with torch.enable_grad():
            output_embedding.backward(grad_out)
        return None, weight.grad, None, None, None, None, None, None, None, None
