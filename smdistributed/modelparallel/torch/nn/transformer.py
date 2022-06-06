# Standard Library
import math
import os
from collections import OrderedDict
from functools import lru_cache

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import (
    core,
    local_rank,
    pp_rank,
    rdp_rank,
    tp_rank,
    tp_size,
)
from smdistributed.modelparallel.torch.nn.dist_module import DistributedModule
from smdistributed.modelparallel.torch.nn.embedding import DistributedEmbedding
from smdistributed.modelparallel.torch.nn.gelu import bias_gelu_impl
from smdistributed.modelparallel.torch.nn.layer_norm import DistributedLayerNorm
from smdistributed.modelparallel.torch.nn.softmax import (
    ScaledCausalMaskedSoftmax,
    ScaledMaskedSoftmax,
)
from smdistributed.modelparallel.torch.nn.utils import (
    allgather_for_tp,
    batch_collective,
    bwd_allreduce_for_tp,
    fused_allgather_for_tp,
    fwd_allreduce_for_tp,
    get_local_channels,
    get_merge_shapes,
    initialize_with_input_partition,
    initialize_with_output_partition,
    narrow_batch_for_tracing,
    narrow_for_tp,
    parameter_creation_scope,
    parse_args,
    reduce_scatter_for_tp,
    scale_batch_for_tracing,
    scatter_and_merge_for_tp,
    shard_sequence,
    unshard_sequence,
    update_copy_dict,
)
from smdistributed.modelparallel.torch.smp_torch_cuda_lib import get_batch_per_block

try:
    from smdistributed.modelparallel.torch.nn.layer_norm import FusedLayerNorm
    from smdistributed.modelparallel.torch.nn.layer_norm import MixedFusedLayerNorm

    LayerNorm = FusedLayerNorm
    MixedLayerNorm = MixedFusedLayerNorm
except ImportError:
    LayerNorm = nn.LayerNorm
    MixedLayerNorm = None

logger = get_logger()


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


@lru_cache(maxsize=1024)
def can_use_fused_kernel(config_obj, input_tensor_shape):
    # enforce the requirements of the cuda kernel

    from smdistributed.modelparallel.torch.state_mod import state

    # the fused kernel requires fp16 inputs, and does not support cpu device
    if (
        not state.cfg.fp16_params
        or not config_obj.fused_softmax
        or state.is_tracing()
        and state.model.trace_device == "cpu"
    ):
        return False

    b, np, sq, sk = input_tensor_shape
    attn_batches = b * np

    # the fused cuda kernel has these conditions as requirements; otherwise we fall back to original implementation
    if 16 < sk <= 2048 and sq % 4 == 0 and attn_batches % 4 == 0:
        batch_per_block = get_batch_per_block(sq, sk, b, np)

        cross_attention_only = hasattr(config_obj, "cross_attention") and config_obj.cross_attention
        if config_obj.causal_mask_size is not None and not cross_attention_only:
            if attn_batches % batch_per_block == 0:
                return True
        else:
            if sq % batch_per_block == 0:
                return True

    return False


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(2, 3), sincos
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class DistributedTransformerLMHead(DistributedModule):
    """
    Distributed transformer implementation with generic hyperparameters and embeddings and LM head.
    """

    _KEYS = OrderedDict(
        [
            ("num_layers", 12),
            ("num_attention_heads", 32),
            ("attention_head_size", 32),
            ("hidden_size", 1024),
            ("intermediate_size", 4096),
            ("vocab_size", 30522),
            ("num_positions", 1024),
            ("attention_dropout_prob", 0.1),
            ("hidden_dropout_prob", 0.1),
            ("activation", "gelu"),
            ("layernorm_epsilon", 1e-5),
            ("num_token_types", 0),
            ("causal_mask_size", None),
            ("add_cross_attention", False),
            ("add_lm_head", True),
            ("initializer_range", 0.02),
            ("use_normal_initialization", False),
            ("pre_layernorm", False),
            ("post_layernorm", True),
            ("attention_in_fp32", False),
            ("query_key_layer_scaling", False),
            ("fp32_residual_addition", False),
            ("fused_softmax", True),
            ("fused_bias_gelu", False),
            ("distribute_embedding", False),
            ("_scale_qkv_fan_out", False),
            ("_precision_test", False),
            ("rotary_dim", None),
        ]
    )

    def __init__(self, *args, **kwargs):
        super(DistributedTransformerLMHead, self).__init__()

        config = self._parse_args(args, kwargs)
        self.fp16_params = core.cfg.fp16_params
        dtype = torch.float16 if self.fp16_params else None

        # Use distributed embeddings which will split the embedding layer across the embedding_dim
        if self.distribute_embedding:
            self.word_embedding = DistributedEmbedding(
                self.vocab_size,
                self.hidden_size,
                initializer_range=self.initializer_range,
                _skip_allgather=True,
                _skip_scatter_and_merge=True,
            )
            if self.rotary_dim is None:
                self.position_embedding = DistributedEmbedding(
                    self.num_positions,
                    self.hidden_size,
                    initializer_range=self.initializer_range,
                    _skip_allgather=True,
                    _skip_scatter_and_merge=True,
                )

            if self.num_token_types > 0:
                self.token_type_embedding = DistributedEmbedding(
                    self.num_token_types, self.hidden_size
                )
        else:
            with parameter_creation_scope(
                module=self, scaled_batch=False, dtype=dtype, use_normal=True
            ):
                self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
                if self.rotary_dim is None:
                    self.position_embedding = nn.Embedding(self.num_positions, self.hidden_size)

                if self.num_token_types > 0:
                    self.token_type_embedding = nn.Embedding(self.num_token_types, self.hidden_size)

        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Set full_hidden_states_sharded_hidden_dim to True, since we shard across hidden_dim when embedding is split
        # These are full_hidden_states for the entire batch across tp_ranks, since the batch is allgathered across tp_ranks
        # when prescaled_batch == False, and the batch is the same across tp_ranks when prescaled_batch == True
        if self.distribute_embedding:
            kwargs["full_hidden_states_sharded_hidden_dim"] = True
        self.transformer = DistributedTransformer(*args, **kwargs)
        self.ce_loss = nn.CrossEntropyLoss()
        if self.add_lm_head:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.lm_head.weight = self.word_embedding.weight

    def _parse_args(self, args, kwargs):
        self.batch_dim = 0
        config = parse_args(args, kwargs, DistributedTransformerLMHead._KEYS, self)

        if not config["pre_layernorm"] and not config["post_layernorm"]:
            raise ValueError("At least one of pre_layernorm and post_layernorm must be True.")

        if self.causal_mask_size is not None and self.num_positions > self.causal_mask_size:
            raise ValueError(
                f"If causal mask size ({self.causal_mask_size}) is provided, it must be larger than or equal to the number of positions ({self.num_positions})."
            )

        if self.fp32_residual_addition and MixedLayerNorm == None:
            raise ValueError("fp32 residual addition requires apex's MixedFusedLayerNorm.")

        if self.fp32_residual_addition and core.cfg.optimize == "memory":
            raise ValueError("fp32 residual addition only supports optimize == speed.")

    def forward(self, inputs):
        from smdistributed.modelparallel.torch.state_mod import state

        if self.add_cross_attention:
            input_ids, attention_mask, token_type_ids, position_ids, cross_states, cross_mask, labels = (
                inputs
            )
        else:
            input_ids, attention_mask, token_type_ids, position_ids, labels = inputs

        input_shape = input_ids.size()
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        local_attention_heads = get_local_channels(self.num_attention_heads)

        if state.is_tracing() and state.model.current_trace_device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", local_rank())

        # prepare position ids
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # prepare attention masks: normally, we expect the attention mask input to be prepared by the user - masked locations should be
        # some large negative value, and the rest should be zero. however, we add the mask preparing logic here for compatibility with
        # huggingface gpt implementation, which assumes the attention mask is 0-1, and the masked locations are 0. this discrepancy
        # is of little practical impact because typically gpt training (which this module is typically used for) does not actually use
        # additional attention masks other than the upper triangular/causal mask which is generated within the model itself.
        if self.fp16_params:
            # TODO revisit
            attention_mask = attention_mask.to(torch.float16)

        if self.causal_mask_size is None or not can_use_fused_kernel(
            self, (batch_size, local_attention_heads, seq_length, seq_length)
        ):
            # since the fused kernel for upper triangular masking ignores attention mask input, no need to prepare
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

        if self.add_cross_attention:
            _, cross_seq_length, _ = cross_states.size()
            cross_hidden_shape = (batch_size, cross_seq_length)
            if cross_mask is None:
                cross_mask = torch.ones(cross_hidden_shape, device=device)
            cross_mask = cross_mask[:, None, None, :]
            if self.fp16_params:
                # TODO revisit
                cross_mask = cross_mask.to(torch.float16)
                cross_mask = (1.0 - cross_mask) * -10000.0
            else:
                cross_mask = (1.0 - cross_mask) * -1e9
        else:
            cross_mask = None

        # embeddings
        if not state.cfg.prescaled_batch and self.distribute_embedding:
            # If prescaled_batch is not set, each tp_rank will have different data
            # allgather across tp_ranks so that each input has the full batch
            input_ids = allgather_for_tp(input_ids, self.batch_dim)
            if token_type_ids is not None:
                token_type_ids = allgather_for_tp(token_type_ids, self.batch_dim)
        elif state.cfg.prescaled_batch and not self.distribute_embedding:
            seq_length = input_ids.shape[1]
            input_ids, position_ids, token_type_ids = shard_sequence(
                input_ids, position_ids, token_type_ids, bwd_allgather=False
            )

        # If we are using DistributedEmbedding, hidden_states (full) will be sharded across hidden_dim
        if self.rotary_dim is not None:
            hidden_states = self.word_embedding(input_ids)
        else:
            hidden_states = self.word_embedding(input_ids) + self.position_embedding(position_ids)

        if token_type_ids is not None:
            if self.num_token_types > 0:
                hidden_states += self.token_type_embedding(token_type_ids)
            else:
                # HF GPT-2 uses word embedding as token type embedding too
                hidden_states += self.word_embedding(token_type_ids)

        # dropout
        with state.rng_manager.consistent_rng_state(enabled=state.cfg.prescaled_batch):
            hidden_states = self.dropout(hidden_states)

        # hidden_states is split across hidden_dim/embedding_dim. Thus merge across
        # embedding_dim to obtain full hidden_states
        # When prescaled_batch = False, skip allgather here since allgather happens
        # in DistributedAttentionLayer
        if state.cfg.prescaled_batch:
            if self.distribute_embedding:
                merge_shapes = get_merge_shapes(self.hidden_size)
                hidden_states = fused_allgather_for_tp(hidden_states, -1, merge_shapes=merge_shapes)
            else:
                hidden_states, = unshard_sequence(seq_length, hidden_states)

        # transformer
        if self.add_cross_attention:
            tx_inputs = hidden_states, attention_mask, cross_states, cross_mask
        else:
            tx_inputs = hidden_states, attention_mask
        hidden_states, *_ = self.transformer(tx_inputs)

        if self.distribute_embedding:
            hidden_states = narrow_for_tp(hidden_states, -1)
            lm_logits = self.lm_head(hidden_states)

        if state.cfg.prescaled_batch:
            if self.distribute_embedding:
                # Reduce across tp_ranks, and shard across sequence dim.
                # Equivalent to allreduce followed by narrow, but lesser communication
                # for forward backward combined.
                lm_logits = reduce_scatter_for_tp(lm_logits, 0, shift=-1, transpose=(0, 1))
            else:
                hidden_states, = shard_sequence(hidden_states, shift=-1)

            labels, = shard_sequence(labels)
            if tp_rank() == 0:
                # labels: [sharded_seq_length, batch_size]
                labels = labels[1:, :].contiguous()
        elif self.distribute_embedding:
            # Scatter across batch_dim
            lm_logits = reduce_scatter_for_tp(lm_logits, 0)

        if not self.distribute_embedding:
            lm_logits = self.lm_head(hidden_states)

        if labels is not None:
            if state.cfg.prescaled_batch:
                loss = self.ce_loss(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

                # when not using prescaled batch, loss is computed over batch_size * (seq_len - 1) tokens
                # when using prescaled batch, loss is computed over different number of tokens by each rank
                # we scale the loss up/down so that losses from different tp ranks have the same weight
                # as if prescaled ranks process same number of tokens as in non prescaled case.
                if tp_rank() == 0:
                    seq_length_sharded = labels.size(0) + 1
                    seq_length = seq_length_sharded * tp_size()
                    if seq_length <= tp_size():
                        # not a practical scenario
                        raise RuntimeError(
                            "Sequence length has to be larger than tp_size when using prescaled batch"
                        )
                    else:
                        # Loss was computed over tp_size * batch_size * (seqlen - tp_size)/tp_size
                        # below scales down loss to match num tokens used in non-prescaled case.
                        loss *= (seq_length - tp_size()) / (seq_length - 1)
                else:
                    seq_length_sharded = labels.size(0)
                    seq_length = seq_length_sharded * tp_size()
                    # Loss was computed over tp_size * batch_size * (seqlen/tp_size)
                    # below scales up loss to match num tokens used in non-prescaled case.
                    loss *= seq_length / (seq_length - 1)
            else:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.ce_loss(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
            return (loss, lm_logits)

        return lm_logits


class DistributedTransformer(DistributedModule):
    """
    Distributed transformer implementation with generic hyperparameters.
    """

    _KEYS = OrderedDict(
        [
            ("num_layers", 12),
            ("num_attention_heads", 32),
            ("attention_head_size", 32),
            ("hidden_size", 1024),
            ("intermediate_size", 4096),
            ("attention_dropout_prob", 0.1),
            ("hidden_dropout_prob", 0.1),
            ("activation", "gelu"),
            ("layernorm_epsilon", 1e-5),
            ("initializer_range", 0.02),
            ("use_normal_initialization", False),
            ("causal_mask_size", None),
            ("add_cross_attention", False),
            ("pre_layernorm", False),
            ("post_layernorm", True),
            ("attention_in_fp32", False),
            ("query_key_layer_scaling", False),
            ("fp32_residual_addition", False),
            ("fused_softmax", True),
            ("fused_bias_gelu", False),
            ("_scale_qkv_fan_out", False),
            # hidden states is for the full batch (allgathered when prescale_batch=False)
            # or full_batch == batch (if prescale_batch=True), and is sharded across hidden_dim
            ("full_hidden_states_sharded_hidden_dim", False),
            ("rotary_dim", None),
        ]
    )

    def __init__(self, *args, **kwargs):
        super(DistributedTransformer, self).__init__()

        self.optimize = core.cfg.optimize
        self.fp16_params = core.cfg.fp16_params
        config = self._parse_args(args, kwargs)

        if self.pre_layernorm and self.post_layernorm:
            use_post_layernorm = lambda l: (l == self.num_layers - 1)
        elif self.pre_layernorm and not self.post_layernorm:
            use_post_layernorm = lambda l: False
        elif not self.pre_layernorm and self.post_layernorm:
            use_post_layernorm = lambda l: True

        self.layers = []
        for l in range(self.num_layers):
            self.layers.append(
                DistributedTransformerLayer(
                    **update_copy_dict(
                        config,
                        input_layer=(l == 0),
                        output_layer=(l == self.num_layers - 1),
                        full_attention_mask_and_cross_states=(l > 0),
                        post_layernorm=use_post_layernorm(l),
                        num_layers=self.num_layers,
                        layer_idx=l,
                    )
                )
            )

        self.seq_layers = nn.Sequential(*self.layers)
        self._find_first_and_last_tx_layers()

    def _parse_args(self, args, kwargs):
        self.batch_dim = 0
        config = parse_args(args, kwargs, DistributedTransformer._KEYS, self)

        if not config["pre_layernorm"] and not config["post_layernorm"]:
            raise ValueError("At least one of pre_layernorm and post_layernorm must be True.")

        if self.fp32_residual_addition and MixedLayerNorm == None:
            raise ValueError("fp32 residual addition requires apex's MixedFusedLayerNorm.")

        if self.fp32_residual_addition and core.cfg.optimize == "memory":
            raise ValueError("fp32 residual addition only supports optimize == speed.")

        return config

    def _find_first_and_last_tx_layers(self):
        from smdistributed.modelparallel.torch.state_mod import state

        def first_last_tx_layer_hook(model, optimizer):
            first, last, first_pp_rank, last_pp_rank = None, None, None, None

            for m in model.modules():
                if isinstance(m, DistributedTransformerLayer):
                    if first_pp_rank is None:
                        first_pp_rank = state.module_manager.get_partition(m)
                    last_pp_rank = state.module_manager.get_partition(m)

                    if state.module_manager.get_partition(m) == pp_rank():
                        if first is None:
                            first = m
                        last = m

            if first is not None:
                first.is_first_tx_layer = True
                first.is_first_pp_rank = first_pp_rank == pp_rank()

            if last is not None:
                last.is_last_tx_layer = True
                last.is_last_pp_rank = last_pp_rank == pp_rank()

        state.module_manager.register_post_partition_hook(first_last_tx_layer_hook)

    def forward(self, inp):
        return self.seq_layers(inp)


class DistributedTransformerLayer(DistributedModule):
    _KEYS = OrderedDict(
        [
            ("num_attention_heads", 32),
            ("attention_head_size", 32),
            ("hidden_size", 1024),
            ("intermediate_size", 4096),
            ("attention_dropout_prob", 0.1),
            ("hidden_dropout_prob", 0.1),
            ("activation", "gelu"),
            ("layernorm_epsilon", 1e-5),
            ("initializer_range", 0.02),
            ("use_normal_initialization", False),
            ("causal_mask_size", None),
            ("add_cross_attention", False),
            ("pre_layernorm", False),
            ("post_layernorm", True),
            ("attention_in_fp32", False),
            ("query_key_layer_scaling", False),
            ("fp32_residual_addition", False),
            ("fused_softmax", True),
            ("fused_bias_gelu", False),
            ("input_layer", True),
            ("output_layer", True),
            # hidden states is for the full batch (allgathered when prescale_batch=False)
            # or full_batch == batch (if prescale_batch=True), and is sharded across hidden_dim
            ("full_hidden_states_sharded_hidden_dim", False),
            ("full_attention_mask_and_cross_states", False),
            ("_scale_qkv_fan_out", False),
            ("_precision_test", False),
            ("layer_idx", 0),
            ("rotary_dim", None),
        ]
    )

    def __init__(self, *args, **kwargs):
        super(DistributedTransformerLayer, self).__init__()

        # if self.input_layer is True, then this layer assumes that the input tensor has only the local batch,
        # and starts by allgathering the inputs across tp_ranks. Otherwise, assumes that the input already
        # has the full batch across tp_ranks. Similarly, self.output_layer is True, then it narrows the output
        # to only the local batch before returning. if called as part of smp.nn.DistributedTransformer, then
        # self.input_layer and self.output_layer is set automatically.

        self.fp16_params = core.cfg.fp16_params
        self.optimize = core.cfg.optimize

        self.is_first_pp_rank = False
        self.is_last_pp_rank = False
        self.is_first_tx_layer = False
        self.is_last_tx_layer = False
        config = self._parse_args(args, kwargs)

        if self.pre_layernorm and self.post_layernorm:
            att_layer_post_layernorm = False
        else:
            att_layer_post_layernorm = self.post_layernorm

        self.attention = DistributedAttentionLayer(
            **update_copy_dict(
                config,
                output_layer=False,
                full_attention_mask_and_cross_states=True,
                post_layernorm=att_layer_post_layernorm,
                num_layers=(self.num_layers if hasattr(self, "num_layers") else None),
            )
        )
        if self.add_cross_attention:
            self.cross_attention = DistributedAttentionLayer(
                **update_copy_dict(
                    config,
                    input_layer=False,
                    output_layer=False,
                    cross_attention=True,
                    full_attention_mask_and_cross_states=True,
                    post_layernorm=att_layer_post_layernorm,
                    num_layers=(self.num_layers if hasattr(self, "num_layers") else None),
                )
            )
        self.output = DistributedTransformerOutputLayer(
            **update_copy_dict(config, input_layer=False)
        )

    def _parse_args(self, args, kwargs):
        self.batch_dim = 0
        config = parse_args(args, kwargs, DistributedTransformerLayer._KEYS, self)

        if not config["pre_layernorm"] and not config["post_layernorm"]:
            raise ValueError("At least one of pre_layernorm and post_layernorm must be True.")

        if self.fp32_residual_addition and MixedLayerNorm == None:
            raise ValueError("fp32 residual addition requires apex's MixedFusedLayerNorm.")

        if self.fp32_residual_addition and core.cfg.optimize == "memory":
            raise ValueError("fp32 residual addition only supports optimize == speed.")

        return config

    def can_shard_activation_offloading(self):
        """ can shard activation offloading if and only if the activations are the same within the TP_GROUP """
        from smdistributed.modelparallel.torch.state_mod import state

        # when prescaled_batch is True, the first transformer layers in each pp_rank will have their inputs sharded
        # across sequence dimension, so in those cases we cannot shard activation offloading. the exception is the
        # first pp_rank. the input to the first transformer layer in the first pp_rank is not sharded because it does
        # not involve cross-pp_rank communication.
        if (
            state.cfg.prescaled_batch
            and not state.is_tracing()
            and self.is_first_tx_layer
            and not self.is_first_pp_rank
        ):
            return False

        if not state.cfg.prescaled_batch and (self.is_first_tx_layer or self.is_last_tx_layer):
            return False

        return True

    def forward(self, inp):
        from smdistributed.modelparallel.torch.state_mod import state

        if self.add_cross_attention:
            # cross_states: [local_bs, cross_seq_len, hidden_size]
            hidden_states, cross_states, attention_mask, cross_mask = inp
        else:
            # hidden: [local_bs, seq_len, hidden_size]
            # attention_mask: [local_bs, 1, 1, seq_len]
            hidden_states, attention_mask = inp

        if state.cfg.prescaled_batch and not state.is_tracing():
            if self.is_first_tx_layer and not self.is_first_pp_rank:
                seq_length = hidden_states.shape[0] * tp_size()
                hidden_states, = unshard_sequence(seq_length, hidden_states)

        if not state.cfg.prescaled_batch and not self.full_attention_mask_and_cross_states:
            # full_attention_mask: [full_bs, 1, 1, seq_len]
            full_attention_mask = allgather_for_tp(attention_mask, self.batch_dim)
            if self.add_cross_attention:
                if self.optimize == "memory":
                    # full_cross_states: [full_bs, cross_seq_len, local_hidden_size]
                    full_cross_states = scatter_and_merge_for_tp(
                        cross_states, merge_dim=self.batch_dim, split_dim=2
                    )
                else:
                    # full_cross_states: [full_bs, cross_seq_len, hidden_size]
                    full_cross_states = fused_allgather_for_tp(cross_states, self.batch_dim)

                full_cross_mask = allgather_for_tp(cross_mask, self.batch_dim)
        else:
            full_attention_mask = attention_mask
            if self.add_cross_attention:
                full_cross_states = cross_states
                full_cross_mask = cross_mask

        if self.fp32_residual_addition and self.is_first_tx_layer and self.is_first_pp_rank:
            # cast the hidden into fp32 from the very first transformer layer for the residual addition
            # it will be casted back to the origin dtype (which should be the same as the param dtype)
            # after the mixed layer norm, and casted again to fp32 after the residual addition
            hidden_states = hidden_states.to(torch.float32)

        # memory: [full_bs, seq_len, local_hidden_size]
        # speed:  [full_bs, seq_len, full_hidden_size]
        attention_output = self.attention((hidden_states, full_attention_mask))
        if self.add_cross_attention:
            attention_output = self.cross_attention(
                (attention_output, full_cross_states, full_cross_mask)
            )
        # not an output layer:
        # memory: [full_bs, seq_len, local_hidden_size]
        # speed:  [full_bs, seq_len, hidden_size]
        # output layer: [local_bs, seq_len, hidden_size]
        output = self.output(attention_output)

        if state.cfg.prescaled_batch and not state.is_tracing():
            if self.is_last_tx_layer and not self.is_last_pp_rank:
                output, = shard_sequence(output)

        if self.output_layer:
            local_bs = full_attention_mask.size(self.batch_dim) // tp_size()
            output_attention_mask = full_attention_mask.narrow(
                self.batch_dim, tp_rank() * local_bs, local_bs
            )
            if self.add_cross_attention:
                if self.optimize == "memory":
                    merge_shapes = get_merge_shapes(self.hidden_size)
                    # output_cross_states: [local_bs, cross_seq_len, hidden_size]
                    output_cross_states = scatter_and_merge_for_tp(
                        full_cross_states,
                        split_dim=self.batch_dim,
                        merge_dim=2,
                        merge_shapes=merge_shapes,
                    )
                else:
                    # output_cross_states: [local_bs, cross_seq_len, hidden_size]
                    output_cross_states = full_cross_states.narrow(
                        self.batch_dim, tp_rank() * local_bs, local_bs
                    )

                output_cross_mask = full_cross_mask.narrow(
                    self.batch_dim, tp_rank() * local_bs, local_bs
                )
        else:
            output_attention_mask = full_attention_mask
            if self.add_cross_attention:
                output_cross_states = full_cross_states
                output_cross_mask = full_cross_mask

        if self.add_cross_attention:
            return output, output_cross_states, output_attention_mask, output_cross_mask
        else:
            return output, output_attention_mask


class DistributedTransformerOutputLayer(DistributedModule):
    _KEYS = OrderedDict(
        [
            ("hidden_size", 1024),
            ("intermediate_size", 4096),
            ("hidden_dropout_prob", 0.1),
            ("activation", "gelu"),
            ("layernorm_epsilon", 1e-5),
            ("fused_bias_gelu", False),
            ("initializer_range", 0.02),
            ("use_normal_initialization", False),
            ("pre_layernorm", False),
            ("post_layernorm", True),
            ("input_layer", True),
            ("output_layer", True),
            ("fp32_residual_addition", False),
            # hidden states is for the full batch (allgathered when prescale_batch=False)
            # or full_batch == batch (if prescale_batch=True), and is sharded across hidden_dim
            ("full_hidden_states_sharded_hidden_dim", True),
            ("layer_idx", 0),
        ]
    )

    def __init__(self, *args, **kwargs):
        super(DistributedTransformerOutputLayer, self).__init__()

        self.fp16_params = core.cfg.fp16_params
        self.optimize = core.cfg.optimize
        config = self._parse_args(args, kwargs)
        dtype = torch.float16 if self.fp16_params else None
        self.use_hf_gelu = os.environ.get("SMP_USE_HF_GELU", None) == "1"

        with parameter_creation_scope(
            module=self,
            scaled_batch=True,
            dtype=dtype,
            use_normal=self.use_normal_initialization,
            initializer_range=self.initializer_range,
        ):
            if self.optimize == "memory":
                with initialize_with_input_partition(self, exclude_from_dist=["dense1.bias"]):
                    self.dense1 = nn.Linear(self.local_hidden_size, self.intermediate_size)
                    if tp_rank() != 0:
                        self.dense1.register_parameter("bias", None)
            else:
                with initialize_with_output_partition(self):
                    self.dense1 = nn.Linear(self.hidden_size, self.local_intermediate_size)

            with initialize_with_input_partition(self, exclude_from_dist=["dense2.bias"]):
                self.dense2 = nn.Linear(self.local_intermediate_size, self.hidden_size)
                if tp_rank() != 0:
                    self.dense2.register_parameter("bias", None)

            if self.optimize == "memory":
                with initialize_with_output_partition(self):
                    if self.pre_layernorm:
                        self.pre_layernorm = DistributedLayerNorm(
                            self.hidden_size, eps=self.layernorm_epsilon
                        )

                    if self.post_layernorm:
                        self.layernorm = DistributedLayerNorm(
                            self.hidden_size, eps=self.layernorm_epsilon
                        )
            else:
                LayerNormImpl = MixedLayerNorm if self.fp32_residual_addition else LayerNorm
                if self.pre_layernorm:
                    self.pre_layernorm = LayerNormImpl(self.hidden_size, eps=self.layernorm_epsilon)

                if self.post_layernorm:
                    self.layernorm = LayerNormImpl(self.hidden_size, eps=self.layernorm_epsilon)

        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def _parse_args(self, args, kwargs):
        config = parse_args(args, kwargs, DistributedTransformerOutputLayer._KEYS, self)
        self.local_intermediate_size = get_local_channels(self.intermediate_size)
        self.local_hidden_size = get_local_channels(self.hidden_size)
        self.batch_dim = 0

        if not self.post_layernorm and not self.pre_layernorm and not self.output_layer:
            raise ValueError(
                f"DistributedTransformerOutputLayer must include LayerNorm if called as part of another distributed module."
            )

        if self.activation not in {"gelu", "relu"}:
            raise ValueError(
                "Only relu and gelu activations are supported by DistributedTransformerOutput layer."
            )

        if self.fp32_residual_addition:
            if MixedLayerNorm == None:
                raise ValueError("fp32 residual addition requires apex's MixedFusedLayerNorm.")

            if core.cfg.optimize == "memory":
                raise ValueError("fp32 residual addition only supports optimize == speed.")

            if not self.pre_layernorm:
                raise ValueError("fp32 residual addition requires pre-layernorm to be true.")

        return config

    def forward(self, inp):
        from smdistributed.modelparallel.torch.state_mod import state

        if self.input_layer:
            if self.optimize == "memory":
                # full_input: [full_bs, seq_len, local_hidden_size]
                full_input = scatter_and_merge_for_tp(inp, split_dim=2, merge_dim=self.batch_dim)
            else:
                # full_input: [full_bs, seq_len, hidden_size]
                full_input = allgather_for_tp(inp, 0)
        else:
            full_input = inp

        res_input = full_input

        if self.pre_layernorm:
            full_input = self.pre_layernorm(full_input)

        fused_bias = (
            self.activation == "gelu"
            and not self.use_hf_gelu
            and self.fused_bias_gelu
            and core.cfg.optimize == "speed"
        )
        bias = None if fused_bias else self.dense1.bias

        if self.optimize == "memory":
            # dense1_output: [full_bs, seq_len, intermediate_size]
            dense1_output = F.linear(full_input, self.dense1.weight, bias)
            # dense1_output: [full_bs, seq_len, local_intermediate_size]
            dense1_output = reduce_scatter_for_tp(dense1_output, 2)
        else:
            mixed_input = bwd_allreduce_for_tp(full_input)
            # dense1_output: [full_bs, seq_len, local_intermediate_size]
            dense1_output = F.linear(mixed_input, self.dense1.weight, bias)

        if self.activation == "gelu":
            # F.gelu is faster, so default to that implementation
            if self.use_hf_gelu:
                act_output = gelu_new(dense1_output)
            else:
                if fused_bias:
                    act_output = bias_gelu_impl(dense1_output, self.dense1.bias)
                else:
                    act_output = F.gelu(dense1_output)
        else:
            act_output = F.relu(dense1_output)

        # dense2_output: [full_bs, seq_len, hidden_size]
        dense2_output = F.linear(act_output, self.dense2.weight, self.dense2.bias)

        if self.optimize == "memory":
            # dense2_output: [full_bs, seq_len, local_hidden_size]
            dense2_output = reduce_scatter_for_tp(dense2_output, 2)
        else:
            # dense2_output: [full_bs, seq_len, hidden_size]
            dense2_output = fwd_allreduce_for_tp(dense2_output)

        with state.rng_manager.consistent_rng_state(enabled=state.cfg.prescaled_batch):
            intermediate = self.dropout(dense2_output)

        layer_output = res_input + intermediate

        if self.post_layernorm:
            layer_output = self.layernorm(layer_output)

        if not state.cfg.prescaled_batch and self.output_layer:
            if self.optimize == "memory":
                if not self.full_hidden_states_sharded_hidden_dim:
                    merge_shapes = get_merge_shapes(self.hidden_size)
                    # layer_output: [local_bs, seq_len, hidden_size]
                    layer_output = scatter_and_merge_for_tp(
                        layer_output,
                        split_dim=self.batch_dim,
                        merge_dim=2,
                        merge_shapes=merge_shapes,
                    )
                else:
                    # layer_output: [full_bs, seq_len, hidden_size]
                    layer_output = fused_allgather_for_tp(layer_output, 2)
            else:
                # layer_output, self.full_hidden_states_sharded_hidden_dim == False: [local_bs, seq_len, hidden_size]
                # layer_output, self.full_hidden_states_sharded_hidden_dim == True: [full_bs, seq_len, hidden_size]
                if not self.full_hidden_states_sharded_hidden_dim:
                    layer_output = narrow_for_tp(layer_output, self.batch_dim)

        return layer_output


class DistributedAttentionLayer(DistributedModule):
    _KEYS = OrderedDict(
        [
            ("num_attention_heads", 32),
            ("attention_head_size", 32),
            ("hidden_size", 1024),
            ("attention_dropout_prob", 0.1),
            ("hidden_dropout_prob", 0.1),
            ("layernorm_epsilon", 1e-5),
            ("initializer_range", 0.02),
            ("use_normal_initialization", False),
            ("cross_attention", False),
            ("causal_mask_size", None),
            ("pre_layernorm", False),
            ("post_layernorm", True),
            ("attention_in_fp32", False),
            ("query_key_layer_scaling", False),
            ("fp32_residual_addition", False),
            ("fused_softmax", True),
            ("input_layer", True),
            ("output_layer", True),
            ("full_attention_mask_and_cross_states", False),
            # hidden states is for the full batch (allgathered when prescale_batch=False)
            # or full_batch == batch (if prescale_batch=True), and is sharded across hidden_dim
            ("full_hidden_states_sharded_hidden_dim", False),
            ("_scale_qkv_fan_out", False),
            ("layer_idx", 0),
            ("rotary_dim", None),
        ]
    )

    def __init__(self, *args, **kwargs):
        from smdistributed.modelparallel.torch.utils import add_weight_split_shapes

        super(DistributedAttentionLayer, self).__init__()

        self.fp16_params = core.cfg.fp16_params
        self.optimize = core.cfg.optimize
        config = self._parse_args(args, kwargs)
        dtype = torch.float16 if self.fp16_params else None
        self.weight_split_shapes = {}

        with parameter_creation_scope(
            module=self,
            scaled_batch=True,
            dtype=dtype,
            use_normal=self.use_normal_initialization,
            initializer_range=self.initializer_range,
        ):
            if self.optimize == "memory":
                if not self._scale_qkv_fan_out:
                    q_fan_out_scale = 1
                    kv_fan_out_scale = 1
                elif not self.cross_attention:
                    q_fan_out_scale = 1
                    kv_fan_out_scale = 2
                else:
                    q_fan_out_scale = 3
                    kv_fan_out_scale = 3

                with initialize_with_input_partition(
                    self, fan_out_scale=q_fan_out_scale, exclude_from_dist=["query.bias"]
                ):
                    self.query = nn.Linear(self.local_hidden_size, self.total_attention_size)
                    if tp_rank() != 0:
                        self.query.register_parameter("bias", None)

                with initialize_with_input_partition(
                    self,
                    fan_out_scale=kv_fan_out_scale,
                    exclude_from_dist=["key.bias", "value.bias"],
                ):
                    self.key = nn.Linear(self.local_hidden_size, self.total_attention_size)
                    self.value = nn.Linear(self.local_hidden_size, self.total_attention_size)
                    if tp_rank() != 0:
                        self.key.register_parameter("bias", None)
                        self.value.register_parameter("bias", None)
            else:
                with initialize_with_output_partition(self):
                    self.query = nn.Linear(self.hidden_size, self.local_attention_size)
                    self.key = nn.Linear(self.hidden_size, self.local_attention_size)
                    self.value = nn.Linear(self.hidden_size, self.local_attention_size)
                    add_weight_split_shapes(self.query.weight, self.all_local_attention_sizes)
                    add_weight_split_shapes(self.key.weight, self.all_local_attention_sizes)
                    add_weight_split_shapes(self.value.weight, self.all_local_attention_sizes)
                    add_weight_split_shapes(self.query.bias, self.all_local_attention_sizes)
                    add_weight_split_shapes(self.key.bias, self.all_local_attention_sizes)
                    add_weight_split_shapes(self.value.bias, self.all_local_attention_sizes)

            self.dropout1 = nn.Dropout(self.attention_dropout_prob)

            with initialize_with_input_partition(self, exclude_from_dist=["dense.bias"]):
                self.dense = nn.Linear(self.local_attention_size, self.hidden_size)
                add_weight_split_shapes(self.dense.weight, self.all_local_attention_sizes)
                if tp_rank() != 0:
                    self.dense.register_parameter("bias", None)

            if self.optimize == "memory":
                with initialize_with_output_partition(self):
                    if self.pre_layernorm:
                        self.pre_layernorm = DistributedLayerNorm(
                            self.hidden_size, eps=self.layernorm_epsilon
                        )

                    if self.post_layernorm:
                        self.layernorm = DistributedLayerNorm(
                            self.hidden_size, eps=self.layernorm_epsilon
                        )
            else:
                LayerNormImpl = MixedLayerNorm if self.fp32_residual_addition else LayerNorm
                if self.pre_layernorm:
                    self.pre_layernorm = LayerNormImpl(self.hidden_size, eps=self.layernorm_epsilon)

                if self.post_layernorm:
                    self.layernorm = LayerNormImpl(self.hidden_size, eps=self.layernorm_epsilon)

        self.dropout2 = nn.Dropout(self.hidden_dropout_prob)

        scaling_factor = self.num_layers if self.query_key_layer_scaling else 1.0
        self.fused_masked_softmax = ScaledMaskedSoftmax(scaling_factor)

        if self.causal_mask_size is not None:
            self.fused_upper_triang_softmax = ScaledCausalMaskedSoftmax(scaling_factor)
            self.warned_fused_softmax_attention_mask = False

            device = torch.device("cuda", local_rank())
            self.causal_mask = torch.tril(
                torch.ones(
                    (self.causal_mask_size, self.causal_mask_size), dtype=torch.uint8, device=device
                )
            )
            self.causal_mask = self.causal_mask.reshape(
                1, 1, self.causal_mask_size, self.causal_mask_size
            )
            self.mask_value = torch.tensor(-1e4, dtype=torch.float32, device=device)
            if self.query_key_layer_scaling:
                self.mask_value /= self.num_layers

    def _parse_args(self, args, kwargs):
        config = parse_args(args, kwargs, DistributedAttentionLayer._KEYS, self)

        if not self.pre_layernorm and not self.post_layernorm and not self.output_layer:
            raise ValueError(
                f"DistributedAttentionLayer must include LayerNorm if called as part of another distributed module."
            )

        if self.query_key_layer_scaling:
            if "num_layers" not in kwargs:
                raise ValueError(
                    "To use query_key_layer_scaling feature, num_layers keyword argument must also be supplied."
                )

            self.num_layers = kwargs["num_layers"]

        if self.fp32_residual_addition:
            if MixedLayerNorm == None:
                raise ValueError("fp32 residual addition requires apex's MixedFusedLayerNorm.")

            if core.cfg.optimize == "memory":
                raise ValueError("fp32 residual addition only supports optimize == speed.")

            if not self.pre_layernorm:
                raise ValueError("fp32 residual addition requires pre-layernorm to be true.")

        self.total_attention_size = self.num_attention_heads * self.attention_head_size
        self.local_attention_heads = get_local_channels(self.num_attention_heads)
        self.local_attention_size = self.local_attention_heads * self.attention_head_size
        # local_attention_sizes for each tp rank
        self.all_local_attention_sizes = [
            self.attention_head_size * get_local_channels(self.num_attention_heads, rank=r)
            for r in range(tp_size())
        ]
        self.local_hidden_size = get_local_channels(self.hidden_size)
        self.batch_dim = 0
        return config

    def forward(self, inp):
        from smdistributed.modelparallel.torch.state_mod import state

        if self.cross_attention:
            # cross_states: [local_bs, cross_seq_len, hidden_size]
            hidden_states, cross_states, attention_mask = inp
        else:
            # hidden: [local_bs, seq_len, hidden_size]
            # attention_mask: [local_bs, 1, 1, seq_len]
            hidden_states, attention_mask = inp

        if not self.full_attention_mask_and_cross_states:
            # full_attention_mask: [full_bs, 1, 1, seq_len]
            full_attention_mask = allgather_for_tp(attention_mask, self.batch_dim)
            if self.cross_attention:
                if self.optimize == "memory":
                    mixed_cross_states = scatter_and_merge_for_tp(
                        cross_states, merge_dim=self.batch_dim, split_dim=2
                    )
                else:
                    mixed_cross_states = fused_allgather_for_tp(cross_states, self.batch_dim)
        else:
            full_attention_mask = attention_mask
            if self.cross_attention:
                mixed_cross_states = cross_states

        if not state.cfg.prescaled_batch and self.input_layer:
            if self.optimize == "memory":
                if self.full_hidden_states_sharded_hidden_dim:
                    # full_hidden_states: [full_bs, seq_len, local_hidden_size]
                    full_hidden_states = hidden_states

                else:
                    # full_hidden_states: [full_bs, seq_len, local_hidden_size]
                    full_hidden_states = scatter_and_merge_for_tp(
                        hidden_states, split_dim=2, merge_dim=self.batch_dim
                    )
            else:
                if self.full_hidden_states_sharded_hidden_dim:
                    # full_hidden_states: [full_bs, seq_len, hidden_size]
                    # Allgather across hidden_dim
                    merge_shapes = get_merge_shapes(self.hidden_size)
                    full_hidden_states = allgather_for_tp(hidden_states, -1, merge_shapes)
                else:
                    full_hidden_states = allgather_for_tp(hidden_states, self.batch_dim)
        else:
            full_hidden_states = hidden_states

        res_hidden_states = full_hidden_states

        if self.pre_layernorm:
            full_hidden_states = self.pre_layernorm(full_hidden_states)

        if self.optimize == "memory":
            # mixed_query_layer: [full_bs, seq_len, total_attention_size]
            mixed_query_layer = F.linear(full_hidden_states, self.query.weight, self.query.bias)

            if not self.cross_attention:
                # mixed_key_layer: [full_bs, seq_len, total_attention_size]
                # mixed_value_layer: [full_bs, seq_len, total_attention_size]
                mixed_key_layer = F.linear(full_hidden_states, self.key.weight, self.key.bias)
                mixed_value_layer = F.linear(full_hidden_states, self.value.weight, self.value.bias)
            else:
                mixed_key_layer = F.linear(mixed_cross_states, self.key.weight, self.key.bias)
                mixed_value_layer = F.linear(mixed_cross_states, self.value.weight, self.value.bias)

            # mixed_query_layer: [full_bs, seq_len, local_attention_size]
            mixed_query_layer = reduce_scatter_for_tp(
                mixed_query_layer, 2, split_shapes=self.all_local_attention_sizes
            )

            batch_size = mixed_query_layer.size(0)
            combined_kv = torch.cat((mixed_key_layer, mixed_value_layer), 0).contiguous()
            reduced_kv = reduce_scatter_for_tp(
                combined_kv, 2, split_shapes=self.all_local_attention_sizes
            )

            # mixed_key_layer: [full_bs, seq_len, local_attention_size]
            # mixed_value_layer: [full_bs, seq_len, local_attention_size]
            mixed_key_layer = reduced_kv.narrow(0, 0, batch_size)
            mixed_value_layer = reduced_kv.narrow(0, batch_size, batch_size)
        else:
            mixed_hidden_states = bwd_allreduce_for_tp(full_hidden_states)
            # mixed_query_layer: [full_bs, seq_len, local_attention_size]
            mixed_query_layer = F.linear(mixed_hidden_states, self.query.weight, self.query.bias)

            if not self.cross_attention:
                # mixed_key_layer: [full_bs, seq_len, local_attention_size]
                # mixed_value_layer: [full_bs, seq_len, local_attention_size]
                mixed_key_layer = F.linear(mixed_hidden_states, self.key.weight, self.key.bias)
                mixed_value_layer = F.linear(
                    mixed_hidden_states, self.value.weight, self.value.bias
                )
            else:
                mixed_key_layer = F.linear(mixed_cross_states, self.key.weight, self.key.bias)
                mixed_value_layer = F.linear(mixed_cross_states, self.value.weight, self.value.bias)

        if core.cfg.checkpoint_attentions:
            context_layer = torch.utils.checkpoint.checkpoint(
                self.attn,
                mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer,
                full_attention_mask,
                self.attention_in_fp32,
                self.query_key_layer_scaling,
            )
        else:
            # context_layer: [full_bs, local_attention_heads, seq_len, attention_head_size]
            context_layer = self.attn(
                mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer,
                full_attention_mask,
                self.attention_in_fp32,
                self.query_key_layer_scaling,
            )

        # context_layer: [full_bs, seq_len, local_attention_heads, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.local_attention_size,)
        # context_layer: [full_bs, seq_len, local_attention_size]
        context_layer = torch.reshape(context_layer, new_context_layer_shape)

        # self_attention: [full_bs, seq_len, hidden_size]
        self_attention = F.linear(context_layer, self.dense.weight, self.dense.bias)

        if self.optimize == "memory":
            # self_attention: [full_bs, seq_len, local_hidden_size]
            self_attention = reduce_scatter_for_tp(self_attention, 2)
        else:
            # self_attention: [full_bs, seq_len, full_hidden_size]
            self_attention = fwd_allreduce_for_tp(self_attention)

        with state.rng_manager.consistent_rng_state(enabled=state.cfg.prescaled_batch):
            self_attention = self.dropout2(self_attention)

        layer_output = self_attention + res_hidden_states

        if self.post_layernorm:
            layer_output = self.layernorm(layer_output)

        if self.output_layer:
            if self.optimize == "memory":
                merge_shapes = get_merge_shapes(self.hidden_size)
                # layer_output: [local_bs, seq_len, full_hidden_size]
                layer_output = scatter_and_merge_for_tp(
                    layer_output, split_dim=self.batch_dim, merge_dim=2, merge_shapes=merge_shapes
                )
            else:
                # layer_output: [local_bs, seq_len, full_hidden_size]
                layer_output = narrow_for_tp(layer_output, self.batch_dim)

        return layer_output

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(
                0, 1, 3, 2, 4
            )  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )

    def _update_with_rotary_pos_emb(self, query, key, value):

        query = self._split_heads(query, self.local_attention_heads, self.attention_head_size, True)
        key = self._split_heads(key, self.local_attention_heads, self.attention_head_size, True)
        value = self._split_heads(
            value, self.local_attention_heads, self.attention_head_size, False
        )

        seq_len = key.shape[1]
        offset = 0

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 3, 1)
        query = query.permute(0, 2, 1, 3)

        if self.query_key_layer_scaling:
            new_shape_key = (-1,) + key.size()[2:]
            # [batch * num_heads, head_size, seq_len]
            key = key.reshape(new_shape_key)

            new_shape_query = (-1,) + query.size()[2:]
            # [batch * num_heads, head_size, seq_len]
            query = query.reshape(new_shape_query)

        if self.fp16_params:
            query = query.to(torch.float16)
            key = key.to(torch.float16)

        return query, key, value

    def attn(
        self,
        mixed_query_layer,
        mixed_key_layer,
        mixed_value_layer,
        full_attention_mask,
        attention_in_fp32=False,
        query_key_layer_scaling=True,
    ):
        from smdistributed.modelparallel.torch.state_mod import state

        if self.rotary_dim is not None:
            query_layer, key_layer, value_layer = self._update_with_rotary_pos_emb(
                mixed_query_layer, mixed_key_layer, mixed_value_layer
            )
        else:
            # query_layer: [full_bs, local_attention_heads, seq_len, attention_head_size]
            # if query_key_layer_scaling: [full_bs * local_attention_heads, seq_len, attention_head_size]
            query_layer = self.transpose_for_scores(mixed_query_layer, component="query")

            # key_layer: [full_bs, local_attention_heads, attention_head_size, seq_len]
            # if query_key_layer_scaling: [full_bs * local_attention_heads, attention_head_size, seq_len]
            key_layer = self.transpose_for_scores(mixed_key_layer, component="key")

            # value_layer: [full_bs, local_attention_heads, seq_len, attention_head_size]
            value_layer = self.transpose_for_scores(mixed_value_layer, component="value")

        # no cross att: [full_bs, local_attention_heads, seq_len, seq_len]
        # cross att: [full_bs, local_attention_heads, seq_len, cross_seq_len]

        attention_scores = self.compute_attention_scores(
            query_layer, key_layer, attention_in_fp32, query_key_layer_scaling
        )
        attention_probs = self.compute_attention_probs(attention_scores, full_attention_mask)

        with state.rng_manager.consistent_rng_state(enabled=state.cfg.prescaled_batch):
            # no cross att: [full_bs, local_attention_heads, seq_len, seq_len]
            # cross att: [full_bs, local_attention_heads, seq_len, cross_seq_len]
            attention_probs = self.dropout1(attention_probs)

        # [full_bs, local_attention_heads, seq_len, attention_head_size]
        return torch.matmul(attention_probs, value_layer)

    def transpose_for_scores(self, tensor, component="query"):
        new_shape = tensor.size()[:-1] + (self.local_attention_heads, self.attention_head_size)

        # [batch, seq, num_heads, head_size]
        tensor = tensor.reshape(new_shape)

        if component == "key":
            # [batch, num_heads, head_size, seq_len]
            tensor = tensor.permute(0, 2, 3, 1)
        else:
            # [batch, num_heads, seq_len, head_size]
            tensor = tensor.permute(0, 2, 1, 3)

        if self.query_key_layer_scaling and component != "value":
            new_shape = (-1,) + tensor.size()[2:]

            # [batch * num_heads, head_size, seq_len]
            tensor = tensor.reshape(new_shape)

        return tensor

    def compute_attention_probs(self, attention_scores, full_attention_mask):
        """ Compute attention probabilities from scores. Choose algorithm/kernel based on input and configuration. """

        # apply causal mask if needed (e.g. GPT-2 self-attention layers)
        causal_mask = self.causal_mask_size is not None and not self.cross_attention
        if causal_mask and can_use_fused_kernel(self, attention_scores.shape):
            if not self.warned_fused_softmax_attention_mask and tp_rank() == 0 and rdp_rank() == 0:
                logger.warning(
                    "Using fused softmax kernel in attention computation, which ignores the attention mask input. To use an attention mask that masks at least one token, disable the fused softmax kernel by passing fused_softmax=False into the smp.tensor_parallelism or smp.set_tensor_parallelism calls."
                )
                self.warned_fused_softmax_attention_mask = True
            attention_probs = self.fused_upper_triang_softmax(attention_scores)
        elif causal_mask:
            attention_scores = self.apply_causal_mask(attention_scores)
            attention_probs = self.compute_attention_probs_nonfused(
                attention_scores, full_attention_mask
            )

        elif can_use_fused_kernel(self, attention_scores.shape):
            full_attention_mask = self._format_attention_mask(
                full_attention_mask, attention_scores.shape
            )
            attention_probs = self.fused_masked_softmax(attention_scores, full_attention_mask)
        else:
            attention_probs = self.compute_attention_probs_nonfused(
                attention_scores, full_attention_mask
            )

        return attention_probs

    def apply_causal_mask(self, attention_scores):
        """ Apply upper triangular mask on the attention scores, for GPT-type autoregressive models. """

        from smdistributed.modelparallel.torch.state_mod import state

        if state.is_tracing() and state.model.current_trace_device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", local_rank())

        self.causal_mask = self.causal_mask.to(device)
        self.mask_value = self.mask_value.to(device)

        seq_len1, seq_len2 = attention_scores.size(-2), attention_scores.size(-1)
        sliced_mask = self.causal_mask[:, :, (seq_len2 - seq_len1) : seq_len2, :seq_len2].bool()
        return torch.where(
            sliced_mask, attention_scores, self.mask_value.to(attention_scores.dtype)
        )

    def _format_attention_mask(self, attention_mask, scores_shape):
        """ To use the fused masked kernel, attention mask should have dtype uint8, shape [B, 1, 1, Sk] or [B, 1, Sq, Sk], and the masked locations
        set to 1, while the rest set to 0. Sk = sequence length for key, Sq = sequence length for query. """

        batch, _, query_seq_len, key_seq_len = scores_shape

        if (
            attention_mask.dim() != 4
            or attention_mask.size(0) != batch
            or attention_mask.size(1) != 1
            or attention_mask.size(2) not in [1, query_seq_len]
            or attention_mask.size(3) != key_seq_len
        ):
            raise ValueError(
                "To use the fused masked softmax, attention mask must have the shape [batch, 1, sequence_length, sequence_length] or [batch, 1, sequence_length, sequence_length]."
            )

        attention_mask = (attention_mask != 0.0).to(torch.uint8)

        if attention_mask.size(2) == 1:
            return attention_mask.repeat((1, 1, query_seq_len, 1))
        else:
            return attention_mask

    def compute_attention_probs_nonfused(self, attention_scores, full_attention_mask):
        if self.query_key_layer_scaling:
            existing_dtype = attention_scores.dtype
            attention_scores = attention_scores.float() * self.num_layers
            full_attention_mask = full_attention_mask.float()

        attention_scores = attention_scores + full_attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)

        if self.query_key_layer_scaling:
            attention_probs = attention_probs.to(existing_dtype)

        return attention_probs

    def compute_attention_scores(
        self, query_layer, key_layer, attention_in_fp32=False, query_key_layer_scaling=True
    ):
        existing_dtype = query_layer.dtype

        if attention_in_fp32:
            # execute query/key multiplication in fp32 to avoid overflow
            query_layer = query_layer.to(torch.float32)
            key_layer = key_layer.to(torch.float32)

        if query_key_layer_scaling:
            # (batch, num_heads, seq_len for query, seq_len for key)
            # seq_len for key and query will be the same unless cross attention
            # is used with encoder/decoder having different sequence lengths

            # [full_bs, local_attention_heads, seq_len (query), seq_len (key)]
            output_size = (
                query_layer.size(0) // self.local_attention_heads,
                self.local_attention_heads,
                query_layer.size(1),
                key_layer.size(2),
            )

            matmul_result = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=torch.device("cuda", local_rank()),
            )

            div = math.sqrt(self.attention_head_size) * self.num_layers

            # [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_result, query_layer, key_layer, beta=0.0, alpha=(1.0 / div)
            )
            attention_scores = matmul_result.view(*output_size)
        else:
            attention_scores = torch.matmul(query_layer, key_layer)
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        return attention_scores.to(existing_dtype)
