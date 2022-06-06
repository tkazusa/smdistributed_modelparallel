# Future
from __future__ import absolute_import, division, print_function, unicode_literals

# Standard Library
import math

# Third Party
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter

# First Party
from smdistributed.modelparallel.torch.core import core, local_rank
from smdistributed.modelparallel.torch.nn.transformer import gelu_new
from smdistributed.modelparallel.torch.nn.utils import update_copy_dict

try:
    from smdistributed.modelparallel.torch.nn import FusedLayerNorm

    LayerNorm = FusedLayerNorm
except ImportError:
    LayerNorm = nn.LayerNorm

def bias_gelu_training(bias, y):
    x = bias + y
    return gelu_new(x)
def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).float()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(2, 3), sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

class LinearActivation(Module):
    r"""Fused Linear and activation Module.
    """
    __constants__ = ["bias"]

    def __init__(self, in_features, out_features, act="gelu"):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.biased_act_fn = bias_gelu_training
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.biased_act_fn(self.bias, F.linear(input, self.weight, None))


class TransformerConfig(object):
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_head_size=64,
        intermediate_size=3072,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        layernorm_epsilon=1e-5,
        post_layernorm=True,
        pre_layernorm=True,
        causal_mask_size=None,
        add_cross_attention=False,
        rotary_dim=None,
        _precision_test=True,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layernorm_epsilon = layernorm_epsilon
        self.post_layernorm = post_layernorm
        self.pre_layernorm = pre_layernorm
        self.causal_mask_size = causal_mask_size
        self.add_cross_attention = add_cross_attention
        self.rotary_dim = rotary_dim
        self._precision_test = _precision_test

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_size": self.attention_head_size,
            "intermediate_size": self.intermediate_size,
            "attention_dropout_prob": self.attention_dropout_prob,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "layernorm_epsilon": self.layernorm_epsilon,
            "post_layernorm": self.post_layernorm,
            "pre_layernorm": self.pre_layernorm,
            "causal_mask_size": self.causal_mask_size,
            "add_cross_attention": self.add_cross_attention,
            "rotary_dim": self.rotary_dim,
            "_precision_test": self._precision_test,
        }


class TransformerLMHeadConfig(object):
    def __init__(
        self,
        num_layers=4,
        num_attention_heads=24,
        attention_head_size=64,
        hidden_size=1536,
        intermediate_size=6144,
        vocab_size=50257,
        num_positions=1024,
        attention_dropout_prob=0,
        hidden_dropout_prob=0,
        activation="gelu",
        layernorm_epsilon=1e-05,
        causal_mask_size=1024,
        add_cross_attention=False,
        add_lm_head=True,
        initializer_range=0.02,
        use_normal_initialization=True,
        pre_layernorm=True,
        post_layernorm=True,
        _scale_qkv_fan_out=True,
        num_token_types=0,
        rotary_dim=None,
    ):
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.activation = activation
        self.layernorm_epsilon = layernorm_epsilon
        self.causal_mask_size = causal_mask_size
        self.add_cross_attention = add_cross_attention
        self.add_lm_head = add_lm_head
        self.initializer_range = initializer_range
        self.use_normal_initialization = use_normal_initialization
        self.pre_layernorm = pre_layernorm
        self.post_layernorm = post_layernorm
        self._scale_qkv_fan_out = _scale_qkv_fan_out
        self.num_token_types = num_token_types
        self.rotary_dim = rotary_dim

    def to_dict(self):
        return {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_size": self.attention_head_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "num_positions": self.num_positions,
            "attention_dropout_prob": self.attention_dropout_prob,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "activation": self.activation,
            "layernorm_epsilon": self.layernorm_epsilon,
            "causal_mask_size": self.causal_mask_size,
            "add_cross_attention": self.add_cross_attention,
            "add_lm_head": self.add_lm_head,
            "initializer_range": self.initializer_range,
            "use_normal_initialization": self.use_normal_initialization,
            "pre_layernorm": self.pre_layernorm,
            "post_layernorm": self.post_layernorm,
            "_scale_qkv_fan_out": self._scale_qkv_fan_out,
            "num_token_types": self.num_token_types,
            "rotary_dim": self.rotary_dim,
        }


class TransformerSelfAttention(nn.Module):
    def __init__(self, config):
        super(TransformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.use_pre_layernorm = config.pre_layernorm
        self.add_cross_attention = config.add_cross_attention
        self.rotary_dim = config.rotary_dim
        self.fp16_params = core.cfg.fp16_params

        if self.use_pre_layernorm:
            self.PreLayerNorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

        self.causal_mask_size = config.causal_mask_size
        if self.causal_mask_size is not None:
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")


    def _update_with_rotary_pos_emb(self, query, key, value):

        query = self._split_heads(query, self.num_attention_heads, self.attention_head_size, True)
        key = self._split_heads(key, self.num_attention_heads, self.attention_head_size, True)
        value = self._split_heads(value, self.num_attention_heads, self.attention_head_size, False)
        
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

        if self.fp16_params:
            query = query.to(torch.float16)
            key = key.to(torch.float16)

        return query, key, value

    def forward(self, hidden_states, attention_mask):
        if self.use_pre_layernorm:
            hidden_states = self.PreLayerNorm(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.rotary_dim is not None:
            query_layer, key_layer, value_layer = self._update_with_rotary_pos_emb(mixed_query_layer, mixed_key_layer, mixed_value_layer)
        else:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_key_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if self.causal_mask_size is not None and not self.add_cross_attention:
            device = torch.device("cuda", local_rank())
            self.causal_mask = self.causal_mask.to(device)
            self.mask_value = self.mask_value.to(device)

            seq_len1, seq_len2 = attention_scores.size(-2), attention_scores.size(-1)
            sliced_mask = self.causal_mask[:, :, (seq_len2 - seq_len1) : seq_len2, :seq_len2].bool()
            attention_scores = torch.where(
                sliced_mask, attention_scores, self.mask_value.to(attention_scores.dtype)
            )

        # Apply the attention mask is (precomputed for all layers in TransformerModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


class TransformerSelfOutput(nn.Module):
    def __init__(self, config):
        super(TransformerSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_post_layernorm = config.post_layernorm
        if self.use_post_layernorm:
            self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        layer_output = hidden_states + input_tensor
        if self.use_post_layernorm:
            layer_output = self.LayerNorm(layer_output)
        return layer_output


class TransformerAttention(nn.Module):
    def __init__(self, config):
        super(TransformerAttention, self).__init__()
        self.self = TransformerSelfAttention(config)
        self.output = TransformerSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class TransformerIntermediate(nn.Module):
    def __init__(self, config):
        super(TransformerIntermediate, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size, config.intermediate_size)
        self.use_pre_layernorm = config.pre_layernorm
        if self.use_pre_layernorm:
            self.PreLayerNorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states):
        if self.use_pre_layernorm:
            hidden_states = self.PreLayerNorm(hidden_states)
        layer_output = self.dense_act(hidden_states)
        return layer_output


class TransformerOutput(nn.Module):
    def __init__(self, config):
        super(TransformerOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.use_post_layernorm = config.post_layernorm
        if self.use_post_layernorm:
            self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        layer_output = hidden_states + input_tensor
        if self.use_post_layernorm:
            layer_output = self.LayerNorm(layer_output)
        return layer_output


class TransformerLMHead(nn.Module):
    def __init__(self, config):
        super(TransformerLMHead, self).__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_dim = config.rotary_dim
        if self.rotary_dim is None:
            self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)
        if config.num_token_types > 0:
            self.token_type_embedding = nn.Embedding(config.num_token_types, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.transformer = Transformer(config)
        self.ce_loss = nn.CrossEntropyLoss()
        self.fp16_params = core.cfg.fp16_params
        if config.add_lm_head:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.word_embedding.weight
        self.add_cross_attention = config.add_cross_attention

    def forward(self, inputs):
        if self.add_cross_attention:
            input_ids, attention_mask, token_type_ids, position_ids, cross_states, cross_mask, labels = (
                inputs
            )
        else:
            input_ids, attention_mask, token_type_ids, position_ids, labels = inputs

        input_shape = input_ids.size()
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        device = torch.device("cuda", local_rank())

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        if self.fp16_params:
            attention_mask = attention_mask.to(torch.float16)
        attention_mask = (1.0 - attention_mask) * -10000.0

        if self.add_cross_attention:
            _, cross_seq_length, _ = cross_states.size()
            cross_hidden_shape = (batch_size, cross_seq_length)
            if cross_mask is None:
                cross_mask = torch.ones(cross_hidden_shape, device=device)
            cross_mask = cross_mask[:, None, None, :]
            if self.fp16_params:
                cross_mask = cross_mask.to(torch.float16)
                cross_mask = (1.0 - cross_mask) * -10000.0
            else:
                cross_mask = (1.0 - cross_mask) * -1e9

        else:
            cross_mask = None

        if self.rotary_dim is None:
            hidden_states = self.word_embedding(input_ids) + self.position_embedding(position_ids)
        else:
            hidden_states = self.word_embedding(input_ids)
        if token_type_ids is not None:
            if self.num_token_types > 0:
                hidden_states += self.token_type_embedding(token_type_ids)
            else:
                # HF GPT-2 uses word embedding as token type embedding too
                hidden_states += self.word_embedding(token_type_ids)
        hidden_states = self.dropout(hidden_states)

        # transformer
        if self.add_cross_attention:
            tx_inputs = hidden_states, attention_mask, cross_states, cross_mask
        else:
            tx_inputs = hidden_states, attention_mask
        hidden_states, *_ = self.transformer(tx_inputs)
        lm_logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss, lm_logits)
        return lm_logits


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        if config.pre_layernorm and config.post_layernorm:
            use_post_layernorm = lambda l: (l == config.num_layers - 1)
        elif config.pre_layernorm and not config.post_layernorm:
            use_post_layernorm = lambda l: False
        elif not config.pre_layernorm and config.post_layernorm:
            use_post_layernorm = lambda l: True

        self.layers = []
        for l in range(config.num_layers):
            self.layers.append(
                TransformerLayer(
                    TransformerConfig(
                        **update_copy_dict(config.to_dict(), post_layernorm=use_post_layernorm(l))
                    )
                )
            )
        self.seq_layers = nn.Sequential(*self.layers)

    def forward(self, inp):
        return self.seq_layers(inp)


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.use_pre_layernorm = config.pre_layernorm
        self.use_post_layernorm = config.post_layernorm
        if self.use_pre_layernorm and self.use_post_layernorm:
            att_layer_post_layernorm = False
        else:
            att_layer_post_layernorm = self.use_post_layernorm

        self.attention = TransformerAttention(
            TransformerConfig(
                **update_copy_dict(
                    config.to_dict(),
                    full_attention_mask_and_cross_states=True,
                    post_layernorm=att_layer_post_layernorm,
                )
            )
        )

        self.intermediate = TransformerIntermediate(config)
        self.output = TransformerOutput(config)

    def forward(self, input):
        hidden_states, attention_mask = input
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_mask
