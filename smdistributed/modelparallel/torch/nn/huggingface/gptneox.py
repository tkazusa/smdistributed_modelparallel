# Third Party
import torch

# First Party
from smdistributed.modelparallel.torch.exceptions import HFGPTNeoxConfigError

try:
    from transformers.models.gpt_neox.modeling_gpt_neox import CausalLMOutputWithPast

    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False

max_seq_len_gptneox = None
num_attention_heads = None

if hf_transformers_available:

    def get_last_layer(state_dict):
        cur_last_layer = -1
        for key in state_dict:
            if key.startswith("gpt_neox.layers."):
                tokens = key.split(".")
                layer_idx = int(tokens[2])
                cur_last_layer = max(cur_last_layer, layer_idx)
        return cur_last_layer

    def get_hf_gptneox_transformer_lm_head_hooks():
        return (
            hf_gptneox_transformer_init_hook,
            hf_gptneox_transformer_forward_hook,
            hf_gptneox_transformer_return_hook,
        )

    def hf_gptneox_transformer_init_hook(config):
        if config.hidden_size % config.num_attention_heads != 0:
            raise HFGPTNeoxConfigError(
                f"Embedding size ({config.hidden_size}) must be divisible by the number of attention heads ({config.num_attention_heads}) for HuggingFace GPT-NEOX model."
            )

        if config.hidden_act not in ["gelu", "gelu_new", "relu"]:
            raise HFGPTNeoxConfigError("Only 'gelu_new', 'gelu', and 'relu' activations are supported.")

        global max_seq_len_gptneox
        global num_attention_heads
        max_seq_len_gptneox = config.max_position_embeddings
        num_attention_heads = config.num_attention_heads
        attention_head_size = config.hidden_size // config.num_attention_heads
        kwargs = {
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "attention_head_size": config.hidden_size // config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "rotary_pct": config.rotary_pct,
            "rotary_emb_base": config.rotary_emb_base, 
            "mask_value": -1e9,
            "use_positional_embedding": False,
            "parallel_attn_output": True,
            "use_lm_head_bias": False,
            "tie_input_output_embedding": config.tie_word_embeddings,
            "use_attn_dense_bias": True,
            "use_qkv_bias": True,
            "final_layernorm": True,
            "single_pre_layernorm": False,
            "vocab_size": config.vocab_size,
            "activation": "gelu" if config.hidden_act == "gelu_new" else config.hidden_act,
            "add_lm_head": True,
            "intermediate_size": config.intermediate_size
            if config.intermediate_size is not None
            else 4 * config.hidden_size,
            "attention_dropout_prob": 0,
            "hidden_dropout_prob": 0,
            "embedding_dropout_prob":0,
            "layernorm_epsilon": config.layer_norm_eps,
            "add_cross_attention": False,
            "initializer_range": config.initializer_range,
            "use_normal_initialization": True,
            "pre_layernorm": True,
            "post_layernorm": False,
            "causal_mask_size": config.max_position_embeddings,
            "num_positions": config.max_position_embeddings,
            "scale_attention_scores": True,
            "_scale_qkv_fan_out": True,
            "query_key_layer_scaling": False,
            "attention_in_fp32": False,
            "rotary_dim": int(attention_head_size * config.rotary_pct),
            "gpt_neox_type_rotary": True,
        }

        return (), kwargs

    def hf_gptneox_transformer_forward_hook(
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if (
            past_key_values
            or inputs_embeds
            or head_mask
            or use_cache
            or output_attentions
            or output_hidden_states
        ):
            raise HFGPTNeoxConfigError(
                f"past_key_values, inputs_embeds, head_mask, use_cache, output_attentions, and output_hidden_states arguments of HuggingFace GPTNeoXForCausalLM forward method are not supported."
            )
        if return_dict is not None and bool(return_dict) == False:
            raise HFGPTNeoxConfigError(
                "Setting False for the return_dict argument of HuggingFace GPTNeoXForCausalLM forward method is not supported."
            )

        if encoder_hidden_states is not None:
            input_tuple = (
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                labels,
            )
        else:
            input_tuple = (input_ids, attention_mask, token_type_ids, position_ids, labels)

        return (input_tuple,), {}

    def hf_gptneox_transformer_return_hook(outputs):
        return CausalLMOutputWithPast(
            loss=None if len(outputs) == 1 else outputs[0],
            logits=outputs[-1],
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def translate_hf_state_dict_to_smdistributed_gptneox(state_dict):
        """ For optimize == 'speed' only """

        translated_state_dict = {}
        global num_attention_heads

        last_layer = get_last_layer(state_dict)

        for name, param in state_dict.items():
            tokens = name.split(".")

            if name.endswith(".attention.bias") or name.endswith(".attention.masked_bias"):
                continue

            param_type = tokens[-1]

            if tokens[:2] == ["gpt_neox", "layers"]:
                layer_idx = tokens[2]
                block = tokens[3]

                if block == "mlp":
                    layer = tokens[4]
                    dense_idx = "1" if layer == "dense_h_to_4h" else "2"

                    translated_name = (
                        "transformer.seq_layers."
                        + layer_idx
                        + ".output.dense"
                        + dense_idx
                        + "."
                        + param_type
                    )
                    translated_state_dict[translated_name] = param

                elif block == "attention":
                    layer = tokens[4]
                    if layer == "dense":
                        translated_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.dense." + param_type
                        )
                        translated_state_dict[translated_name] = param
                    elif layer == "rotary_emb":
                        translated_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.rotary_emb." + param_type
                        )
                        translated_state_dict[translated_name] = param
                    elif layer == "query_key_value":
                        query_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.query." + param_type
                        )
                        key_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.key." + param_type
                        )
                        value_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.value." + param_type
                        )

                        attention_size = param.size(0) // 3
                        attention_head_size = attention_size // num_attention_heads
                        if param_type == "weight":
                            new_qkv_shape = (num_attention_heads, 3 * attention_head_size) + param.size()[1:]
                            param = param.view(*new_qkv_shape)
                            new_shape = (attention_size, attention_size)
                            query = param.narrow(1, 0, attention_head_size).reshape(*new_shape)
                            key = param.narrow(1, attention_head_size, attention_head_size).reshape(*new_shape)
                            value = param.narrow(1, 2 * attention_head_size, attention_head_size).reshape(*new_shape)
                        else:
                            new_qkv_shape = (num_attention_heads, 3 * attention_head_size)
                            param = param.view(*new_qkv_shape)
                            new_shape = (attention_size,)
                            query = param.narrow(1, 0, attention_head_size).reshape(*new_shape)
                            key = param.narrow(1, attention_head_size, attention_head_size).reshape(*new_shape)
                            value = param.narrow(1, 2 * attention_head_size, attention_head_size).reshape(*new_shape)

                        translated_state_dict[query_name] = query
                        translated_state_dict[key_name] = key
                        translated_state_dict[value_name] = value
                elif block in ["input_layernorm", "post_attention_layernorm"]:
                    att_vs_out = "attention" if block == "input_layernorm" else "output"
                    translated_name = (
                            "transformer.seq_layers."
                            + layer_idx
                            + "."
                            + att_vs_out
                            + ".pre_layernorm."
                            + param_type
                    )
                    translated_state_dict[translated_name] = param
            elif tokens[:2] == ["gpt_neox", "final_layer_norm"]:
                translated_name = "layernorm." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[:2] == ["gpt_neox", "embed_in"]:
                translated_name = "word_embedding." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[0] == "embed_out":
                translated_name = "lm_head." + param_type
                translated_state_dict[translated_name] = param

        return translated_state_dict

    # For backward compatibility
    translate_hf_gptneox_state_dict_to_smdistributed = lambda state_dict, max_seq_len: translate_hf_state_dict_to_smdistributed_gptneox(
        state_dict
    )

    def translate_state_dict_to_hf_gptneox(state_dict, max_seq_len=None):
        if max_seq_len is None:
            global max_seq_len_gptneox
            max_seq_len = max_seq_len_gptneox
        global num_attention_heads

        translated_state_dict = {}

        qkv = []

        for name, param in state_dict.items():
            tokens = name.split(".")

            # handle causal mask
            if tokens[-3:] == ["attention", "query", "weight"]:
                layer_id = tokens[-4]
                translated_state_dict["gpt_neox.layers." + layer_id + ".attention.bias"] = torch.tril(
                    torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)
                ).view(1, 1, max_seq_len, max_seq_len)
                translated_state_dict[
                    "gpt_neox.layers." + layer_id + ".attention.masked_bias"
                ] = torch.tensor(-1e9)

            if tokens[-2] in ["query", "key", "value"]:
                qkv.append((tokens, param))
                continue

            if tokens[0] == "lm_head":
                translated_name, translated_param = "embed_out.weight", param
            elif tokens[0] == "layernorm":
                translated_name, translated_param = get_layernorm_translation(tokens[1], param)
            elif tokens[0] == "word_embedding":
                translated_name, translated_param = get_word_embedding_translation(tokens[1], param)
            elif tokens[0] == "transformer":
                translated_name, translated_param = get_transformer_translation(
                    tokens[2:], param
                )
            else:
                raise HFGPTNeoxConfigError(f"Unknown token {tokens[0]}.")

            translated_state_dict[translated_name] = translated_param

        qkv_dict = handle_qkv(qkv, num_attention_heads)
        translated_state_dict.update(qkv_dict)

        return translated_state_dict

    def handle_qkv(qkv_list, num_attention_heads):
        qkv_dict = {}

        for idx, (tokens, param) in enumerate(qkv_list):
            # Clear the tensor reference
            # So at one time there is only 2 copies of the same layer weights at most
            qkv_list[idx] = None

            translated_name = "gpt_neox.layers." + tokens[2] + ".attention.query_key_value." + tokens[5]

            if tokens[4] == "query":
                slice_ind = 0
            elif tokens[4] == "key":
                slice_ind = 1
            elif tokens[4] == "value":
                slice_ind = 2
            else:
                raise HFGPTNeoxConfigError(f"Unrecognized token {tokens[2]}.")

            if translated_name not in qkv_dict:
                scaled_shape = list(param.shape)
                scaled_shape[0] *= 3
                qkv_dict[translated_name] = torch.empty(
                    *scaled_shape[:], dtype=param.dtype, device=param.device
                )
            slice_size = param.shape[0]
            weight_slice = qkv_dict[translated_name].narrow(
                0, slice_ind * slice_size, slice_size
            )
            weight_slice.copy_(param)

        for name, param in qkv_dict.items():
            tokens = name.split(".")
            if tokens[-1] == "weight":
                attention_size = param.size(0) // 3
                attention_head_size = attention_size // num_attention_heads
                new_param = torch.clone(param)

                q = new_param.narrow(0, 0, attention_size)
                k = new_param.narrow(0, attention_size, attention_size)
                v = new_param.narrow(0, 2 * attention_size, attention_size)

                for i in range(num_attention_heads):
                    start = i * attention_head_size * 3
                    qkv_start = i * attention_head_size
                    param.narrow(0, start, attention_head_size).copy_(
                        q.narrow(0, qkv_start, attention_head_size))
                    param.narrow(0, start+attention_head_size, attention_head_size).copy_(
                        k.narrow(0, qkv_start, attention_head_size))
                    param.narrow(0, start+attention_head_size*2, attention_head_size).copy_(
                        v.narrow(0, qkv_start, attention_head_size))

        return qkv_dict

    def get_layernorm_translation(name, param):
        return "gpt_neox.final_layer_norm." + name, param

    def get_word_embedding_translation(name, param):
        return "gpt_neox.embed_in." + name, param

    def get_transformer_translation(tokens, param):
        # tokens = [<layer_id>, <attention/output>, ...]

        prefix = "gpt_neox.layers." + tokens[0] + "."
        block = tokens[1]

        layer = tokens[2]
        param_type = tokens[3]

        if block == "output" and layer == "dense1":
            return prefix + "mlp.dense_h_to_4h." + param_type, param

        if block == "output" and layer == "dense2":
            return prefix + "mlp.dense_4h_to_h." + param_type, param

        if block == "attention" and layer == "dense":
            return prefix + "attention.dense." + param_type, param

        if block == "attention" and layer == "rotary_emb":
            return prefix + "attention.rotary_emb." + param_type, param

        if block == "attention" and layer == "pre_layernorm":
            return prefix + "input_layernorm." + param_type, param

        if block == "output" and layer == "pre_layernorm":
            return prefix + "post_attention_layernorm." + param_type, param

        raise HFGPTNeoxConfigError(f"Unknown tokens {tokens}.")
