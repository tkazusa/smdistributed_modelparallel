# Third Party
import torch

# First Party
from smdistributed.modelparallel.torch.exceptions import HFGPT2ConfigError

try:
    from transformers.models.gpt2.modeling_gpt2 import CausalLMOutputWithCrossAttentions

    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False

max_seq_len_gpt2 = None

if hf_transformers_available:

    def get_last_layer(state_dict):
        cur_last_layer = -1
        for key in state_dict:
            if key.startswith("transformer.h."):
                tokens = key.split(".")
                layer_idx = int(tokens[2])
                cur_last_layer = max(cur_last_layer, layer_idx)
        return cur_last_layer

    def get_hf_gpt2_transformer_lm_head_hooks():
        return (
            hf_gpt2_transformer_lm_head_init_hook,
            hf_gpt2_transformer_lm_head_forward_hook,
            hf_gpt2_transformer_lm_head_return_hook,
        )

    def get_hf_gpt2_transformer_layer_hooks():
        return (
            hf_gpt2_transformer_layer_init_hook,
            hf_gpt2_transformer_layer_forward_hook,
            hf_gpt2_transformer_layer_return_hook,
        )

    def hf_gpt2_transformer_lm_head_init_hook(config):
        if config.n_embd % config.n_head != 0:
            raise HFGPT2ConfigError(
                f"Embedding size ({config.n_embd}) must be divisible by the number of attention heads ({config.n_head}) for HuggingFace GPT-2 model."
            )

        if config.activation_function not in ["gelu_new", "relu"]:
            raise HFGPT2ConfigError("Only 'gelu_new' and 'relu' activations are supported.")

        global max_seq_len_gpt2
        max_seq_len_gpt2 = config.n_positions
        kwargs = {
            "num_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "attention_head_size": config.n_embd // config.n_head,
            "hidden_size": config.n_embd,
            "vocab_size": config.vocab_size,
            "activation": "gelu" if config.activation_function == "gelu_new" else "relu",
            "add_lm_head": True,
            "intermediate_size": config.n_inner
            if config.n_inner is not None
            else 4 * config.n_embd,
            "attention_dropout_prob": config.attn_pdrop,
            "hidden_dropout_prob": config.resid_pdrop,
            "embedding_dropout_prob": config.embd_pdrop,
            "layernorm_epsilon": config.layer_norm_epsilon,
            "add_cross_attention": False,
            "initializer_range": config.initializer_range,
            "use_normal_initialization": True,
            "pre_layernorm": True,
            "post_layernorm": True,
            "causal_mask_size": config.n_positions,
            "num_positions": config.n_positions,
            "_scale_qkv_fan_out": True,
            "scale_attention_scores": config.scale_attn_weights,
            "scale_attn_by_layer_idx": config.scale_attn_by_inverse_layer_idx,
            "query_key_layer_scaling": config.reorder_and_upcast_attn,
            "attention_in_fp32": config.reorder_and_upcast_attn,
        }

        return (), kwargs

    def hf_gpt2_transformer_lm_head_forward_hook(
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
            raise HFGPT2ConfigError(
                f"past_key_values, inputs_embeds, head_mask, use_cache, output_attentions, and output_hidden_states arguments of HuggingFace GPT2ModelWithLMHead forward method are not supported."
            )
        if return_dict is not None and bool(return_dict) == False:
            raise HFGPT2ConfigError(
                "Setting False for the return_dict argument of HuggingFace GPT2ModelWithLMHead forward method is not supported."
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

    def hf_gpt2_transformer_lm_head_return_hook(outputs):
        return CausalLMOutputWithCrossAttentions(
            loss=None if len(outputs) == 1 else outputs[0],
            logits=outputs[-1],
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def hf_gpt2_transformer_layer_init_hook(config, layer_idx=None):

        if config.n_embd % config.n_head != 0:
            raise HFGPT2ConfigError(
                f"Embedding size ({config.n_embd}) must be divisible by the number of attention heads ({config.n_head}) for HuggingFace GPT-2 model."
            )
        if not config.use_return_dict:
            raise HFGPT2ConfigError(
                "return_dict argument of HuggingFace GPT-2 configuration must be True."
            )

        global max_seq_len_gpt2
        max_seq_len_gpt2 = config.n_positions
        kwargs = {
            "num_attention_heads": config.n_head,
            "attention_head_size": config.n_embd // config.n_head,
            "hidden_size": config.n_embd,
            "intermediate_size": config.n_inner
            if config.n_inner is not None
            else 4 * config.n_embd,
            "attention_dropout_prob": config.attn_pdrop,
            "hidden_dropout_prob": config.resid_pdrop,
            "layernorm_epsilon": config.layer_norm_epsilon,
            "add_cross_attention": False,
            "initializer_range": config.initializer_range,
            "use_normal_initialization": True,
            "pre_layernorm": True,
            "post_layernorm": False,
            "causal_mask_size": config.n_positions,
            "full_attention_mask_and_cross_states": False,
            "_scale_qkv_fan_out": True,
        }

        return (), kwargs

    def hf_gpt2_transformer_layer_forward_hook(
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if head_mask or use_cache or output_attentions:
            raise HFGPT2ConfigError(
                f"head_mask, use_cache, and output_attentions arguments of GPT-2 Block forward method are not supported."
            )

        if encoder_hidden_states is not None:
            input_tuple = (
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                encoder_attention_mask,
            )
        else:
            input_tuple = (hidden_states, attention_mask)

        return (input_tuple,), {}

    def hf_gpt2_transformer_layer_return_hook(outputs):
        return (outputs[0], None)

    def translate_hf_state_dict_to_smdistributed_gpt2_layer(state_dict):
        translated_state_dict = {}

        for name, param in state_dict.items():
            tokens = name.split(".")

            if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
                continue

            param_type = tokens[-1]

            if tokens[:2] == ["transformer", "h"]:
                layer_idx = tokens[2]
                block = tokens[3]

                if block == "mlp":
                    layer = tokens[4]
                    dense_idx = "1" if layer == "c_fc" else "2"

                    translated_name = (
                        "transformer.h."
                        + layer_idx
                        + ".output.dense"
                        + dense_idx
                        + "."
                        + param_type
                    )
                    if param_type == "weight":
                        translated_param = param.t()
                    else:
                        translated_param = param

                    translated_state_dict[translated_name] = translated_param

                elif block == "attn":
                    layer = tokens[4]
                    if layer == "c_proj":
                        translated_name = (
                            "transformer.h." + layer_idx + ".attention.dense." + param_type
                        )
                        if param_type == "weight":
                            translated_param = param.t()
                        else:
                            translated_param = param

                        translated_state_dict[translated_name] = translated_param
                    elif layer == "c_attn":
                        query_name = "transformer.h." + layer_idx + ".attention.query." + param_type
                        key_name = "transformer.h." + layer_idx + ".attention.key." + param_type
                        value_name = "transformer.h." + layer_idx + ".attention.value." + param_type

                        if param_type == "weight":
                            attention_size = param.size(1) // 3
                            mod_q_param = param.narrow(1, 0, attention_size).t()
                            mod_k_param = param.narrow(1, attention_size, attention_size).t()
                            mod_v_param = param.narrow(1, 2 * attention_size, attention_size).t()
                        else:
                            attention_size = param.size(0) // 3
                            mod_q_param = param.narrow(0, 0, attention_size)
                            mod_k_param = param.narrow(0, attention_size, attention_size)
                            mod_v_param = param.narrow(0, 2 * attention_size, attention_size)

                        translated_state_dict[query_name] = mod_q_param
                        translated_state_dict[key_name] = mod_k_param
                        translated_state_dict[value_name] = mod_v_param
                elif block in ["ln_1", "ln_2"]:
                    att_vs_out = "attention" if block == "ln_1" else "output"
                    translated_name = (
                        "transformer.h."
                        + layer_idx
                        + "."
                        + att_vs_out
                        + ".pre_layernorm."
                        + param_type
                    )
                    translated_state_dict[translated_name] = param
            elif tokens[0] == "lm_head" or tokens[1] in ("ln_f", "wte", "wpe"):
                translated_state_dict[name] = param

        return translated_state_dict

    # For backward compatibility
    translate_hf_block_state_dict_to_smdistributed = lambda state_dict, max_seq_len: translate_hf_state_dict_to_smdistributed_gpt2_layer(
        state_dict
    )

    def translate_state_dict_to_hf_gpt2_layer(state_dict, max_seq_len=None):
        if max_seq_len is None:
            global max_seq_len_gpt2
            max_seq_len = max_seq_len_gpt2

        translated_state_dict = {}

        qkv = []

        for name, param in state_dict.items():
            # Clear the tensor reference
            state_dict[name] = None

            tokens = name.split(".")

            # handle causal mask
            if tokens[-3:] == ["attention", "query", "weight"]:
                layer_id = tokens[-4]
                translated_state_dict["transformer.h." + layer_id + ".attn.bias"] = torch.tril(
                    torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)
                ).view(1, 1, max_seq_len, max_seq_len)
                translated_state_dict[
                    "transformer.h." + layer_id + ".attn.masked_bias"
                ] = torch.tensor(-1e4)

            if tokens[-2] in ["query", "key", "value"]:
                qkv.append((tokens, param))
                continue

            if tokens[0] == "lm_head":
                translated_name, translated_param = name, param
            elif tokens[0] == "transformer":
                if tokens[1] in ("ln_f", "wte", "wpe"):
                    translated_name, translated_param = name, param
                else:
                    translated_name, translated_param = get_transformer_translation(
                        tokens[2:], param
                    )
            else:
                raise HFGPT2ConfigError(f"Unknown token {tokens[0]}.")

            translated_state_dict[translated_name] = translated_param

        qkv_dict = handle_qkv(qkv)
        translated_state_dict.update(qkv_dict)

        return translated_state_dict

    # For backward compatibility
    translate_layer_state_dict_to_hf_gpt2_block = lambda state_dict, max_seq_len: translate_state_dict_to_hf_gpt2_layer(
        state_dict, max_seq_len=max_seq_len
    )

    def translate_hf_state_dict_to_smdistributed_gpt2(state_dict):
        """ For optimize == 'speed' only """

        translated_state_dict = {}

        last_layer = get_last_layer(state_dict)

        for name, param in state_dict.items():
            tokens = name.split(".")

            if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
                continue

            param_type = tokens[-1]

            if tokens[:2] == ["transformer", "h"]:
                layer_idx = tokens[2]
                block = tokens[3]

                if block == "mlp":
                    layer = tokens[4]
                    dense_idx = "1" if layer == "c_fc" else "2"

                    translated_name = (
                        "transformer.seq_layers."
                        + layer_idx
                        + ".output.dense"
                        + dense_idx
                        + "."
                        + param_type
                    )
                    if param_type == "weight":
                        translated_param = param.t()
                    else:
                        translated_param = param

                    translated_state_dict[translated_name] = translated_param

                elif block == "attn":
                    layer = tokens[4]
                    if layer == "c_proj":
                        translated_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.dense." + param_type
                        )
                        if param_type == "weight":
                            translated_param = param.t()
                        else:
                            translated_param = param

                        translated_state_dict[translated_name] = translated_param
                    elif layer == "c_attn":
                        query_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.query." + param_type
                        )
                        key_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.key." + param_type
                        )
                        value_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.value." + param_type
                        )

                        if param_type == "weight":
                            attention_size = param.size(1) // 3
                            mod_q_param = param.narrow(1, 0, attention_size).t()
                            mod_k_param = param.narrow(1, attention_size, attention_size).t()
                            mod_v_param = param.narrow(1, 2 * attention_size, attention_size).t()
                        else:
                            attention_size = param.size(0) // 3
                            mod_q_param = param.narrow(0, 0, attention_size)
                            mod_k_param = param.narrow(0, attention_size, attention_size)
                            mod_v_param = param.narrow(0, 2 * attention_size, attention_size)

                        translated_state_dict[query_name] = mod_q_param
                        translated_state_dict[key_name] = mod_k_param
                        translated_state_dict[value_name] = mod_v_param
                elif block in ["ln_1", "ln_2"]:
                    att_vs_out = "attention" if block == "ln_1" else "output"
                    translated_name = (
                        "transformer.seq_layers."
                        + layer_idx
                        + "."
                        + att_vs_out
                        + ".pre_layernorm."
                        + param_type
                    )
                    translated_state_dict[translated_name] = param
            elif tokens[:2] == ["transformer", "ln_f"]:
                translated_name = (
                    "transformer.seq_layers."
                    + str(last_layer)
                    + "."
                    + "output.layernorm."
                    + param_type
                )
                translated_state_dict[translated_name] = param
            elif tokens[:2] == ["transformer", "wte"]:
                translated_name = "word_embedding." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[:2] == ["transformer", "wpe"]:
                translated_name = "position_embedding." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[0] == "lm_head":
                translated_state_dict[name] = param

        return translated_state_dict

    # For backward compatibility
    translate_hf_state_dict_to_smdistributed = lambda state_dict, max_seq_len: translate_hf_state_dict_to_smdistributed_gpt2(
        state_dict
    )

    def translate_state_dict_to_hf_gpt2(state_dict, max_seq_len=None):
        if max_seq_len is None:
            global max_seq_len_gpt2
            max_seq_len = max_seq_len_gpt2

        translated_state_dict = {}

        qkv = []

        for name, param in state_dict.items():
            # Clear the tensor reference
            state_dict[name] = None

            tokens = name.split(".")

            # handle causal mask
            if tokens[-3:] == ["attention", "query", "weight"]:
                layer_id = tokens[-4]
                translated_state_dict["transformer.h." + layer_id + ".attn.bias"] = torch.tril(
                    torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)
                ).view(1, 1, max_seq_len, max_seq_len)
                translated_state_dict[
                    "transformer.h." + layer_id + ".attn.masked_bias"
                ] = torch.tensor(-1e4)

            if tokens[-2] in ["query", "key", "value"]:
                qkv.append((tokens, param))
                continue

            if tokens[0] == "lm_head":
                translated_name, translated_param = name, param
            elif tokens[0] == "word_embedding":
                translated_name, translated_param = get_word_embedding_translation(tokens[1], param)
            elif tokens[0] == "position_embedding":
                translated_name, translated_param = get_pos_embedding_translation(tokens[1], param)
            elif tokens[0] == "transformer":
                translated_name, translated_param = get_transformer_translation(tokens[2:], param)
            else:
                raise HFGPT2ConfigError(f"Unknown token {tokens[0]}.")

            translated_state_dict[translated_name] = translated_param

        qkv_dict = handle_qkv(qkv)
        translated_state_dict.update(qkv_dict)

        return translated_state_dict

    def handle_qkv(qkv_list):
        qkv_dict = {}

        for idx, (tokens, param) in enumerate(qkv_list):
            # Clear the tensor reference
            # So at one time there is only 2 copies of the same layer weights at most
            qkv_list[idx] = None

            translated_name = "transformer.h." + tokens[2] + ".attn.c_attn." + tokens[5]

            if tokens[4] == "query":
                slice_ind = 0
            elif tokens[4] == "key":
                slice_ind = 1
            elif tokens[4] == "value":
                slice_ind = 2
            else:
                raise HFGPT2ConfigError(f"Unrecognized token {tokens[2]}.")

            if translated_name not in qkv_dict:
                scaled_shape = list(param.shape)
                scaled_shape[0] *= 3
                qkv_dict[translated_name] = torch.empty(
                    *scaled_shape[::-1], dtype=param.dtype, device=param.device
                )

            slice_size = param.shape[0]
            if tokens[5] == "weight":
                weight_slice = qkv_dict[translated_name].narrow(
                    1, slice_ind * slice_size, slice_size
                )
                weight_slice.copy_(param.t())
            else:
                weight_slice = qkv_dict[translated_name].narrow(
                    0, slice_ind * slice_size, slice_size
                )
                weight_slice.copy_(param)

        return qkv_dict

    def get_pos_embedding_translation(name, param):
        return "transformer.wpe." + name, param

    def get_word_embedding_translation(name, param):
        return "transformer.wte." + name, param

    def get_transformer_translation(tokens, param):
        # tokens = [<layer_id>, <attention/output>, ...]

        block = tokens[1]
        layer = tokens[2]
        param_type = tokens[3]
        prefix = "transformer.h." + tokens[0] + "."

        if block == "output" and layer == "dense1":
            return prefix + "mlp.c_fc." + param_type, param.t()

        if block == "output" and layer == "dense2":
            return prefix + "mlp.c_proj." + param_type, param.t()

        if block == "output" and layer == "pre_layernorm":
            return prefix + "ln_2." + param_type, param

        if block == "output" and layer == "layernorm":
            # final layernorm
            return "transformer.ln_f." + param_type, param

        if block == "attention" and layer == "dense":
            return prefix + "attn.c_proj." + param_type, param.t()

        if block == "attention" and layer == "pre_layernorm":
            return prefix + "ln_1." + param_type, param

        raise HFGPT2ConfigError(f"Unknown tokens {tokens}.")
