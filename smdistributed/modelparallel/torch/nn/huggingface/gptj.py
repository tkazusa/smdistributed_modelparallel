# Third Party
import torch

try:
    from transformers.models.gptj.modeling_gptj import CausalLMOutputWithPast
    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False

if hf_transformers_available:

    def get_last_layer(state_dict):
        cur_last_layer = -1
        for key in state_dict:
            if key.startswith("transformer.h."):
                tokens = key.split(".")
                layer_idx = int(tokens[2])
                cur_last_layer = max(cur_last_layer, layer_idx)
        return cur_last_layer

    def get_hf_gptj_transformer_hooks():
        return (
            hf_gptj_transformer_init_hook,
            hf_gptj_transformer_forward_hook,
            hf_gptj_transformer_return_hook,
        )

    def hf_gptj_transformer_init_hook(config):
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"Embedding size ({config.n_embd}) must be divisible by the number of attention heads ({config.n_head}) for HuggingFace GPT-J model."
            )

        if config.activation_function not in ["gelu_new", "relu"]:
            raise ValueError("Only 'gelu_new' and 'relu' activations are supported.")

        kwargs = {
            "num_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "attention_head_size": config.n_embd // config.n_head,
            "hidden_size": config.n_embd,
            "rotary_dim": config.rotary_dim,
            "vocab_size": config.vocab_size,
            "activation": "gelu" if config.activation_function == "gelu_new" else "relu",
            "add_lm_head": True,
            "intermediate_size": config.n_inner
            if config.n_inner is not None
            else 4 * config.n_embd,
            "attention_dropout_prob": config.attn_pdrop,
            "hidden_dropout_prob": config.resid_pdrop,
            "layernorm_epsilon": config.layer_norm_epsilon,
            "add_cross_attention": config.add_cross_attention,
            "initializer_range": config.initializer_range,
            "use_normal_initialization": True,
            "pre_layernorm": True,
            "post_layernorm": True,
            "causal_mask_size": config.n_positions,
            "num_positions": config.n_positions,
            "_scale_qkv_fan_out": True,
            "query_key_layer_scaling": False,
            "attention_in_fp32":False,
        }

        return (), kwargs

    def hf_gptj_transformer_forward_hook(
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
            raise ValueError(
                f"past_key_values, inputs_embeds, head_mask, use_cache, output_attentions, and output_hidden_states arguments of HuggingFace GPTJForCausalLM forward method are not supported."
            )
        if return_dict is not None and bool(return_dict) == False:
            raise ValueError(
                "Setting False for the return_dict argument of HuggingFace GPTJForCausalLM forward method is not supported."
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

    def hf_gptj_transformer_return_hook(outputs):
        return CausalLMOutputWithPast(
            loss=None if len(outputs) == 1 else outputs[0],
            logits=outputs[-1],
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def translate_hf_state_dict_to_smdistributed(state_dict, max_seq_len):
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
            elif tokens[0] == "lm_head":
                translated_state_dict[name] = param

        return translated_state_dict

    def translate_state_dict_to_hf_gptj(state_dict, max_seq_len):
        translated_state_dict = {}

        qkv = []

        for name, param in state_dict.items():
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
            elif tokens[0] == "transformer":
                translated_name, translated_param = get_transformer_translation(tokens[2:], param)
            else:
                raise ValueError(f"Unknown token {tokens[0]}.")

            translated_state_dict[translated_name] = translated_param

        qkv_dict = handle_qkv(qkv)
        translated_state_dict.update(qkv_dict)

        return translated_state_dict

    def handle_qkv(qkv_list):
        qkv_dict = {}

        for tokens, param in qkv_list:
            translated_name = "transformer.h." + tokens[2] + ".attn.c_attn." + tokens[5]

            if tokens[4] == "query":
                slice_ind = 0
            elif tokens[4] == "key":
                slice_ind = 1
            elif tokens[4] == "value":
                slice_ind = 2
            else:
                raise ValueError(f"Unrecognized token {tokens[2]}.")

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

        raise ValueError(f"Unknown tokens {tokens}.")
