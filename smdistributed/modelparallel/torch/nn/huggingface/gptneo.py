# Third Party
import torch

# First Party
from smdistributed.modelparallel.torch.exceptions import HFGPTNeoConfigError

try:
    from transformers.models.gpt_neo.modeling_gpt_neo import CausalLMOutputWithPast

    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False

max_seq_len_gptneo = None

if hf_transformers_available:

    def get_last_layer(state_dict):
        cur_last_layer = -1
        for key in state_dict:
            if key.startswith("transformer.h."):
                tokens = key.split(".")
                layer_idx = int(tokens[2])
                cur_last_layer = max(cur_last_layer, layer_idx)
        return cur_last_layer

    def get_hf_gptneo_transformer_lm_head_hooks():
        return (
            hf_gptneo_transformer_lm_head_init_hook,
            hf_gptneo_transformer_lm_head_forward_hook,
            hf_gptneo_transformer_lm_head_return_hook,
        )

    def hf_gptneo_transformer_lm_head_init_hook(config):
        if config.hidden_size % config.num_heads != 0:
            raise HFGPTNeoConfigError(
                f"Embedding size ({config.n_embd}) must be divisible by the number of attention heads ({config.num_heads}) for HuggingFace GPT-Neo model."
            )

        if config.activation_function not in ["gelu_new", "relu"]:
            raise HFGPTNeoConfigError("Only 'gelu_new' and 'relu' activations are supported.")

        attention_types = config.attention_types
        attention_layers_type = []
        for item in attention_types:
            for _ in range(item[1]):
                attention_layers_type.extend(item[0])
        if len(attention_layers_type) != config.num_layers:
            raise ValueError(
                f"The length of the attention layers {len(attention_layers_type)} must match with "
                f"the number of num_layers {config.num_layers}"
            )

        global max_seq_len_gptneo
        max_seq_len_gptneo = config.max_position_embeddings
        kwargs = {
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_heads,
            "attention_head_size": config.hidden_size // config.num_heads,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "activation": "gelu" if config.activation_function == "gelu_new" else "relu",
            "add_lm_head": True,
            "intermediate_size": config.intermediate_size
            if config.intermediate_size is not None
            else 4 * config.hidden_size,
            "attention_dropout_prob": config.attention_dropout,
            "hidden_dropout_prob": config.resid_dropout,
            "embedding_dropout_prob": config.embed_dropout,
            "layernorm_epsilon": config.layer_norm_epsilon,
            "add_cross_attention": False,
            "initializer_range": config.initializer_range,
            "attention_layers_type": attention_layers_type,
            "use_normal_initialization": True,
            "pre_layernorm": True,
            "post_layernorm": True,
            "causal_mask_size": config.max_position_embeddings,
            "num_positions": config.max_position_embeddings,
            "window_size": config.window_size,
            "_scale_qkv_fan_out": True,
            "scale_attention_scores": False,
            "attention_in_fp32": True,
            "use_qkv_bias": False,
            "mask_value": -1e9,
        }

        return (), kwargs

    def hf_gptneo_transformer_lm_head_forward_hook(
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
            raise HFGPTNeoConfigError(
                f"past_key_values, inputs_embeds, head_mask, use_cache, output_attentions, and output_hidden_states arguments of HuggingFace GPTNeoModelWithLMHead forward method are not supported."
            )
        if return_dict is not None and bool(return_dict) == False:
            raise HFGPTNeoConfigError(
                "Setting False for the return_dict argument of HuggingFace GPTNeoModelWithLMHead forward method is not supported."
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

    def hf_gptneo_transformer_lm_head_return_hook(outputs):
        return CausalLMOutputWithPast(
            loss=None if len(outputs) == 1 else outputs[0],
            logits=outputs[-1],
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def translate_hf_state_dict_to_smdistributed_gptneo(state_dict):
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

                    translated_state_dict[translated_name] = param

                elif block == "attn":
                    layer = tokens[5]
                    if layer == "out_proj":
                        translated_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.dense." + param_type
                        )
                        translated_param = param

                        translated_state_dict[translated_name] = translated_param
                    if layer == "k_proj":
                        translated_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.key." + param_type
                        )
                        translated_param = param

                        translated_state_dict[translated_name] = translated_param
                    if layer == "v_proj":
                        translated_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.value." + param_type
                        )
                        translated_param = param

                        translated_state_dict[translated_name] = translated_param
                    if layer == "q_proj":
                        translated_name = (
                            "transformer.seq_layers." + layer_idx + ".attention.query." + param_type
                        )
                        translated_param = param

                        translated_state_dict[translated_name] = translated_param

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
                translated_name = "lm_head." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[:2] == ["transformer", "wpe"]:
                translated_name = "position_embedding." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[0] == "lm_head":
                translated_state_dict[name] = param

        return translated_state_dict

    # For backward compatibility
    translate_hf_gptneo_state_dict_to_smdistributed = lambda state_dict, max_seq_len: translate_hf_state_dict_to_smdistributed_gptneo(
        state_dict
    )

    def translate_state_dict_to_hf_gptneo(state_dict, max_seq_len=None):
        if max_seq_len is None:
            global max_seq_len_gptneo
            max_seq_len = max_seq_len_gptneo

        translated_state_dict = {}

        for name, param in state_dict.items():
            tokens = name.split(".")

            # handle causal mask
            if tokens[-3:] == ["attention", "query", "weight"]:
                layer_id = tokens[-4]
                translated_state_dict[
                    "transformer.h." + layer_id + ".attn.attention.bias"
                ] = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)).view(
                    1, 1, max_seq_len, max_seq_len
                )
                translated_state_dict[
                    "transformer.h." + layer_id + ".attn.attention.masked_bias"
                ] = torch.tensor(-1e9)

            if tokens[0] == "lm_head":
                translated_name, translated_param = name, param
            elif tokens[0] == "word_embedding":
                translated_name, translated_param = get_word_embedding_translation(tokens[1], param)
            elif tokens[0] == "position_embedding":
                translated_name, translated_param = get_pos_embedding_translation(tokens[1], param)
            elif tokens[0] == "transformer":
                translated_name, translated_param = get_transformer_translation(tokens[2:], param)
            else:
                raise HFGPTNeoConfigError(f"Unknown token {tokens[0]}.")

            translated_state_dict[translated_name] = translated_param

        return translated_state_dict

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
            return prefix + "mlp.c_fc." + param_type, param

        if block == "output" and layer == "dense2":
            return prefix + "mlp.c_proj." + param_type, param

        if block == "output" and layer == "pre_layernorm":
            return prefix + "ln_2." + param_type, param

        if block == "output" and layer == "layernorm":
            # final layernorm
            return "transformer.ln_f." + param_type, param

        if block == "attention" and layer == "dense":
            return prefix + "attn.attention.out_proj." + param_type, param

        if block == "attention" and layer == "query":
            return prefix + "attn.attention.q_proj." + param_type, param

        if block == "attention" and layer == "key":
            return prefix + "attn.attention.k_proj." + param_type, param

        if block == "attention" and layer == "value":
            return prefix + "attn.attention.v_proj." + param_type, param

        if block == "attention" and layer == "pre_layernorm":
            return prefix + "ln_1." + param_type, param

        raise HFGPTNeoConfigError(f"Unknown tokens {tokens}.")
