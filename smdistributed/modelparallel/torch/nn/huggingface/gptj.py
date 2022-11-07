# Third Party
import torch

# First Party
from smdistributed.modelparallel.torch.exceptions import HFGPTJConfigError

try:
    from transformers.models.gptj.modeling_gptj import CausalLMOutputWithPast

    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False

max_seq_len_gptj = None

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
            raise HFGPTJConfigError(
                f"Embedding size ({config.n_embd}) must be divisible by the number of attention heads ({config.n_head}) for HuggingFace GPT-J model."
            )

        if config.activation_function not in ["gelu_new", "relu"]:
            raise HFGPTJConfigError("Only 'gelu_new' and 'relu' activations are supported.")

        global max_seq_len_gptj
        max_seq_len_gptj = config.n_positions
        kwargs = {
            "num_layers": config.n_layer,
            "num_attention_heads": config.n_head,
            "attention_head_size": config.n_embd // config.n_head,
            "hidden_size": config.n_embd,
            "rotary_dim": config.rotary_dim,
            "mask_value": -1e9,
            "use_positional_embedding": False,
            "parallel_attn_output": True,
            "use_lm_head_bias": True,
            "tie_input_output_embedding": config.tie_word_embeddings,
            "use_attn_dense_bias": False,
            "use_qkv_bias": False,
            "final_layernorm": True,
            "single_pre_layernorm": True,
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
            "pre_layernorm": False,
            "post_layernorm": False,
            "causal_mask_size": config.n_positions,
            "num_positions": config.n_positions,
            "scale_attention_scores": config.scale_attn_weights,
            "_scale_qkv_fan_out": True,
            "query_key_layer_scaling": False,
            "attention_in_fp32": False,
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
            raise HFGPTJConfigError(
                f"past_key_values, inputs_embeds, head_mask, use_cache, output_attentions, and output_hidden_states arguments of HuggingFace GPTJForCausalLM forward method are not supported."
            )
        if return_dict is not None and bool(return_dict) == False:
            raise HFGPTJConfigError(
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

    def translate_hf_state_dict_to_smdistributed_gptj(state_dict):
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
                    dense_idx = "1" if layer == "fc_in" else "2"

                    translated_name = (
                        "transformer.seq_layers."
                        + layer_idx
                        + ".output.dense"
                        + dense_idx
                        + "."
                        + param_type
                    )
                    translated_param = param

                    translated_state_dict[translated_name] = translated_param

                elif block == "attn":
                    layer = tokens[4]
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
                elif block == "ln_1":
                    translated_name = (
                        "transformer.seq_layers." + layer_idx + ".pre_layernorm." + param_type
                    )
                    translated_state_dict[translated_name] = param
            elif tokens[:2] == ["transformer", "ln_f"]:
                translated_name = "layernorm." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[:2] == ["transformer", "wte"]:
                translated_name = "word_embedding." + param_type
                translated_state_dict[translated_name] = param
            elif tokens[0] == "lm_head":
                translated_state_dict[name] = param

        return translated_state_dict

    # For backward compatibility
    translate_hf_gptj_state_dict_to_smdistributed = lambda state_dict, max_seq_len: translate_hf_state_dict_to_smdistributed_gptj(
        state_dict
    )

    def translate_state_dict_to_hf_gptj(state_dict, max_seq_len=None):
        if max_seq_len is None:
            global max_seq_len_gptj
            max_seq_len = max_seq_len_gptj

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
                ] = torch.tensor(-1e9)

            if tokens[0] == "lm_head":
                translated_name, translated_param = name, param
            elif tokens[0] == "layernorm":
                translated_name, translated_param = get_layernorm_translation(tokens[1], param)
            elif tokens[0] == "word_embedding":
                translated_name, translated_param = get_word_embedding_translation(tokens[1], param)
            elif tokens[0] == "transformer":
                if tokens[3] == "output" and tokens[4] in ["pre_layernorm", "layernorm"]:
                    pass
                else:
                    translated_name, translated_param = get_transformer_translation(
                        tokens[2:], param
                    )
            else:
                raise HFGPTJConfigError(f"Unknown token {tokens[0]}.")

            translated_state_dict[translated_name] = translated_param

        return translated_state_dict

    def get_layernorm_translation(name, param):
        return "transformer.ln_f." + name, param

    def get_word_embedding_translation(name, param):
        return "transformer.wte." + name, param

    def get_transformer_translation(tokens, param):
        # tokens = [<layer_id>, <attention/output>, ...]

        prefix = "transformer.h." + tokens[0] + "."
        block = tokens[1]

        if block == "pre_layernorm":
            return prefix + "ln_1." + tokens[2], param

        layer = tokens[2]
        param_type = tokens[3]

        if block == "output" and layer == "dense1":
            return prefix + "mlp.fc_in." + param_type, param

        if block == "output" and layer == "dense2":
            return prefix + "mlp.fc_out." + param_type, param

        if block == "attention" and layer == "dense":
            return prefix + "attn.out_proj." + param_type, param

        if block == "attention" and layer == "query":
            return prefix + "attn.q_proj." + param_type, param

        if block == "attention" and layer == "key":
            return prefix + "attn.k_proj." + param_type, param

        if block == "attention" and layer == "value":
            return prefix + "attn.v_proj." + param_type, param

        raise HFGPTJConfigError(f"Unknown tokens {tokens}.")
