# Third Party
import torch
# First Party
from smdistributed.modelparallel.torch.exceptions import HFViTConfigError
import smdistributed.modelparallel.torch as smp

try:
    from transformers.models.vit.modeling_vit import BaseModelOutput

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

    def get_hf_vit_encoder_hooks():
        return (
            hf_vit_encoder_init_hook,
            hf_vit_encoder_forward_hook,
            hf_vit_encoder_return_hook,
        )
    
    def hf_vit_encoder_init_hook(config):
        kwargs = {
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "attention_head_size": config.hidden_size // config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "activation": config.hidden_act,
            "hidden_dropout_prob": config.hidden_dropout_prob,
            "attention_dropout_prob": config.attention_probs_dropout_prob,
            "initializer_range": config.initializer_range,
            "layernorm_epsilon": config.layer_norm_eps,
            "scale_attention_scores": True,
            "pre_layernorm": True,
            "post_layernorm": False,
            "use_qkv_bias": config.qkv_bias,
        }

        return (), kwargs 

    def hf_vit_encoder_forward_hook(
        hidden_states,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if (
            not (all(mask is  None for mask in head_mask))
            or output_attentions
            or output_hidden_states
        ):
            raise HFViTConfigError(
                f"head_mask, output_attentions, and output_hidden_states arguments of HuggingFace VITEncoder forward method are not supported."
            )
        if return_dict is not None and bool(return_dict) == False:
            raise HFViTConfigError(
                "Setting False for the return_dict argument of HuggingFace VITEncoder forward method is not supported."
            )
        attention_mask = torch.zeros(hidden_states.shape[0], 1, 1, hidden_states.shape[1], dtype=torch.float32, device=torch.device("cuda", smp.local_rank()))
        if smp.state.cfg.fp16:
            attention_mask = attention_mask.to(torch.float16)

        input_tuple = (
            hidden_states,
            attention_mask,
        )

        return (input_tuple,), {}


    def hf_vit_encoder_return_hook(output):
        return BaseModelOutput(
            last_hidden_state=output[0],
            hidden_states=None,
            attentions=None,
        )

    def translate_hf_state_dict_to_smdistributed_vit(state_dict):
        translated_state_dict = {}
        for name, param in state_dict.items():
            tokens = name.split(".")
            if len(tokens) >= 2 and tokens[1] == "embeddings":
                translated_name = name
            elif len(tokens) < 4 and tokens[1] == "layernorm":
                translated_name = name
            elif len(tokens) > 0 and tokens[0] == "classifier":
                translated_name = name
            elif len(tokens) >= 4 and tokens[-4] == "attention":
                layer_idx = tokens[-5]
                block = tokens[-3]
                layer = tokens[-2]
                param_type = tokens[-1]

                prefix = "vit.encoder.seq_layers." + layer_idx + ".attention."
                if block == "attention":
                    translated_name = prefix + layer + "." + param_type
                elif block == "output":
                    if layer == "dense":
                        translated_name = prefix + layer + "." + param_type
            elif len(tokens) < 7 and tokens[-2]== "layernorm_before" :
                layer_idx = tokens[-3]
                param_type = tokens[-1]
                prefix = "vit.encoder.seq_layers." + layer_idx  + "."
                translated_name = prefix + "attention.pre_layernorm." + param_type

            elif len(tokens) < 7 and tokens[-2]== "layernorm_after":
                layer_idx = tokens[-3]
                param_type = tokens[-1]
                prefix = "vit.encoder.seq_layers." + layer_idx  + "."
                translated_name = prefix + "output.pre_layernorm." + param_type
            else:
                layer_idx = tokens[-4]
                block = tokens[-3]
                layer = tokens[-2]
                param_type = tokens[-1]

                prefix = "vit.encoder.seq_layers." + layer_idx + "."
                if block == "output":
                    if layer == "dense":
                        translated_name = prefix + "output.dense2." + param_type
                elif block == "intermediate":
                    translated_name = prefix + "output.dense1." + param_type

            translated_state_dict[translated_name] = param

        return translated_state_dict

    # For backward compatibility
    translate_hf_state_dict_to_smdistributed = translate_hf_state_dict_to_smdistributed_vit

    def translate_state_dict_to_hf_vit(state_dict):
        translated_state_dict = {}

        for name, param in state_dict.items():
            tokens = name.split(".")

            if len(tokens) >= 2 and tokens[1] == "embeddings":
                translated_name = name
            elif len(tokens) > 0 and tokens[1] == "layernorm":
                translated_name = name
            elif len(tokens) > 0 and tokens[0] == "classifier":
                translated_name = name
            else:
                layer_idx = tokens[-4]
                block = tokens[-3]
                layer = tokens[-2]
                param_type = tokens[-1]
                

                prefix = "vit.encoder.layer." + layer_idx + "."

                if block == "attention":
                    if layer in ["query", "key", "value"]:
                        translated_name = prefix + "attention.attention." + layer + "." + param_type
                    elif layer == "dense":
                        translated_name = prefix + "attention.output." + layer + "." + param_type
                    elif layer == "pre_layernorm":
                        translated_name = prefix + "layernorm_before" + "." + param_type
                else:
                    if layer == "dense1":
                        translated_name = prefix + "intermediate.dense." + param_type
                    elif layer == "dense2":
                        translated_name = prefix + "output.dense." + param_type
                    elif layer == "pre_layernorm":
                        translated_name = prefix + "layernorm_after." + param_type

            translated_state_dict[translated_name] = param

        return translated_state_dict
