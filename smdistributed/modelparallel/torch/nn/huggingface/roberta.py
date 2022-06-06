# Standard Library
from collections.abc import Iterable

try:
    from transformers.models.roberta.modeling_roberta import (
        BaseModelOutputWithPastAndCrossAttentions,
    )

    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False

if hf_transformers_available:

    def translate_hf_state_dict_to_smdistributed(state_dict):
        translated_state_dict = {}

        for name, param in state_dict.items():
            tokens = name.split(".")

            if len(tokens) >= 2 and tokens[1] == "embeddings":
                translated_name = name
            elif len(tokens) > 0 and tokens[0] == "lm_head":
                translated_name = name
            elif len(tokens) >= 4 and tokens[-4] == "attention":
                layer_idx = tokens[-5]
                block = tokens[-3]
                layer = tokens[-2]
                param_type = tokens[-1]

                prefix = "roberta.encoder.seq_layers." + layer_idx + ".attention."
                if block == "self":
                    translated_name = prefix + layer + "." + param_type
                elif block == "output":
                    if layer == "dense":
                        translated_name = prefix + layer + "." + param_type
                    if layer == "LayerNorm":
                        translated_name = prefix + "layernorm." + param_type
            else:
                layer_idx = tokens[-4]
                block = tokens[-3]
                layer = tokens[-2]
                param_type = tokens[-1]

                prefix = "roberta.encoder.seq_layers." + layer_idx + "."
                if block == "output":
                    if layer == "dense":
                        translated_name = prefix + "output.dense2." + param_type
                    elif layer == "LayerNorm":
                        translated_name = prefix + "output.layernorm." + param_type
                elif block == "intermediate":
                    translated_name = prefix + "output.dense1." + param_type

            translated_state_dict[translated_name] = param

        return translated_state_dict

    def translate_state_dict_to_hf_roberta(state_dict):
        translated_state_dict = {}

        for name, param in state_dict.items():
            tokens = name.split(".")

            if len(tokens) >= 2 and tokens[1] == "embeddings":
                translated_name = name
            elif len(tokens) > 0 and tokens[0] == "lm_head":
                translated_name = name
            else:
                layer_idx = tokens[-4]
                block = tokens[-3]
                layer = tokens[-2]
                param_type = tokens[-1]

                prefix = "roberta.encoder.layer." + layer_idx + "."

                if block == "attention":
                    if layer in ["query", "key", "value"]:
                        translated_name = prefix + "attention.self." + layer + "." + param_type
                    elif layer == "dense":
                        translated_name = prefix + "attention.output." + layer + "." + param_type
                    elif layer == "layernorm":
                        translated_name = prefix + "attention.output.LayerNorm" + "." + param_type
                else:
                    if layer == "dense1":
                        translated_name = prefix + "intermediate.dense." + param_type
                    elif layer == "dense2":
                        translated_name = prefix + "output.dense." + param_type
                    elif layer == "layernorm":
                        translated_name = prefix + "output.LayerNorm." + param_type

            translated_state_dict[translated_name] = param

        return translated_state_dict

    def get_hf_roberta_transformer_hooks():
        return (
            hf_roberta_transformer_init_hook,
            hf_roberta_transformer_forward_hook,
            hf_roberta_transformer_return_hook,
        )

    def hf_roberta_transformer_forward_hook(
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if head_mask is not None:
            if not isinstance(head_mask, Iterable) or any([m is not None for m in head_mask]):
                raise ValueError(
                    "head_mask argument of HuggingFace RobertaEncoder is not supported."
                )

        if past_key_values is not None:
            raise ValueError(
                "past_key_values argument of HuggingFace RobertaEncoder is not supported."
            )

        if use_cache:
            raise ValueError("use_cache argument of HuggingFace RobertaEncoder is not supported.")

        if output_attentions or output_hidden_states:
            raise ValueError(
                "output_attentions and output_hidden_states arguments of HuggingFace RobertaEncoder are not supported."
            )

        if return_dict is not None and bool(return_dict) == False:
            raise ValueError(
                "Setting False for the return_dict argument of HuggingFace RobertaEncoder forward method is not supported."
            )

        if attention_mask is None:
            raise ValueError("attention_mask is a required argument of DistributedTransformer.")

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

    def hf_roberta_transformer_return_hook(output):
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output[0],
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def hf_roberta_transformer_init_hook(config):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({config.hidden_size}) must be divisible by the number of attention heads ({config.num_attention_heads}) for HuggingFace RoBERTa model."
            )

        if config.position_embedding_type != "absolute":
            raise ValueError(
                "Only position_embedding_type=='absolute' is supported for HuggingFace RoBERTa model."
            )

        kwargs = {
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "attention_head_size": config.hidden_size // config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "attention_dropout_prob": config.attention_probs_dropout_prob,
            "hidden_dropout_prob": config.hidden_dropout_prob,
            "layernorm_epsilon": config.layer_norm_eps,
            "add_cross_attention": config.add_cross_attention,
            "initializer_range": config.initializer_range,
            "use_normal_initialization": True,
        }

        return (), kwargs
