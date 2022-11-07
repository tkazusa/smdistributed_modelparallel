try:
    hf_transformers_available = True
except ImportError:
    hf_transformers_available = False

# First Party
from smdistributed.modelparallel.torch.exceptions import HFT5ConfigError

if hf_transformers_available:

    def hf_t5_transformer_layer_init_hook(config, has_relative_attention_bias=False):
        if has_relative_attention_bias:
            # has_relative_attention_bias argument of HuggingFace T5Block is not supported
            # this results in only the first layer of T5 not being partitioned
            return None

        if config.d_kv * config.num_heads != config.d_model:
            raise HFT5ConfigError(
                f"The product of d_kv ({config.d_kv}) and num_heads ({config.num_heads}) must be equal to d_model ({d_model}) in T5 configuration."
            )

        kwargs = {
            "num_attention_heads": config.num_heads,
            "hidden_size": config.d_model,
            "intermediate_size": config.d_ff,
            "attention_dropout_prob": config.dropout_rate,
            "hidden_dropout_prob": config.dropout_rate,
            "add_cross_attention": config.is_decoder,
        }

        return (), kwargs

    def hf_t5_transformer_layer_forward_hook(
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if (
            layer_head_mask is not None
            or encoder_layer_head_mask is not None
            or past_key_value is not None
            or use_cache
            or output_attentions
            or not return_dict
        ):
            raise HFT5ConfigError(
                f"layer_head_mask, encoder_layer_head_mask, past_key_value, use_cache, output_attentions, and return_dict arguments of T5Block are not supported."
            )

        if attention_mask is None:
            raise HFT5ConfigError(
                "attention_mask is a required argument of DistributedTransformerLayer."
            )

        # TODO vv this logic is wrong - it keeps re-adding position bias at every layer
        if position_bias is not None:
            # the HF logic effectively adds the position_bias to attention_mask
            num_local_channels = position_bias.size(1) // tp_size()
            position_bias = position_bias.narrow(
                1, num_local_channels * tp_rank(), num_local_channels
            )

            position_bias += attention_mask
            attention_mask = position_bias

        if encoder_decoder_position_bias is not None:
            num_local_channels = encoder_decoder_position_bias.size(1) // tp_size()
            encoder_decoder_position_bias = encoder_decoder_position_bias.narrow(
                1, num_local_channels * tp_rank(), num_local_channels
            )

            encoder_decoder_position_bias += encoder_attention_mask
            encoder_attention_mask = encoder_decoder_position_bias

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

    def hf_t5_transformer_layer_return_hook(outputs):
        return (outputs[0], None, None)

    def get_hf_t5_transformer_layer_hooks():
        return (
            hf_t5_transformer_layer_init_hook,
            hf_t5_transformer_layer_forward_hook,
            hf_t5_transformer_layer_return_hook,
        )
