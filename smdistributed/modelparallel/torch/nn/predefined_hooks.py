# First Party
from smdistributed.modelparallel.backend.logger import get_logger

_SUPPORTED_HF_VERSIONS = ["4.16.2"]
hf_transformers_available = True

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    from transformers.models.bert.modeling_bert import BertEncoder
    from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
    from transformers.models.roberta.modeling_roberta import RobertaEncoder

    from smdistributed.modelparallel.torch.nn.huggingface.bert import get_hf_bert_transformer_hooks
    from smdistributed.modelparallel.torch.nn.huggingface.gpt2 import (
        get_hf_gpt2_transformer_layer_hooks,
        get_hf_gpt2_transformer_lm_head_hooks,
    )
    from smdistributed.modelparallel.torch.nn.huggingface.roberta import (
        get_hf_roberta_transformer_hooks,
    )
    from smdistributed.modelparallel.torch.nn.huggingface.gptj import get_hf_gptj_transformer_hooks

except ImportError:
    hf_transformers_available = False


class PredefinedHookManager:
    def __init__(self):
        self.hf_transformers_available = hf_transformers_available
        self.hf_mappings = self._get_hf_mappings()

        self._all_mappings = {}
        self._all_mappings.update(self.hf_mappings)

    def _get_hf_mappings(self):
        from smdistributed.modelparallel.torch.nn import (
            DistributedTransformer,
            DistributedTransformerLMHead,
            DistributedTransformerLayer,
        )

        if self.hf_transformers_available:
            import transformers

            if transformers.__version__ not in _SUPPORTED_HF_VERSIONS:
                get_logger().warning(
                    f"Found unsupported HuggingFace version {transformers.__version__} for automated tensor parallelism. HuggingFace modules will not be automatically distributed. You can use smp.tp_register_with_module API to register desired modules for tensor parallelism, or directly instantiate an smp.nn.DistributedModule. Supported HuggingFace transformers versions for automated tensor parallelism: {_SUPPORTED_HF_VERSIONS}"
                )
                return {}

            # the order here must be the same order as that expected by tp_registry.register_with_module method
            return {
                "huggingface-bert": (
                    BertEncoder,
                    DistributedTransformer,
                    *get_hf_bert_transformer_hooks(),
                ),
                "huggingface-roberta": (
                    RobertaEncoder,
                    DistributedTransformer,
                    *get_hf_roberta_transformer_hooks(),
                ),
                "huggingface-gpt-2": (
                    GPT2LMHeadModel,
                    DistributedTransformerLMHead,
                    *get_hf_gpt2_transformer_lm_head_hooks(),
                ),
                "huggingface-gpt-2-layer": (
                    GPT2Block,
                    DistributedTransformerLayer,
                    *get_hf_gpt2_transformer_layer_hooks(),
                ),
                "huggingface-gpt-j": (
                    GPTJForCausalLM,
                    DistributedTransformerLMHead,
                    *get_hf_gptj_transformer_hooks(),
                ),
                # TODO skipping T5 until we properly support position_bias
                # "huggingface-t5": (
                #    T5Block,
                #    DistributedTransformerLayer,
                #    *get_hf_t5_transformer_layer_hooks(),
                # ),
            }
        else:
            return {}

    def get_mapping(self, key):
        if key.startswith("huggingface"):
            if key not in self.hf_mappings:
                if not self.hf_transformers_available:
                    raise ImportError(
                        "HuggingFace transformers library not found in the environment."
                    )

                raise KeyError(
                    f"Unrecognized HuggingFace key {key}. Must be one of {[k for k in self.hf_mappings]}."
                )
            return self.hf_mappings[key]
        else:
            raise KeyError(f"Unrecognized key {key}.")

    def all_mappings(self):
        for pair in self._all_mappings.items():
            yield pair
