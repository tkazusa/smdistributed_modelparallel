# Standard Library

# Third Party
import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPTJConfig, GPTNeoConfig, GPTNeoXConfig

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo.smp_test_model import TestModel
from smdistributed.modelparallel.test.torch.mpi_4ps.utils import TransformerLMHeadConfig
from smdistributed.modelparallel.torch.nn.huggingface.gpt2 import (
    translate_hf_state_dict_to_smdistributed_gpt2,
)
from smdistributed.modelparallel.torch.nn.huggingface.gptj import (
    translate_hf_state_dict_to_smdistributed_gptj,
)
from smdistributed.modelparallel.torch.nn.huggingface.gptneo import (
    translate_hf_state_dict_to_smdistributed_gptneo,
)
from smdistributed.modelparallel.torch.nn.huggingface.gptneox import (
    translate_hf_state_dict_to_smdistributed_gptneox,
)

__all__ = [
    "gpt2_base",
    "gptj_base",
    "gptneo_base",
    "gptneox_base",
    "gpt_fp32_threshold",
    "gpt_fp16_threshold",
    "gpt_bf16_threshold",
]

GPT_DEFAULT_BATCH_SIZE = 8
GPT_DEFAULT_SEQ_LEN = 512

gpt_smp_config = {
    "pipeline_parallel_degree": 1,
    "tensor_parallel_degree": 4,
    "ddp": True,
    "prescaled_batch": False,
    "optimize": "speed",
    "checkpoint_attentions": False,
    "offload_activations": False,
    "microbatches": 1,
}

gpt_fp32_threshold = dict(
    grad_atol=1e-3, grad_rtol=1e-3, param_atol=1e-3, param_rtol=5e-3, loss_atol=1e-5, loss_rtol=1e-5
)
gpt_fp16_threshold = dict(
    grad_atol=3.5e-2,
    grad_rtol=1e-2,
    param_atol=5e-3,
    param_rtol=5e-3,
    loss_atol=1e-2,
    loss_rtol=1e-3,
)
gpt_bf16_threshold = dict(
    grad_atol=5e-3, grad_rtol=5e-3, param_atol=5e-3, param_rtol=5e-3, loss_atol=1e-8, loss_rtol=1e-8
)


@smp.step
def train_gpt(model, target, *arg):
    input_ids, attention_mask = arg
    result = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
    model.backward(result["loss"])
    return result["loss"]  # , result["logits"]


def train_gpt_nosmp(model, optimizer, target, *arg):
    input_ids, attention_mask = arg
    hf_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
    hf_logits = hf_output["logits"][..., :-1, :].contiguous()
    hf_loss = hf_output["loss"]
    if smp.state.cfg.fp16 and optimizer:
        optimizer.backward(hf_loss)
    else:
        hf_loss.backward()
    return hf_loss  # , hf_logits


def create_gpt_like_model(model_config):
    return AutoModelForCausalLM.from_config(model_config)


class GPTTestModel(TestModel):
    def create_inputs(self):
        input_ids = torch.randint(0, self.vocab_size, self.input_sizes[0]).to(
            torch.device("cuda", smp.local_rank())
        )
        attention_mask = torch.randint(1, 2, self.input_sizes[0]).to(
            torch.device("cuda", smp.local_rank())
        )
        self.inputs = (input_ids, attention_mask)
        self.target = input_ids

    def set_gpt_config(
        self,
        model_type="gpt2",
        max_context_width=GPT_DEFAULT_SEQ_LEN,
        hidden_width=768,
        num_layers=6,
        num_heads=12,
        resid_pdrop=0,
        attn_pdrop=0,
        rotary_dim=None,
        rotary_pct=None,
        rotary_emb_base=None,
    ):
        self.set_step_function(non_smp_func=train_gpt_nosmp, smp_func=train_gpt)
        if model_type in ["gpt2", "gptneo"]:
            vocab_size = 50257
        elif model_type in ["gptneox"]:
            vocab_size = 50432
        else:
            vocab_size = 50400

        self.vocab_size = vocab_size

        assert (
            max_context_width == self.input_sizes[0][1]
        ), f"max_context_width must match the 2nd dimention of input size"
        config = TransformerLMHeadConfig(
            num_layers=num_layers,
            num_attention_heads=num_heads,
            attention_head_size=(hidden_width // num_heads),
            hidden_size=hidden_width,
            intermediate_size=4 * hidden_width,
            vocab_size=vocab_size,
            num_positions=max_context_width,
            causal_mask_size=max_context_width,
            attention_dropout_prob=attn_pdrop,
            hidden_dropout_prob=resid_pdrop,
            rotary_dim=rotary_dim,
            rotary_pct=rotary_pct,
            rotary_emb_base=rotary_emb_base,
        )
        if model_type == "gpt2":
            model_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_positions=config.num_positions,
                n_embd=config.hidden_size,
                n_layer=config.num_layers,
                n_head=config.num_attention_heads,
                n_inner=None,
                activation_function="gelu_new",
                resid_pdrop=config.hidden_dropout_prob,
                embd_pdrop=0,
                attn_pdrop=config.attention_dropout_prob,
                layer_norm_epsilon=1e-05,
                initializer_range=0.02,
                summary_type="cls_index",
                summary_use_proj=True,
                summary_activation=None,
                summary_proj_to_labels=True,
                summary_first_dropout=0,
                bos_token_id=50256,
                eos_token_id=50256,
                return_dict=True,
                use_cache=False,
            )
        elif model_type == "gptj":
            model_config = GPTJConfig(
                vocab_size=config.vocab_size,
                n_positions=config.num_positions,
                n_embd=config.hidden_size,
                n_layer=config.num_layers,
                n_head=config.num_attention_heads,
                n_inner=None,
                activation_function="gelu_new",
                resid_pdrop=config.hidden_dropout_prob,
                embd_pdrop=0,
                attn_pdrop=config.attention_dropout_prob,
                layer_norm_epsilon=1e-05,
                initializer_range=0.02,
                rotary_dim=config.rotary_dim,
                bos_token_id=50256,
                eos_token_id=50256,
                return_dict=True,
                use_cache=False,
            )
        elif model_type == "gptneo":
            model_config = GPTNeoConfig(
                vocab_size=config.vocab_size,
                max_position_embeddings=config.num_positions,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                attention_types=[[["global", "local"], int(config.num_layers / 2)]],
                num_heads=config.num_attention_heads,
                intermediate_size=None,
                activation_function="gelu_new",
                resid_dropout=config.hidden_dropout_prob,
                embed_dropout=0,
                attention_dropout_prob=config.attention_dropout_prob,
                layer_norm_epsilon=1e-05,
                initializer_range=0.02,
                bos_token_id=50256,
                eos_token_id=50256,
                return_dict=True,
                use_cache=False,
            )
        elif model_type == "gptneox":
            model_config = GPTNeoXConfig(
                vocab_size=config.vocab_size,
                max_position_embeddings=config.num_positions,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                activation_function="gelu_new",
                layer_norm_eps=1e-05,
                initializer_range=0.02,
                return_dict=True,
                use_cache=False,
                rotary_pct=config.rotary_pct,
                rotary_emb_base=config.rotary_emb_base,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        else:
            raise NotImplementedError("model_type can be gpt2, gptj or gptneo.")
        self.set_model_args(model_config)


gpt2_base = GPTTestModel(
    create_gpt_like_model,
    input_sizes=[(GPT_DEFAULT_BATCH_SIZE, GPT_DEFAULT_SEQ_LEN)],
    smp_config=gpt_smp_config,
)
gpt2_base.set_gpt_config()

gptj_base = GPTTestModel(
    create_gpt_like_model,
    input_sizes=[(GPT_DEFAULT_BATCH_SIZE, GPT_DEFAULT_SEQ_LEN)],
    smp_config=gpt_smp_config,
)
gptj_base.set_gpt_config(model_type="gptj", rotary_dim=64)

gptneo_base = GPTTestModel(
    create_gpt_like_model,
    input_sizes=[(GPT_DEFAULT_BATCH_SIZE, GPT_DEFAULT_SEQ_LEN)],
    smp_config=gpt_smp_config,
)
gptneo_base.set_gpt_config(model_type="gptneo")

gptneox_base = GPTTestModel(
    create_gpt_like_model,
    input_sizes=[(GPT_DEFAULT_BATCH_SIZE, GPT_DEFAULT_SEQ_LEN)],
    smp_config=gpt_smp_config,
)
gptneox_base.set_gpt_config(model_type="gptneox", rotary_pct=1.0, rotary_emb_base=10000)