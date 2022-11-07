# Standard Library
import argparse

# Third Party
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPTJConfig,
    GPTNeoConfig,
    default_data_collator,
    set_seed,
)

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.mpi_4ps.utils import (
    TransformerLMHead,
    TransformerLMHeadConfig,
)
from smdistributed.modelparallel.test.torch.utils import (
    equalize_embedding_weights,
    equalize_linear_weights,
)
from smdistributed.modelparallel.torch.nn.huggingface.gpt2 import (
    translate_hf_state_dict_to_smdistributed_gpt2,
)
from smdistributed.modelparallel.torch.nn.huggingface.gptj import (
    translate_hf_state_dict_to_smdistributed_gptj,
)
from smdistributed.modelparallel.torch.nn.huggingface.gptneo import (
    translate_hf_state_dict_to_smdistributed_gptneo,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val

    return conversion_helper(val, half_conversion)


def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class FP16_Module(nn.Module):
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module("module", module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


@smp.step
def train_step(model, input_ids, attention_mask):
    # inputs = input_ids, attention_mask, None, None, input_ids
    result = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    model.backward(result["loss"])
    return result["loss"], result["logits"]


def parse_args():
    parser = argparse.ArgumentParser()

    opt_grp = parser.add_argument_group(
        title="optimization", description="arguments for optimization"
    )
    opt_grp.add_argument("--train_batch_size", type=int, default=8, help="batch size per dp_rank")
    opt_grp.add_argument("--seed", type=int, default=1234)
    opt_grp.add_argument(
        "--activation_checkpointing",
        type=int,
        default=1,
        help="enable gradient checkpointing to reduce memory consumption",
    )

    # configure model size
    model_grp = parser.add_argument_group(
        title="model", description="arguments to describe model configuration"
    )
    model_grp.add_argument("--model_type", type=str, default="gpt2")
    model_grp.add_argument("--max_context_width", type=int, default=512)
    model_grp.add_argument("--hidden_width", type=int, default=768)
    model_grp.add_argument("--num_layers", type=int, default=6)
    model_grp.add_argument("--num_heads", type=int, default=12)
    model_grp.add_argument("--resid_pdrop", type=float, default=0)
    model_grp.add_argument("--attn_pdrop", type=float, default=0)

    smp_grp = parser.add_argument_group(title="smp", description="smp")
    smp_grp.add_argument("--tensor_parallel_degree", type=int, default=4)
    smp_grp.add_argument("--pipeline_parallel_degree", type=int, default=1)
    smp_grp.add_argument("--rotary_dim", type=int, default=None)

    smp_grp.add_argument("--microbatches", type=int, default=1)
    smp_grp.add_argument("--active_microbatches", type=int, default=None)
    smp_grp.add_argument("--distribute_embedding", type=int, default=1)
    smp_grp.add_argument("--optimize", type=str, default="speed")
    smp_grp.add_argument(
        "--run_with_non_prescaled",
        type=int,
        default=1,
        help="If this is 1, runs non-prescaled as well after running pre-scaled",
    )
    smp_grp.add_argument("--activation_strategy", type=str, default="each")
    smp_grp.add_argument("--offload_activations", type=int, default=0)

    args, _ = parser.parse_known_args()
    args.distribute_embedding = args.distribute_embedding and args.model_type == "gpt2"
    return args


def map_and_match_mod_weights(model, dist_model, args):
    layernorm_ptn = "output" if args.optimize == "memory" else None
    linear1_ptn = "input" if args.optimize == "memory" else "output"
    module_mapping = []
    for idx in range(args.num_layers):
        layer = getattr(model.transformer.seq_layers, f"{idx}")
        dist_layer = getattr(dist_model.module.module.module.transformer.seq_layers, f"{idx}")
        module_mapping.extend(
            [
                (layer.attention.self.query, dist_layer.attention.query, linear1_ptn),
                (layer.attention.self.key, dist_layer.attention.key, linear1_ptn),
                (layer.attention.self.value, dist_layer.attention.value, linear1_ptn),
                (layer.attention.output.dense, dist_layer.attention.dense, "input"),
                # (layer.attention.output.LayerNorm, dist_layer.attention.layernorm, layernorm_ptn),
                (layer.intermediate.dense_act, dist_layer.output.dense1, linear1_ptn),
                (layer.output.dense, dist_layer.output.dense2, "input"),
            ]
        )
        if hasattr(layer.output, "LayerNorm"):
            module_mapping.extend(
                [(layer.output.LayerNorm, dist_layer.output.layernorm, layernorm_ptn)]
            )
        if layer.use_post_layernorm:
            module_mapping.extend(
                [(layer.output.LayerNorm, dist_layer.output.layernorm, layernorm_ptn)]
            )

        if layer.attention.output.use_post_layernorm:
            module_mapping.extend(
                [(layer.attention.output.LayerNorm, dist_layer.attention.layernorm, layernorm_ptn)]
            )

    for i, (mod, dist_mod, partition) in enumerate(module_mapping):
        equalize_linear_weights(mod, dist_mod, partition=partition)
    equalize_embedding_weights(
        model.word_embedding,
        dist_model.module.module.module.word_embedding,
        split=args.distribute_embedding > 0,
        vocab_parallel=True,
    )

    if args.model_type == "gpt2":
        equalize_embedding_weights(
            model.position_embedding,
            dist_model.module.module.module.position_embedding,
            split=False,
        )
    return module_mapping


def compare_weights(hf_layer, smp_layer, atol=1e-2):

    assert torch.allclose(
        hf_layer.attention.query.weight, smp_layer.attention.query.weight, atol=atol
    )

    if hasattr(hf_layer.attention.query, "bias") and hf_layer.attention.query.bias is not None:
        assert torch.allclose(
            hf_layer.attention.query.bias, smp_layer.attention.query.bias, atol=atol
        )
    assert torch.allclose(hf_layer.attention.key.weight, smp_layer.attention.key.weight, atol=atol)

    if hasattr(hf_layer.attention.key, "bias") and hf_layer.attention.key.bias is not None:
        assert torch.allclose(hf_layer.attention.key.bias, smp_layer.attention.key.bias, atol=atol)

    assert torch.allclose(
        hf_layer.attention.value.weight, smp_layer.attention.value.weight, atol=atol
    )

    if hasattr(hf_layer.attention.value, "bias") and hf_layer.attention.value.bias is not None:
        assert torch.allclose(
            hf_layer.attention.value.bias, smp_layer.attention.value.bias, atol=atol
        )

    assert torch.allclose(
        hf_layer.attention.dense.weight, smp_layer.attention.dense.weight, atol=atol
    )
    assert torch.allclose(hf_layer.output.dense1.weight, smp_layer.output.dense1.weight, atol=atol)
    assert torch.allclose(hf_layer.output.dense2.weight, smp_layer.output.dense2.weight, atol=atol)

    if hasattr(hf_layer.output, "layernorm"):
        assert torch.allclose(
            hf_layer.output.layernorm.weight, smp_layer.output.layernorm.weight, atol=atol
        )
        assert torch.allclose(
            hf_layer.output.layernorm.bias, smp_layer.output.layernorm.bias, atol=atol
        )

    if hasattr(hf_layer.output, "pre_layernorm") and hf_layer.output.pre_layernorm:
        assert torch.allclose(
            hf_layer.output.pre_layernorm.weight, smp_layer.output.pre_layernorm.weight, atol=atol
        )
        assert torch.allclose(
            hf_layer.output.pre_layernorm.bias, smp_layer.output.pre_layernorm.bias, atol=atol
        )

    if hasattr(hf_layer.attention, "layernorm"):
        assert torch.allclose(
            hf_layer.attention.layernorm.weight, smp_layer.attention.layernorm.weight, atol=atol
        )
        assert torch.allclose(
            hf_layer.attention.layernorm.bias, smp_layer.attention.layernorm.bias, atol=atol
        )

    if hasattr(hf_layer.attention, "pre_layernorm") and hf_layer.attention.pre_layernorm:
        assert torch.allclose(
            hf_layer.attention.pre_layernorm.weight,
            smp_layer.attention.pre_layernorm.weight,
            atol=atol,
        )
        assert torch.allclose(
            hf_layer.attention.pre_layernorm.bias, smp_layer.attention.pre_layernorm.bias, atol=atol
        )


def compare_embedding_weights(hf_module, smp_module, args, atol=1e-2):

    assert torch.allclose(
        hf_module.word_embedding.weight, smp_module.word_embedding.weight, atol=atol
    )

    if args.model_type == "gpt2":
        assert torch.allclose(
            hf_module.position_embedding.weight, smp_module.position_embedding.weight, atol=atol
        )


def main():
    args = parse_args()

    smp_config = {
        "pipeline_parallel_degree": args.pipeline_parallel_degree,
        "tensor_parallel_degree": args.tensor_parallel_degree,
        "ddp": True,
        "prescaled_batch": True,
        "optimize": args.optimize,
        "checkpoint_attentions": False if args.activation_checkpointing else True,
        "offload_activations": args.offload_activations > 0,
        "microbatches": args.microbatches,
    }

    if args.active_microbatches is not None:
        smp_config["active_microbatches"] = args.active_microbatches

    smp.init(smp_config)
    torch.cuda.set_device(smp.local_rank())
    batch_size = args.train_batch_size
    seq_length = args.max_context_width
    if args.model_type in ["gpt2", "gptneo"]:
        vocab_size = 50257
    else:
        vocab_size = 50400

    config = TransformerLMHeadConfig(
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        attention_head_size=(args.hidden_width // args.num_heads),
        hidden_size=args.hidden_width,
        intermediate_size=4 * args.hidden_width,
        vocab_size=vocab_size,
        num_positions=args.max_context_width,
        causal_mask_size=args.max_context_width,
        attention_dropout_prob=args.attn_pdrop,
        hidden_dropout_prob=args.resid_pdrop,
        rotary_dim=args.rotary_dim,
    )

    set_seed(args.seed)

    input_ids = torch.randint(0, 10, (batch_size, seq_length)).to(
        torch.device("cuda", smp.local_rank())
    )

    attention_mask = torch.randint(0, 20, (batch_size, seq_length)).to(
        torch.device("cuda", smp.local_rank())
    )

    if args.model_type == "gpt2":
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
        )
    elif args.model_type == "gptj":
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
            rotary_dim=args.rotary_dim,
            bos_token_id=50256,
            eos_token_id=50256,
            return_dict=True,
        )
    elif args.model_type == "gptneo":
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
        )
    else:
        print("model_type can be gpt2 or gptj.")

    LOGIT_ATOL = 1e-2
    WEIGHT_ATOL = 1e-2

    hf_model = AutoModelForCausalLM.from_config(model_config)
    hf_optimizer = torch.optim.SGD(hf_model.parameters(), lr=1.0)
    hf_model.cuda()
    hf_output = hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    hf_logits = hf_output.logits[..., :-1, :].contiguous()
    hf_logits.cuda()
    hf_loss = hf_output.loss
    hf_loss.backward()

    # disabling fused_bias_gelu because it seems to introduce precision errors in outputs
    with smp.tensor_parallelism(
        enabled=smp.tp_size() > 1,
        fused_bias_gelu=False,
        distribute_embedding=(args.distribute_embedding > 0),
    ):
        smp_model = AutoModelForCausalLM.from_config(model_config)

    smp_model = smp.DistributedModel(smp_model, trace_device="gpu", gradient_as_bucket_view=True)
    if args.model_type == "gpt2":
        smp_model.load_state_dict(
            translate_hf_state_dict_to_smdistributed_gpt2(
                hf_model.state_dict(), args.max_context_width
            ),
            strict=True,
        )
    elif args.model_type == "gptneo":
        smp_model.load_state_dict(
            translate_hf_state_dict_to_smdistributed_gptneo(
                hf_model.state_dict(), args.max_context_width
            ),
            strict=True,
        )
    elif args.model_type == "gptj":
        smp_model.load_state_dict(
            translate_hf_state_dict_to_smdistributed_gptj(
                hf_model.state_dict(), args.max_context_width
            ),
            strict=True,
        )

    if args.activation_checkpointing:
        smp.set_activation_checkpointing(
            smp_model.module.module.transformer.seq_layers, strategy=args.activation_strategy
        )

    smp_optimizer = torch.optim.SGD(smp_model.parameters(), lr=1.0)
    smp_optimizer = smp.DistributedOptimizer(smp_optimizer)
    loss_mb, logits_mb = train_step(smp_model, input_ids, attention_mask)

    logits = torch.cat(tuple(logits_mb.outputs), dim=1)
    logits_gathered = smp.allgather(logits, group=smp.TP_GROUP)

    dist_emb = args.distribute_embedding > 0
    if dist_emb:
        smp_logits = torch.cat([t.cpu() for t in logits_gathered], 2)
    else:
        smp_logits = torch.cat([t.transpose(0, 1).cpu() for t in logits_gathered], 1)

    assert torch.allclose(hf_logits, smp_logits.cuda(), atol=LOGIT_ATOL)
    print(smp.rank(), "Logits with prescaled batch SMP run match HF run")

    # Get smp_model logits after weight update
    smp.reset()
    smp.init(smp_config)
    torch.cuda.set_device(smp.local_rank())
    with smp.tensor_parallelism(
        enabled=smp.tp_size() > 1,
        fused_bias_gelu=False,
        distribute_embedding=(args.distribute_embedding > 0),
    ):
        smp_hf_state_loaded = AutoModelForCausalLM.from_config(model_config)
    smp_hf_state_loaded = smp.DistributedModel(
        smp_hf_state_loaded, trace_device="gpu", gradient_as_bucket_view=True
    )
    if args.model_type == "gpt2":
        smp_hf_state_loaded.load_state_dict(
            translate_hf_state_dict_to_smdistributed_gpt2(
                hf_model.state_dict(), args.max_context_width
            ),
            strict=True,
        )
    elif args.model_type == "gptneo":
        smp_hf_state_loaded.load_state_dict(
            translate_hf_state_dict_to_smdistributed_gptneo(
                hf_model.state_dict(), args.max_context_width
            ),
            strict=True,
        )
    elif args.model_type == "gptj":
        smp_hf_state_loaded.load_state_dict(
            translate_hf_state_dict_to_smdistributed_gptj(
                hf_model.state_dict(), args.max_context_width
            ),
            strict=True,
        )
    loss_mb_updated, logits_mb_updated = train_step(smp_hf_state_loaded, input_ids, attention_mask)
    logits_updated = torch.cat(tuple(logits_mb_updated.outputs), dim=1)
    logits_gathered_updated = smp.allgather(logits_updated, group=smp.TP_GROUP)
    # smp_logits_updated = torch.cat([t.transpose(0, 1).cpu() for t in logits_gathered_updated], 1)
    if dist_emb:
        smp_logits_updated = torch.cat([t.cpu() for t in logits_gathered_updated], 2)
    else:
        smp_logits_updated = torch.cat(
            [t.transpose(0, 1).cpu() for t in logits_gathered_updated], 1
        )

    # compare weights SMP vs HF
    for idx in range(args.num_layers):
        smp_hf_state_loaded_layer = getattr(
            smp_hf_state_loaded.module.module.transformer.seq_layers, f"{idx}"
        )
        smp_layer = getattr(smp_model.module.module.transformer.seq_layers, f"{idx}")
        if smp_hf_state_loaded_layer in set(smp_hf_state_loaded.local_modules()):
            compare_weights(smp_hf_state_loaded_layer.cuda(), smp_layer.cuda(), atol=WEIGHT_ATOL)
    print(smp.rank(), "Weights with prescaled batch SMP run match HF run")

    if smp_hf_state_loaded.module.module.word_embedding in set(smp_hf_state_loaded.local_modules()):
        compare_embedding_weights(
            smp_hf_state_loaded.module.module.cuda(),
            smp_model.module.module.cuda(),
            args,
            atol=WEIGHT_ATOL,
        )

    print(smp.rank(), "Embedding weights with prescaled batch SMP run match HF run")

    # --------
    # run without prescaled
    # -----

    if args.run_with_non_prescaled > 0:
        if args.offload_activations > 0:
            # reset internal state of offloader
            smp.state.offloaders.clear()

        per_tp_rank_bs = batch_size // smp.tp_size()

        input_ids = input_ids.narrow(0, per_tp_rank_bs * smp.tp_rank(), per_tp_rank_bs)
        attention_mask = attention_mask.narrow(0, per_tp_rank_bs * smp.tp_rank(), per_tp_rank_bs)
        smp.core.cfg.prescaled_batch = False
        if isinstance(
            smp_hf_state_loaded.module.module.word_embedding, smp.nn.DistributedEmbedding
        ):
            smp_hf_state_loaded.module.module.word_embedding._skip_allgather = False
            smp_hf_state_loaded.module.module.word_embedding._output_full_batch = False

        loss_mb_nop, logits_mb_nop = train_step(smp_hf_state_loaded, input_ids, attention_mask)
        logits_nop = torch.cat(tuple(logits_mb_nop.outputs), dim=1)

        # compare logits of non-prescaled vs prescaled
        if dist_emb:
            logits_nop_gathered = smp.allgather(logits_nop, group=smp.TP_GROUP)
            smp_logits_nop = torch.cat([t.cpu() for t in logits_nop_gathered], 2)
            assert torch.allclose(smp_logits_nop.cuda(), smp_logits_updated.cuda(), atol=LOGIT_ATOL)
        else:
            smp_logits_nop = logits_nop[:, :-1, :]
            tp_rank_smp_logits = smp_logits_updated[
                smp.tp_rank() * per_tp_rank_bs : (smp.tp_rank() + 1) * per_tp_rank_bs
            ]
            assert torch.allclose(smp_logits_nop.cuda(), tp_rank_smp_logits.cuda(), atol=LOGIT_ATOL)

        print(smp.rank(), "Logits with non-prescaled batch SMP run match prescaled batch SMP run")

        # smp_hf_state_loaded runs with non-prescaled batch
        for idx in range(args.num_layers):
            smp_nop_layer = getattr(
                smp_hf_state_loaded.module.module.transformer.seq_layers, f"{idx}"
            )
            smp_layer = getattr(smp_model.module.module.transformer.seq_layers, f"{idx}")
            if smp_nop_layer in set(smp_hf_state_loaded.local_modules()):
                compare_weights(smp_nop_layer.cuda(), smp_layer.cuda(), atol=WEIGHT_ATOL)
        print(smp.rank(), "Weights with non-prescaled batch SMP run match prescaled batch SMP run")

        if smp_hf_state_loaded.module.module.word_embedding in set(
            smp_hf_state_loaded.local_modules()
        ):
            compare_embedding_weights(
                smp_hf_state_loaded.module.module.cuda(),
                smp_model.module.module.cuda(),
                args,
                atol=WEIGHT_ATOL,
            )

        print(
            smp.rank(),
            "Embedding weights with  non-prescaled batch SMP run match prescaled batch SMP run",
        )


if __name__ == "__main__":
    main()
