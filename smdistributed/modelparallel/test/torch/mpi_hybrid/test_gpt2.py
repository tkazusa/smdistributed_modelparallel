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
    GPTJConfig,
    GPT2Config,
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
    slice_and_compare_grads,
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
        model.word_embedding, dist_model.module.module.module.word_embedding, split=False
    )

    if args.model_type == "gpt2":
        equalize_embedding_weights(
            model.position_embedding, dist_model.module.module.module.position_embedding, split=False
        )
    return module_mapping


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
    if args.model_type == "gpt2":
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
    lm_head = TransformerLMHead(config).to(torch.device("cuda"))
    input_ids = torch.randint(0, 10, (batch_size, seq_length)).to(
        torch.device("cuda", smp.local_rank())
    )

    attention_mask = torch.randint(0, 20, (batch_size, seq_length)).to(
        torch.device("cuda", smp.local_rank())
    )
    inputs = input_ids, attention_mask, None, None, input_ids
    local_output = lm_head(inputs)
    nosmp_logits = local_output[1].cpu()
    local_output[0].backward()

    if args.model_type == "gpt2":
        model_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.num_positions,
            n_ctx=config.num_positions,
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
            # gradient_checkpointing=args.gradient_checkpointing > 0,
            use_cache=True,
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
    else:
        print("model_type can be gpt2 or gptj.")

    torch.set_default_dtype(torch.float16)

    # disabling fused_bias_gelu because it seems to introduce precision errors in outputs
    with smp.tensor_parallelism(enabled=smp.tp_size() > 1, fused_bias_gelu=False):
        model = AutoModelForCausalLM.from_config(model_config)
    torch.set_default_dtype(torch.float32)

    per_tp_rank_bs = batch_size // smp.tp_size()

    model = FP16_Module(model)
    model = smp.DistributedModel(model, trace_device="gpu", gradient_as_bucket_view=True)
    if args.activation_checkpointing:
        smp.set_activation_checkpointing(
            model.module.module.module.transformer.seq_layers,
            pack_args_as_tuple=True,
            strategy=args.activation_strategy,
        )
    mod_weight_mapping = map_and_match_mod_weights(lm_head, model, args)

    set_seed(args.seed)

    loss_mb, logits_mb = train_step(model, input_ids, attention_mask)
    logits = torch.cat(tuple(logits_mb.outputs), dim=1)
    logits_gathered = smp.allgather(logits, group=smp.TP_GROUP)
    smp_logits = torch.cat([t.transpose(0, 1).cpu() for t in logits_gathered], 1)

    # compare logits prescaled vs non-smp
    tp_rank_nosmp_logits = nosmp_logits[
        smp.tp_rank() * per_tp_rank_bs : (smp.tp_rank() + 1) * per_tp_rank_bs
    ][:, :-1, :]
    tp_rank_smp_logits = smp_logits[
        smp.tp_rank() * per_tp_rank_bs : (smp.tp_rank() + 1) * per_tp_rank_bs
    ]

    # logits atol is pretty high likely because logits are large, of the scale e+02 or e+03
    ACTIVATION_ATOL = 2 * 1e-1
    GRAD_ATOL = 1e-2

    assert torch.allclose(tp_rank_nosmp_logits, tp_rank_smp_logits, atol=ACTIVATION_ATOL), (
        smp.tp_rank(),
        tp_rank_nosmp_logits.shape,
        tp_rank_smp_logits.shape,
        tp_rank_nosmp_logits[0][0],
        tp_rank_smp_logits[0][0],
    )
    print(smp.rank(), "Logits with prescaled batch SMP run match non-SMP run")

    # compare grads prescaled vs non-smp
    for i, (mod, dist_mod, partition) in enumerate(mod_weight_mapping):
        if smp.state.module_manager.get_partition(dist_mod) == smp.pp_rank():
            slice_and_compare_grads(mod, dist_mod, partition=partition, atol=GRAD_ATOL)
    print(smp.rank(), "Grads with prescaled batch SMP run match non-SMP run")

    # --------
    # run without prescaled
    # -----
    if args.run_with_non_prescaled > 0:
        if args.offload_activations > 0:
            # reset internal state of offloader
            smp.state.offloaders.clear()

        # clone and zero grads
        prescaled_grads = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                prescaled_grads[n] = p.grad.detach()
                p.grad = None

        input_ids = input_ids.narrow(0, per_tp_rank_bs * smp.tp_rank(), per_tp_rank_bs)
        attention_mask = attention_mask.narrow(0, per_tp_rank_bs * smp.tp_rank(), per_tp_rank_bs)
        smp.state.cfg.prescaled_batch = False
        loss_mb_nop, logits_mb_nop = train_step(model, input_ids, attention_mask)
        logits_nop = torch.cat(tuple(logits_mb_nop.outputs), dim=1)
        smp_logits_nop = logits_nop.cpu()

        tp_rank_nosmp_logits = nosmp_logits[
            smp.tp_rank() * per_tp_rank_bs : (smp.tp_rank() + 1) * per_tp_rank_bs
        ]

        # compare logits of non-prescaled vs non-smp
        assert torch.allclose(smp_logits_nop, tp_rank_nosmp_logits, atol=ACTIVATION_ATOL), (
            smp_logits_nop,
            tp_rank_nosmp_logits,
        )

        # compare logits of non-prescaled vs prescaled
        assert torch.allclose(smp_logits_nop[:, :-1, :], tp_rank_smp_logits, atol=1e-3)

        print(
            smp.rank(),
            "Logits with non-prescaled batch SMP run match non-SMP run and prescaled batch SMP run",
        )

        # compare grads smp non-prescaled vs non-smp
        for i, (mod, dist_mod, partition) in enumerate(mod_weight_mapping):
            if smp.state.module_manager.get_partition(dist_mod) == smp.pp_rank():
                slice_and_compare_grads(mod, dist_mod, partition=partition, atol=GRAD_ATOL)

        # compare grads prescaled vs non-prescaled
        for n, p in model.named_parameters():
            if n in prescaled_grads:
                assert torch.allclose(p.grad, prescaled_grads[n])
            else:
                assert p.grad is None

        print(
            smp.rank(),
            "Grads with non-prescaled batch SMP run match non-SMP run and prescaled batch SMP run",
        )


if __name__ == "__main__":
    main()
