# Third Party
# Standard Library
import argparse

import numpy as np
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTJConfig,
    GPTJForCausalLM,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    ViTConfig,
    ViTForImageClassification,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.nn import DistributedTransformerLayer
from smdistributed.modelparallel.torch.nn.huggingface.gpt2 import (
    get_hf_gpt2_transformer_layer_hooks,
    translate_hf_state_dict_to_smdistributed_gpt2,
    translate_hf_state_dict_to_smdistributed_gpt2_layer,
    translate_state_dict_to_hf_gpt2,
    translate_state_dict_to_hf_gpt2_layer,
)
from smdistributed.modelparallel.torch.nn.huggingface.gptj import (
    translate_hf_state_dict_to_smdistributed_gptj,
    translate_state_dict_to_hf_gptj,
)
from smdistributed.modelparallel.torch.nn.huggingface.gptneo import (
    translate_hf_state_dict_to_smdistributed_gptneo,
    translate_state_dict_to_hf_gptneo,
)
from smdistributed.modelparallel.torch.nn.huggingface.vit import (
    translate_hf_state_dict_to_smdistributed_vit,
    translate_state_dict_to_hf_vit,
)

from smdistributed.modelparallel.torch.nn.huggingface.gptneox import (
    translate_hf_state_dict_to_smdistributed_gptneox,
    translate_state_dict_to_hf_gptneox,
)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="model name")
    parser.add_argument("--atol_hf_to_rubik", type=float, default=9e-2)
    parser.add_argument("--atol_rubik_to_hf", type=float, default=9e-2)
    args, _ = parser.parse_known_args()
    return args


def test_translate_hf_state_dict_to_smdistributed(model, atol):
    torch.manual_seed(0)
    smp.init({"ddp": True, "tensor_parallel_degree": 2, "pipeline_parallel_degree": 1})

    # Custom gpt2 models are gpt2 models with only GPT2Blocks transformed into DistributedTransformerLayer
    if model == "custom_gpt2":
        model_name = "gpt2"
    else:
        model_name = model

    if "gpt" in model:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt")

    if smp.tp_rank() == 0:
        if "gpt2" in model:
            gpt2_configuration = GPT2Config(use_cache=False)
            hf_model = GPT2LMHeadModel(gpt2_configuration)
        elif "gpt-j" in model:
            gptj_configuration = GPTJConfig(n_embd=1024, n_head=8, n_layer=8, use_cache=False)
            hf_model = GPTJForCausalLM(gptj_configuration)
        elif "gpt-neox" in model:
            gptneox_configuration = GPTNeoXConfig(
                hidden_size=1024, num_hidden_layers=8, num_attention_heads=8, use_cache=False
            )
            hf_model = GPTNeoXForCausalLM(gptneox_configuration)
        elif "gpt-neo-" in model:
            gptneo_configuration = GPTNeoConfig(
                num_layers=4, attention_types=[[["global", "local"], 2]]
            )
            hf_model = GPTNeoForCausalLM(gptneo_configuration)
        elif "vit" in model:
            vit_configuration = ViTConfig(use_cache=False)
            hf_model = ViTForImageClassification(vit_configuration)
        else:
            raise ValueError(f"{model} model is not supported currently")
        hf_model.to(0)
        hf_model.eval()
        if "vit" in model:
            input_ids = torch.randint(low=0, high=255, size=[1, 3, 224, 224], dtype=torch.float32).cuda()
            hf_output = hf_model(input_ids)["logits"]
        else:
            input_ids = tokenized.input_ids.to(0)
            hf_output = hf_model(input_ids=input_ids, attention_mask=None)["logits"][0]
        hf_model.save_pretrained("./hf_model")

    smp.barrier()

    if "custom" in model:
        rubik_model = GPT2LMHeadModel.from_pretrained("./hf_model", cache_dir=None)
        smp.set_tensor_parallelism(rubik_model.transformer.h, True)
        smp.tp_register_with_module(
            GPT2Block, DistributedTransformerLayer, *get_hf_gpt2_transformer_layer_hooks()
        )
    elif "vit" in model:
        rubik_model = ViTForImageClassification.from_pretrained("./hf_model", cache_dir=None)
        smp.set_tensor_parallelism(rubik_model.vit.encoder)
    else:
        with smp.tensor_parallelism(enabled=True):
            if "gpt2" in model:
                rubik_model = GPT2LMHeadModel.from_pretrained("./hf_model", cache_dir=None)
            elif "gpt-j" in model:
                rubik_model = GPTJForCausalLM.from_pretrained("./hf_model", cache_dir=None)
            elif "gpt-neox" in model:
                rubik_model = GPTNeoXForCausalLM.from_pretrained("./hf_model", cache_dir=None)
            elif "gpt-neo-" in model:
                rubik_model = GPTNeoForCausalLM.from_pretrained("./hf_model", cache_dir=None)

    rubik_model = smp.DistributedModel(rubik_model)
    checkpoint = torch.load("./hf_model/pytorch_model.bin", map_location=torch.device("cpu"))
    translate_function = None
    if model == "gpt2":
        translate_function = translate_hf_state_dict_to_smdistributed_gpt2
    elif "custom" in model:
        translate_function = translate_hf_state_dict_to_smdistributed_gpt2_layer
    elif "gpt-j" in model:
        translate_function = translate_hf_state_dict_to_smdistributed_gptj
    elif "gpt-neox" in model:
        translate_function = translate_hf_state_dict_to_smdistributed_gptneox
    elif "gpt-neo-" in model:
        translate_function = translate_hf_state_dict_to_smdistributed_gptneo
    elif "vit" in model:
        translate_function = translate_hf_state_dict_to_smdistributed_vit

    rubik_model.load_state_dict(checkpoint, strict=True, translate_function=translate_function)

    @smp.step
    def eval(model, input_ids, attention_mask):
        # ViT case
        if attention_mask == None:
            output = model(input_ids)["logits"]
        else:
            output = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        return output

    if "vit" in model:
        input_ids = torch.randint(low=0, high=255, size=[1, 3, 224, 224], dtype=torch.float32).cuda()
        attention_mask = None
    else:
        input_ids = tokenized.input_ids
        input_size = list(input_ids.size())
        attention_mask = torch.ones(1, input_size[1])

    rubik_model.eval()
    rubik_output = eval(rubik_model, input_ids, attention_mask)[0]
    if "custom" in model:
        rubik_output = rubik_output[0]

    if smp.tp_rank() == 0:
        np.testing.assert_allclose(hf_output.detach().cpu(), rubik_output.detach().cpu(), atol=atol)


def test_translate_state_dict_to_hf_gpt2(model, atol):
    torch.manual_seed(0)
    smp.init({"ddp": True, "tensor_parallel_degree": 2, "pipeline_parallel_degree": 1})

    # Custom gpt2 models are gpt2 models with only GPT2Blocks transformed into DistributedTransformerLayer
    if model == "custom_gpt2":
        model_name = "gpt2"
    else:
        model_name = model

    if "gpt" in model:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt")

    if "custom" in model:
        rubik_model = GPT2LMHeadModel.from_pretrained("./hf_model", cache_dir=None)
        smp.set_tensor_parallelism(rubik_model.transformer.h, True)
        smp.tp_register_with_module(
            GPT2Block, DistributedTransformerLayer, *get_hf_gpt2_transformer_layer_hooks()
        )
    else:
        with smp.tensor_parallelism(enabled=True):
            if "gpt2" in model:
                gpt2_configuration = GPT2Config(use_cache=False)
                rubik_model = GPT2LMHeadModel(gpt2_configuration)
            elif "gpt-j" in model:
                gptj_configuration = GPTJConfig(n_embd=1024, n_head=8, n_layer=8, use_cache=False)
                rubik_model = GPTJForCausalLM(gptj_configuration)
            elif "gpt-neox" in model:
                gptneox_configuration = GPTNeoXConfig(
                    hidden_size=1024, num_hidden_layers=8, num_attention_heads=8, use_cache=False
                )
                rubik_model = GPTNeoXForCausalLM(gptneox_configuration)
            elif "gpt-neo-" in model:
                gptneo_configuration = GPTNeoConfig(
                    num_layers=4, attention_types=[[["global", "local"], 2]]
                )
                rubik_model = GPTNeoForCausalLM(gptneo_configuration)
            elif "vit" in model:
                vit_configuration = ViTConfig(use_cache=False)
                rubik_model = ViTForImageClassification(vit_configuration)
            else:
                raise ValueError(f"{model} model is not supported currently")
    rubik_model = smp.DistributedModel(rubik_model)

    @smp.step
    def eval(model, input_ids, attention_mask):
        if attention_mask == None:
            output = model(input_ids)["logits"]
        else:
            output = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        return output

    if "vit" in model:
        input_ids = torch.ones([1, 3, 224, 224], dtype=torch.float32).cuda()
        attention_mask = None
    else:
        input_ids = tokenized.input_ids
        input_size = list(input_ids.size())
        attention_mask = torch.ones(1, input_size[1])
    rubik_model.eval()
    rubik_output = eval(rubik_model, input_ids, attention_mask)[0]
    if "custom" in model:
        rubik_output = rubik_output[0]
    model_state_dict = rubik_model.state_dict(gather_to_rank0=True)

    smp.reset()
    if smp.tp_rank() == 0:
        if "gpt2" in model:
            gpt2_configuration = GPT2Config(use_cache=False)
            hf_model = GPT2LMHeadModel(gpt2_configuration)
        if "gpt-j" in model:
            gptj_configuration = GPTJConfig(n_embd=1024, n_head=8, n_layer=8, use_cache=False)
            hf_model = GPTJForCausalLM(gptj_configuration)
        if "gpt-neo-" in model:
            gptneo_configuration = GPTNeoConfig(
                num_layers=4, attention_types=[[["global", "local"], 2]]
            )
            hf_model = GPTNeoForCausalLM(gptneo_configuration)
        if "vit" in model:
            vit_configuration = ViTConfig(use_cache=False)
            hf_model = ViTForImageClassification(vit_configuration)
        if "gpt-neox" in model:
            gptneox_configuration = GPTNeoXConfig(
                hidden_size=1024, num_hidden_layers=8, num_attention_heads=8, use_cache=False
            )
            hf_model = GPTNeoXForCausalLM(gptneox_configuration)
        hf_model.to(0)
        hf_model.eval()
        if model == "gpt2":
            hf_state_dict = translate_state_dict_to_hf_gpt2(
                model_state_dict, hf_model.config.n_positions
            )
        elif "custom" in model:
            hf_state_dict = translate_state_dict_to_hf_gpt2_layer(
                model_state_dict, hf_model.config.n_positions
            )
        elif "gpt-j" in model:
            hf_state_dict = translate_state_dict_to_hf_gptj(
                model_state_dict, hf_model.config.n_positions
            )
        elif "gpt-neox" in model:
            hf_state_dict = translate_state_dict_to_hf_gptneox(
                model_state_dict, hf_model.config.max_position_embeddings
            )
        elif "gpt-neo-" in model:
            hf_state_dict = translate_state_dict_to_hf_gptneo(
                model_state_dict, hf_model.config.max_position_embeddings
            )
        elif "vit" in model:
            hf_state_dict = translate_state_dict_to_hf_vit(
                model_state_dict
            )

        hf_model.load_state_dict(hf_state_dict, strict=True)

        if "vit" in model:
            input_ids = torch.ones([1, 3, 224, 224], dtype=torch.float32).cuda()
            hf_output = hf_model(input_ids)["logits"]
        else:
            input_ids = tokenized.input_ids.to(0)
            hf_output = hf_model(input_ids=input_ids, attention_mask=None)["logits"][0]
        np.testing.assert_allclose(hf_output.detach().cpu(), rubik_output.detach().cpu(), atol=atol)


def main():
    args = parse_args()
    test_translate_hf_state_dict_to_smdistributed(args.model, args.atol_hf_to_rubik)
    test_translate_state_dict_to_hf_gpt2(args.model, args.atol_rubik_to_hf)


if __name__ == "__main__":
    main()
