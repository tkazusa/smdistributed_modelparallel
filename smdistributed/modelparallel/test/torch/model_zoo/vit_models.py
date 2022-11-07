# Standard Library

# Third Party
import torch
from transformers import AutoModelForImageClassification, ViTConfig

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo.smp_test_model import TestModel

__all__ = ["vit_base"]

VIT_DEFAULT_BATCH_SIZE = 8
VIT_DEFAULT_IMAGE_SIZE = 224
VIT_DEFAULT_NUM_CHANNELS = 3


vit_smp_config = {
    "pipeline_parallel_degree": 1,
    "tensor_parallel_degree": 4,
    "ddp": True,
    "prescaled_batch": False,
    "optimize": "speed",
    "checkpoint_attentions": False,
    "offload_activations": False,
    "microbatches": 1,
    "shard_optimizer_state": False,
}


@smp.step
def train_vit(model, labels, *arg):
    pixel_values,  = arg
    result = model(pixel_values=pixel_values, labels=labels)
    loss = result["loss"] if isinstance(result, dict) else result[0]
    model.backward(loss)
    return loss  # , result["logits"]


def train_vit_nosmp(model, optimizer, labels, *arg):
    pixel_values,  = arg
    hf_output = model(pixel_values=pixel_values, labels=labels)
    hf_logits = hf_output["logits"][..., :-1, :].contiguous()
    hf_loss = hf_output["loss"]
    if smp.state.cfg.fp16 and optimizer:
        optimizer.backward(hf_loss)
    else:
        hf_loss.backward()
    return hf_loss  # , hf_logits


def create_vit_like_model(model_config):
    return AutoModelForImageClassification.from_config(model_config)

class ViTTestModel(TestModel):
    def create_inputs(self):
        pixel_values = torch.randint(low=0, high=255, size=self.input_sizes, dtype=torch.float32, device=torch.device("cuda", smp.local_rank()))
        labels = torch.randint(low=0, high=2, size=(VIT_DEFAULT_BATCH_SIZE, 2), dtype=torch.float32, device = torch.device("cuda", smp.local_rank()))

        self.inputs = (pixel_values, )
        self.target = labels

    def create_smp_model(self, **kwargs):
        self.smp_model = self.model_cls(*self.model_args, **self.model_kwargs)
        smp.set_tensor_parallelism(self.smp_model.vit.encoder)

    def set_vit_config(
        self,
        model_type="vit",
        hidden_width=768,
        num_layers=4,
        num_heads=12,
    ):
        id2label = {
                "0": "airplane",
                "1": "automobile",
        }
        
        label2id = {
                "airplane": 0,
                "automobile": 1,
        }

        if model_type  == "vit":
            model_config = ViTConfig(
                    num_hidden_layers=num_layers,
                    num_attention_heads=num_heads,
                    hidden_size = hidden_width,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.0,
                    attention_probs_dropout_prob=0.0,
                    layer_norm_eps=1e-12,
                    initializer_range=0.02,
                    return_dict=True,
                    use_cache=False,
                    is_encoder_decoder = False,
                    image_size = 224,
                    patch_size = 16,
                    num_channels = 3,
                    qkv_bias = True,
                    encoder_stride = 16,
                    num_labels=2,
                    id2label=id2label,
                    label2id=label2id,
                    intermediate_size=3072,
                )
        else:
            raise NotImplementedError("model_type can be vit.")

        self.set_step_function(non_smp_func=train_vit_nosmp, smp_func=train_vit)
        self.set_model_args(model_config)

vit_base = ViTTestModel(
    create_vit_like_model,
    input_sizes=(VIT_DEFAULT_BATCH_SIZE, VIT_DEFAULT_NUM_CHANNELS, VIT_DEFAULT_IMAGE_SIZE, VIT_DEFAULT_IMAGE_SIZE),
    smp_config=vit_smp_config,
)
vit_base.set_vit_config(model_type="vit")
