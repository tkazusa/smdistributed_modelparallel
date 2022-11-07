# Standard Library
import unittest

# Third Party
from transformers import set_seed

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo import (
    gpt2_base,
    gpt_bf16_threshold,
    gpt_fp16_threshold,
    gpt_fp32_threshold,
    gptj_base,
    gptneo_base,
    gptneox_base,
)
from smdistributed.modelparallel.test.torch.smp_test_base import SMPTestBase, SMPTestConfig

OPTIMIZERS = ["adam", "adamw"]


class TestZeroGPTBase(SMPTestBase):
    def setUp(self):
        super(TestZeroGPTBase, self).setUp()
        set_seed(2)

    def smp_enable_checkpointing_hook(self):
        module = self.current_run_model.get_module()

        if hasattr(module, "gpt_neox"):
            for c in module.gpt_neox.layers.children():
                smp.set_activation_checkpointing(c)
        elif hasattr(module.transformer, "seq_layers"):
            smp.set_activation_checkpointing(module.transformer.seq_layers, strategy="each")
        else:
            for c in module.transformer.h.children():
                smp.set_activation_checkpointing(c)

    def run_gpt(
        self,
        activation_checkpointing=False,
        activation_offloading=False,
        use_dist_transformer=False,
        optimizer=None,
        fp16=False,
        bf16=False,
        delayed_param=False,
    ):
        threshold = gpt_fp32_threshold
        if fp16:
            threshold = gpt_fp16_threshold
        elif bf16:
            threshold = gpt_bf16_threshold

        config = SMPTestConfig(
            verify_memory=False,
            verify_grad_counts=False,
            smp_config={
                "offload_activations": activation_offloading,
                "sharded_data_parallel_degree": 4,
                "skip_tracing": True,
                "tensor_parallel_degree": 1,
                "pipeline_parallel_degree": 1,
                "fp16": fp16,
                "bf16": bf16,
                "delayed_parameter_initialization": delayed_param,
            },
            optimizer=optimizer,
            tensor_parallel_kwargs={"tensor_parallelism": use_dist_transformer},
            **threshold,
        )
        self.set_test_config(config)
        if activation_checkpointing:
            self.register_pre_train_hook(smp_hook=self.smp_enable_checkpointing_hook)
        self.run_test(test_models=[gpt2_base, gptj_base, gptneo_base, gptneox_base])

class TestGPT(TestZeroGPTBase):
    def test_gpt(self):
        for opt in OPTIMIZERS:
            self.run_gpt(activation_checkpointing=False, activation_offloading=False, optimizer=opt)

    def test_gpt_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(activation_checkpointing=True, activation_offloading=False, optimizer=opt)

    def test_gpt_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(activation_checkpointing=True, activation_offloading=True, optimizer=opt)


class TestGPTDistTransformer(TestZeroGPTBase):
    def test_gpt_dist_transformer(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=False,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
            )

    def test_gpt_dist_transformer_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
            )

    def test_gpt_dist_transformer_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=True,
                use_dist_transformer=True,
                optimizer=opt,
            )


class TestGPTFP16(TestZeroGPTBase):
    def test_gpt(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=False,
                activation_offloading=False,
                optimizer=opt,
                fp16=True,
            )

    def test_gpt_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True, activation_offloading=False, optimizer=opt, fp16=True
            )

    def test_gpt_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True, activation_offloading=True, optimizer=opt, fp16=True
            )


class TestGPTFP16DistTransformer(TestZeroGPTBase):
    def test_gpt_dist_transformer(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=False,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
                fp16=True,
            )

    def test_gpt_dist_transformer_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
                fp16=True,
            )

    def test_gpt_dist_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=True,
                use_dist_transformer=True,
                optimizer=opt,
                fp16=True,
            )


class TestGPTBF16(TestZeroGPTBase):
    def test_gpt(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=False,
                activation_offloading=False,
                optimizer=opt,
                bf16=True,
            )

    def test_gpt_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True, activation_offloading=False, optimizer=opt, bf16=True
            )

    def test_gpt_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True, activation_offloading=True, optimizer=opt, bf16=True
            )


class TestGPTBF16DistTransformer(TestZeroGPTBase):
    def test_gpt_dist_transformer(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=False,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
                bf16=True,
            )

    def test_gpt_dist_transformer_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
                bf16=True,
            )

    def test_gpt_dist_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=True,
                use_dist_transformer=True,
                optimizer=opt,
                bf16=True,
            )


# Separated the unit tests to avoid hanging
# TODO: Investigate the root cause of the hanging
# class TestGPTFP16DelayedParam(TestZeroGPTBase):
class TestGPTFP16DelayedParamGPT(TestZeroGPTBase):
    def test_gpt(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=False,
                activation_offloading=False,
                optimizer=opt,
                fp16=True,
                delayed_param=True,
            )


class TestGPTFP16DelayedParamGPTCheckpoint(TestZeroGPTBase):
    def test_gpt_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=False,
                optimizer=opt,
                fp16=True,
                delayed_param=True,
            )


class TestGPTFP16DelayedParamGPTOffload(TestZeroGPTBase):
    def test_gpt_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=True,
                optimizer=opt,
                fp16=True,
                delayed_param=True,
            )


# Separated the unit tests to avoid hanging
# TODO: Investigate the root cause of the hanging
# class TestGPTFP16DelayedParamDistTransformer(TestZeroGPTBase):
class TestGPTFP16DelayedParamDistTransformerGPT(TestZeroGPTBase):
    def test_gpt_dist_transformer(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=False,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
                fp16=True,
                delayed_param=True,
            )


class TestGPTFP16DelayedParamDistTransformerGPTCheckpoint(TestZeroGPTBase):
    def test_gpt_dist_transformer_checkpoint(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=False,
                use_dist_transformer=True,
                optimizer=opt,
                fp16=True,
                delayed_param=True,
            )


class TestGPTFP16DelayedParamDistTransformerGPTOffload(TestZeroGPTBase):
    def test_gpt_dist_transformer_offload(self):
        for opt in OPTIMIZERS:
            self.run_gpt(
                activation_checkpointing=True,
                activation_offloading=True,
                use_dist_transformer=True,
                optimizer=opt,
                fp16=True,
                delayed_param=True,
            )


if __name__ == "__main__":
    unittest.main()
