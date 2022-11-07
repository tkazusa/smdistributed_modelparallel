# Standard Library
import os
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


class TestGPTBase(SMPTestBase):
    def setUp(self):
        super(TestGPTBase, self).setUp()
        set_seed(2)

    def smp_enable_checkpointing_hook(self):
        dist_module = self.current_run_model
        module = dist_module.get_module()
        checkpointing_module = module.transformer.seq_layers
        smp.set_activation_checkpointing(checkpointing_module, strategy="each")

    def run_gpt(
        self,
        activation_checkpointing=False,
        activation_offloading=False,
        prescaled_batch=False,
        shard_optimizer_state=False,
        optimizer=None,
        fp16=False,
        bf16=False,
        delayed_parameter_initialization=False,
        distribute_embedding=False,
        tensor_parallel_degree=None,
        pipeline_parallel_degree=None,
        fp32_residual_addition=False,
    ):
        threshold = gpt_fp32_threshold
        if fp16:
            threshold = gpt_fp16_threshold
        elif bf16:
            threshold = gpt_bf16_threshold
        test_torchdistx = int(os.getenv("SMP_TORCHDISTX_DEFERRED_INIT", 0)) > 0

        smp_config = {
            "offload_activations": activation_offloading,
            "prescaled_batch": prescaled_batch,
            "shard_optimizer_state": shard_optimizer_state,
            "fp16": fp16,
            "bf16": bf16,
            "delayed_parameter_initialization": delayed_parameter_initialization,
        }
        if tensor_parallel_degree is not None:
            smp_config["tensor_parallel_degree"] = tensor_parallel_degree
        if pipeline_parallel_degree is not None:
            smp_config["pipeline_parallel_degree"] = pipeline_parallel_degree

        config = SMPTestConfig(
            verify_parameters=shard_optimizer_state,
            smp_config=smp_config,
            optimizer=optimizer,
            tensor_parallel_kwargs={
                "distribute_embedding": distribute_embedding,
                "fp32_residual_addition": fp32_residual_addition,
            },
            **threshold,
        )
        self.set_test_config(config)
        if activation_checkpointing:
            self.register_pre_train_hook(smp_hook=self.smp_enable_checkpointing_hook)
        model_list = [gpt2_base, gptj_base, gptneo_base, gptneox_base]
        if test_torchdistx:
            model_list = [gpt2_base, gptneo_base]
        self.run_test(test_models=model_list)


class TestGPT(TestGPTBase):
    def test_gpt(self):
        self.run_gpt()

    def test_gpt_opt_sharding(self):
        for opt in ["adam", "adamw", "sgd", "adamax", "rmsprop"]:
            self.run_gpt(shard_optimizer_state=True, optimizer=opt)

    def test_gpt_activation_checkpointing(self):
        self.run_gpt(activation_checkpointing=True)

    def test_gpt_activation_offloading(self):
        self.run_gpt(activation_checkpointing=True, activation_offloading=True)

    def test_gpt_activation_prescaled_batch(self):
        self.run_gpt(
            activation_checkpointing=True, activation_offloading=True, prescaled_batch=True
        )


class TestGPTDelayedParam(TestGPTBase):
    def test_gpt_delayed_param_base(self):
        self.run_gpt(delayed_parameter_initialization=True)

    def test_gpt_delayed_param_full(self):
        self.run_gpt(
            delayed_parameter_initialization=True,
            activation_checkpointing=True,
            activation_offloading=True,
            prescaled_batch=True,
            shard_optimizer_state=True,
            optimizer="adam",
        )

    def test_gpt_delayed_param_fp16_base(self):
        self.run_gpt(delayed_parameter_initialization=True, fp16=True)

    def test_gpt_delayed_param_fp16_full(self):
        self.run_gpt(
            delayed_parameter_initialization=True,
            fp16=True,
            activation_checkpointing=True,
            activation_offloading=True,
            prescaled_batch=True,
            shard_optimizer_state=True,
            optimizer="adam",
        )


class TestGPTFP16(TestGPTBase):
    def test_gpt_opt_sharding_fp16(self):
        for opt in ["adam", "adamw", "sgd", "adamax", "rmsprop"]:
            self.run_gpt(shard_optimizer_state=True, optimizer=opt, fp16=True)

    def test_gpt_non_opt_sharding_fp16(self):
        for opt in ["adam", "adamw", "sgd", "adamax", "rmsprop"]:
            self.run_gpt(shard_optimizer_state=False, optimizer=opt, fp16=True)


class TestGPTDistributeEmbedding(TestGPTBase):
    def test_gpt_distribute_embedding(self):
        self.run_gpt(distribute_embedding=True)

    def test_gpt_distribute_embedding_prescaled_batch(self):
        self.run_gpt(prescaled_batch=True, distribute_embedding=True)

    def test_gpt_distribute_embedding_fp16(self):
        self.run_gpt(fp16=True, distribute_embedding=True)


class TestGPTFp32ResidualAddition(TestGPTBase):
    def test_gpt_fp32_residual_addition(self):
        self.run_gpt(fp32_residual_addition=True)

    def test_gpt_fp32_residual_addition_prescaled_batch(self):
        self.run_gpt(prescaled_batch=True, fp32_residual_addition=True)


class TestGPTBF16(TestGPTBase):
    def test_gpt_bf16(self):
        self.run_gpt(bf16=True)

    def test_gpt_bf16_tp_pp(self):
        self.run_gpt(bf16=True, pipeline_parallel_degree=2, tensor_parallel_degree=2)

    def test_gpt_bf16_opt_sharding(self):
        for opt in ["adam", "adamw", "sgd", "adamax", "rmsprop"]:
            self.run_gpt(
                bf16=True,
                pipeline_parallel_degree=2,
                tensor_parallel_degree=2,
                optimizer=opt,
                shard_optimizer_state=True,
            )

    def test_gpt_bf16_full(self):
        self.run_gpt(
            delayed_parameter_initialization=True,
            bf16=True,
            activation_checkpointing=True,
            activation_offloading=True,
            prescaled_batch=True,
            shard_optimizer_state=True,
            optimizer="adam",
            pipeline_parallel_degree=2,
            tensor_parallel_degree=2,
        )


if __name__ == "__main__":
    unittest.main()
