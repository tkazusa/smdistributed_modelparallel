# Standard Library
import unittest

# Third Party
from transformers import set_seed

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo import vit_base
from smdistributed.modelparallel.test.torch.smp_test_base import SMPTestBase, SMPTestConfig

class TestViTBase(SMPTestBase):
    def setUp(self):
        super(TestViTBase, self).setUp()
        set_seed(2)

    def smp_enable_checkpointing_hook(self):
        dist_module = self.current_run_model
        module = dist_module.get_module()
        checkpointing_module = module.vit.encoder.seq_layers
        smp.set_activation_checkpointing(checkpointing_module, strategy="each")
    def run_vit(
            self,
            activation_checkpointing=False,
            activation_offloading=False,
            prescaled_batch=False,
            shard_optimizer_state=False,
            optimizer=None,
            fp16=False,
            delayed_parameter_initialization=False,
            distribute_embedding=False,
        ):
            config = SMPTestConfig(
                grad_atol=5e-1 if fp16 else 1e-3,
                grad_rtol=4e-1 if fp16 else 1e-3,
                param_atol=5e-3 if fp16 else 1e-3,
                param_rtol=5e-3,
                loss_atol=1e-2 if fp16 else 1e-5,
                loss_rtol=1e-3 if fp16 else 1e-5,
                verify_parameters=shard_optimizer_state,
                smp_config={
                    "offload_activations": activation_offloading,
                    "prescaled_batch": prescaled_batch,
                    "shard_optimizer_state": shard_optimizer_state,
                    "fp16": fp16,
                    "delayed_parameter_initialization": delayed_parameter_initialization,
                },
                optimizer=optimizer,
                tensor_parallel_kwargs={
                    "distribute_embedding": distribute_embedding,
                },
            )
            self.set_test_config(config)
            if activation_checkpointing:
                self.register_pre_train_hook(smp_hook=self.smp_enable_checkpointing_hook)
            self.run_test(test_models=[vit_base])

class TestViT(TestViTBase):
    def test_vit(self):
        self.run_vit()

    def test_vit_opt_sharding(self):
        for opt in ["adam", "adamw", "sgd", "adamax", "rmsprop"]:
            self.run_vit(shard_optimizer_state=True, optimizer=opt)

    def test_vit_activation_checkpointing(self):
        self.run_vit(activation_checkpointing=True)

    def test_vit_activation_offloading(self):
        self.run_vit(activation_checkpointing=True, activation_offloading=True)

    def test_vit_opt_sharding_fp16(self):
        for opt in ["adam", "adamw", "sgd", "adamax", "rmsprop"]:
            self.run_vit(shard_optimizer_state=True, optimizer=opt, fp16=True)

    def test_vit_non_opt_sharding_fp16(self):
        for opt in ["adam", "adamw", "sgd", "adamax", "rmsprop"]:
            self.run_vit(shard_optimizer_state=False, optimizer=opt, fp16=True)

if __name__ == "__main__":
    unittest.main()
