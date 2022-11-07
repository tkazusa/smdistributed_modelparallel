# Standard Library
import os
import unittest

# Third Party
from transformers import set_seed

# First Party
from smdistributed.modelparallel.test.torch.model_zoo import gpt2_base, gptj_base, gptneo_base
from smdistributed.modelparallel.test.torch.smp_test_base import SMPTestBase, SMPTestConfig


class TestGPTCheckpointBase(SMPTestBase):
    def setUp(self):
        super(TestGPTCheckpointBase, self).setUp()
        set_seed(2)

    def run_checkpoint(
        self,
        tensor_parallel_degree=2,
        pipeline_parallel_degree=2,
        shard_optimizer_state=False,
        fp16=False,
        delayed_parameter_initialization=False,
    ):
        config = SMPTestConfig(
            smp_config={
                "tensor_parallel_degree": tensor_parallel_degree,
                "pipeline_parallel_degree": pipeline_parallel_degree,
                "shard_optimizer_state": shard_optimizer_state,
                "fp16": fp16,
                "delayed_parameter_initialization": delayed_parameter_initialization,
            },
            optimizer="adam",
        )
        self.set_test_config(config)
        test_torchdistx = int(os.getenv("SMP_TORCHDISTX_DEFERRED_INIT", 0)) > 0
        model_list = [gpt2_base, gptj_base, gptneo_base]
        if test_torchdistx:
            model_list = [gpt2_base, gptneo_base]       
        self.run_checkpoint_test(test_models=model_list)


class TestGPTCheckpoint(TestGPTCheckpointBase):
    def test_tp_pp(self):
        self.run_checkpoint()

    def test_tp_pp_fp16(self):
        self.run_checkpoint(fp16=True)

    def test_tp_pp_fp16_shard(self):
        self.run_checkpoint(fp16=True, shard_optimizer_state=True)


class TestGPTCheckpointDelayedParam(TestGPTCheckpointBase):
    def test_tp_pp_fp16_shard_delayparam(self):
        self.run_checkpoint(
            fp16=True, shard_optimizer_state=True, delayed_parameter_initialization=True
        )


if __name__ == "__main__":
    unittest.main()
