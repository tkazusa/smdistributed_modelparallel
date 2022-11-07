# Standard Library
import unittest

# Third Party
from transformers import set_seed

# First Party
from smdistributed.modelparallel.test.torch.model_zoo import gpt2_base, gptj_base, gptneo_base
from smdistributed.modelparallel.test.torch.smp_test_base import SMPTestBase, SMPTestConfig


class TestZeroCheckpointBase(SMPTestBase):
    def setUp(self):
        super(TestZeroCheckpointBase, self).setUp()
        set_seed(2)

    def run_checkpoint(self, use_dist_transformer=False, fp16=False, delayed_param=False):
        config = SMPTestConfig(
            verify_memory=False,
            smp_config={
                "sharded_data_parallel_degree": 4,
                "skip_tracing": True,
                "tensor_parallel_degree": 1,
                "pipeline_parallel_degree": 1,
                "fp16": fp16,
                "delayed_parameter_initialization": delayed_param,
            },
            tensor_parallel_kwargs={"tensor_parallelism": use_dist_transformer},
            optimizer="adam",
        )
        self.set_test_config(config)
        self.run_checkpoint_test(test_models=[gpt2_base, gptj_base, gptneo_base])


class TestGPTCheckpoint(TestZeroCheckpointBase):
    def test_gpt_base(self):
        self.run_checkpoint()

    def test_gpt_dist_transformer(self):
        self.run_checkpoint(use_dist_transformer=True)

    def test_gpt_dist_transformer_fp16(self):
        self.run_checkpoint(use_dist_transformer=True, fp16=True)


if __name__ == "__main__":
    unittest.main()
