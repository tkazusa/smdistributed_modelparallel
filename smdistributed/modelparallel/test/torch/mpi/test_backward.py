# Standard Library
import os
import unittest

# First Party
from smdistributed.modelparallel.test.torch.model_zoo import (
    backward_models_pp2,
    checkpointing_models_no_grad,
    checkpointing_models_pp2,
    remote_inner_ckpt_failure,
    seq_backward_break,
)
from smdistributed.modelparallel.test.torch.smp_test_base import SMPTestBase, SMPTestConfig


class TestGrad(SMPTestBase):
    def test_backward(self):
        self.run_test(test_models=backward_models_pp2)


class TestCheckpointing(SMPTestBase):
    def non_smp_enable_checkpointing_hook(self):
        module = self.current_run_model
        module.checkpointing = True

    def test_non_checkpointing(self):
        self.run_test(test_models=checkpointing_models_pp2)

    def test_checkpointing(self):
        # register the hook to enable checkpointing
        self.register_pre_train_hook(non_smp_hook=self.non_smp_enable_checkpointing_hook)
        self.run_test(test_models=checkpointing_models_pp2)

    def test_no_grad(self):
        with self.only_run_smp():
            self.run_test(test_models=checkpointing_models_no_grad)


class TestCheckpointingOffload(TestCheckpointing):
    def setUp(self):
        super(TestCheckpointingOffload, self).setUp()
        config = SMPTestConfig(smp_config={"offload_activations": True})
        self.set_test_config(config)


@unittest.skipIf(
    int(os.getenv("RUN_FAIL_TESTS", 0)) < 1,
    "skips because this will fail. this test is run independently with execute_xfail in CI",
)
class TestCheckpointFailures(SMPTestBase):
    def tearDown(self):
        # remove barrier so process can exit after failing
        pass

    def test_seq_backward_break(self):
        with self.only_run_smp():
            self.run_test(test_models=[seq_backward_break])

    def test_remote_inner_ckpt_failure(self):
        with self.only_run_smp():
            self.run_test(test_models=[remote_inner_ckpt_failure])


if __name__ == "__main__":
    unittest.main()
