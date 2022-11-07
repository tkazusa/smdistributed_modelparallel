# Standard Library
import sys
import unittest

# First Party
from smdistributed.modelparallel.test.torch.mpi.test_backward import TestCheckpointing, TestGrad
from smdistributed.modelparallel.test.torch.smp_test_base import SMPTestConfig


class TestAllreduceDDP(TestGrad):
    def setUp(self):
        super(TestAllreduceDDP, self).setUp()
        config = SMPTestConfig(
            batch_size=80,
            smp_dist_model_kwargs={"overlapping_allreduce": True, "gradient_as_bucket_view": False},
        )
        self.set_test_config(config)


class TestAllreduceDDPCheckpointing(TestCheckpointing):
    def setUp(self):
        super(TestAllreduceDDPCheckpointing, self).setUp()
        config = SMPTestConfig(
            batch_size=80,
            smp_dist_model_kwargs={"overlapping_allreduce": True, "gradient_as_bucket_view": False},
        )
        self.set_test_config(config)


class TestAllreduceDDPGradBucket(TestGrad):
    def setUp(self):
        super(TestAllreduceDDPGradBucket, self).setUp()
        config = SMPTestConfig(
            batch_size=80,
            smp_dist_model_kwargs={"overlapping_allreduce": True, "gradient_as_bucket_view": True},
        )
        self.set_test_config(config)


class TestAllreduceDDPGradBucketCheckpointing(TestCheckpointing):
    def setUp(self):
        super(TestAllreduceDDPGradBucketCheckpointing, self).setUp()
        config = SMPTestConfig(
            batch_size=80,
            smp_dist_model_kwargs={"overlapping_allreduce": True, "gradient_as_bucket_view": True},
        )
        self.set_test_config(config)


class TestAllreduceDDPNonoverlapping(TestGrad):
    def setUp(self):
        super(TestAllreduceDDPNonoverlapping, self).setUp()
        config = SMPTestConfig(
            batch_size=80,
            smp_dist_model_kwargs={
                "overlapping_allreduce": False,
                "gradient_as_bucket_view": False,
            },
        )
        self.set_test_config(config)


class TestAllreduceDDPNonoverlappingCheckpointing(TestCheckpointing):
    def setUp(self):
        super(TestAllreduceDDPNonoverlappingCheckpointing, self).setUp()
        config = SMPTestConfig(
            batch_size=80,
            smp_dist_model_kwargs={
                "overlapping_allreduce": False,
                "gradient_as_bucket_view": False,
            },
        )
        self.set_test_config(config)


class TestAllreduceDDPGradAccum(TestGrad):
    def setUp(self):
        super(TestAllreduceDDPGradAccum, self).setUp()
        num_steps = 3
        config = SMPTestConfig(
            num_steps=num_steps,
            batch_size=160,
            smp_config={"microbatches": 4},
            smp_dist_model_kwargs={
                "overlapping_allreduce": True,
                "gradient_as_bucket_view": False,
                "backward_passes_per_step": num_steps,
            },
        )
        self.set_test_config(config)


class TestAllreduceDDPGradAccumCheckpointing(TestCheckpointing):
    def setUp(self):
        super(TestAllreduceDDPGradAccumCheckpointing, self).setUp()
        num_steps = 3
        config = SMPTestConfig(
            num_steps=num_steps,
            batch_size=160,
            smp_config={"microbatches": 4},
            smp_dist_model_kwargs={
                "overlapping_allreduce": True,
                "gradient_as_bucket_view": False,
                "backward_passes_per_step": num_steps,
            },
        )
        self.set_test_config(config)


class TestAllreduceDDPGradAccumNonoverlapping(TestGrad):
    def setUp(self):
        super(TestAllreduceDDPGradAccumNonoverlapping, self).setUp()
        num_steps = 3
        config = SMPTestConfig(
            num_steps=num_steps,
            batch_size=160,
            smp_config={"microbatches": 4},
            smp_dist_model_kwargs={
                "overlapping_allreduce": False,
                "gradient_as_bucket_view": False,
                "backward_passes_per_step": num_steps,
            },
        )
        self.set_test_config(config)


class TestAllreduceFP32GradBucket(TestGrad):
    def setUp(self):
        super(TestAllreduceFP32GradBucket, self).setUp()
        # Requires reduced tolerance check when _fp32_grad_accumulation is enabled, not sure why since computation
        # is exactly the same.
        config = SMPTestConfig(
            grad_atol=5e-3,
            grad_rtol=1e-3,
            batch_size=80,
            smp_config={"_fp32_grad_accumulation": True, "fp16": True},
            smp_dist_model_kwargs={"overlapping_allreduce": True, "gradient_as_bucket_view": True},
            upscale_model_output=True,
        )
        self.set_test_config(config)


class TestAllreduceFP32NonGradBucket(TestGrad):
    def setUp(self):
        super(TestAllreduceFP32NonGradBucket, self).setUp()
        config = SMPTestConfig(
            grad_atol=5e-3,
            grad_rtol=1e-3,
            batch_size=80,
            smp_config={"_fp32_grad_accumulation": True, "fp16": True},
            smp_dist_model_kwargs={"overlapping_allreduce": True, "gradient_as_bucket_view": False},
            upscale_model_output=True,
        )
        self.set_test_config(config)


class TestAllreduceFP32DDPGradBucketGradAccum(TestGrad):
    def setUp(self):
        super(TestAllreduceFP32DDPGradBucketGradAccum, self).setUp()
        num_steps = 3
        config = SMPTestConfig(
            grad_atol=5e-3,
            grad_rtol=1e-3,
            loss_atol=1e-4,
            num_steps=num_steps,
            batch_size=160,
            smp_config={"microbatches": 4, "_fp32_grad_accumulation": True, "fp16": True},
            smp_dist_model_kwargs={
                "overlapping_allreduce": True,
                "gradient_as_bucket_view": True,
                "backward_passes_per_step": num_steps,
            },
            upscale_model_output=True,
        )
        self.set_test_config(config)


class TestAllreduceFP32DDPGradBucketCheckpointing(TestCheckpointing):
    def setUp(self):
        super(TestAllreduceFP32DDPGradBucketCheckpointing, self).setUp()
        config = SMPTestConfig(
            grad_atol=5e-3,
            grad_rtol=1e-3,
            batch_size=160,
            smp_config={"_fp32_grad_accumulation": True, "fp16": True},
            smp_dist_model_kwargs={"overlapping_allreduce": True, "gradient_as_bucket_view": True},
            upscale_model_output=True,
        )
        self.set_test_config(config)
        self.register_begin_hook(self.add_fp16_module_hook)

    def add_fp16_module_hook(self):
        for checkpointing_config in self.current_model.smp_activation_checkpointing_config:
            # Add the fp16 module
            module_name = checkpointing_config[0]
            checkpointing_config[0] = module_name.replace("main", "main/module")


if __name__ == "__main__":
    classes_to_run = []
    if len(sys.argv) == 1:
        to_search = "testallreduce"

        for clsname in dir():
            if clsname.lower().startswith(to_search):
                classes_to_run.append(clsname)
        print("Running tests:", classes_to_run)
        argv = sys.argv[:1]
    else:
        # default unittest behavior
        classes_to_run = None
        argv = sys.argv
    unittest.main(defaultTest=classes_to_run, verbosity=1, argv=argv)
