# Standard Library
import sys
import unittest

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.mpi.test_backward import (
    TestBackwardBase,
    TestCheckpointing,
    TestGrad,
)
from smdistributed.modelparallel.test.torch.utils import add_num_grads_hook
from smdistributed.modelparallel.torch.step import PTTensorSplitter


class TestAllreduceBase(TestBackwardBase):
    def _run_without_smp(self, model, args_device, target, step_fns):
        # run without smp
        num_hook_calls, handles = add_num_grads_hook(model)
        for i in range(self.backward_passes_per_step):
            step_fns[0](model, target, *args_device)
        grads = self._collect_grads(model)
        for h in handles:
            h.remove()
        return grads

    def _run_with_smp(
        self,
        model,
        args,
        target,
        step_fns,
        checkpointing=False,
        module_config_to_checkpoint_with_non_torch_api=None,
        collect_grads=True,
    ):
        if isinstance(model, smp.DistributedModel):
            if self.allreduce_engine == "horovod":
                model.module.checkpointing = checkpointing
            elif self.allreduce_engine == "ddp":
                model.module.module.checkpointing = checkpointing
        else:
            model.checkpointing = checkpointing
            model = smp.DistributedModel(
                model,
                backward_passes_per_step=self.backward_passes_per_step,
                overlapping_allreduce=self.overlapping_allreduce,
                gradient_as_bucket_view=self.gradient_as_bucket_view,
            )

        if module_config_to_checkpoint_with_non_torch_api:
            mod, confs = module_config_to_checkpoint_with_non_torch_api

            # remove the module wrapper
            # names are for ddp by default, remove the extra module
            if self.allreduce_engine == "horovod":
                mod = "/".join(mod.split("/")[:2] + mod.split("/")[3:])
            smp.set_activation_checkpointing(smp.state.module_manager.get_module(mod), **confs)

        split_arg, split_target = self._split_data_for_dp(step_fns, args, target)
        num_hook_calls, handles = add_num_grads_hook(model)
        for i in range(self.backward_passes_per_step - 1):
            step_fns[1](model, split_target, *split_arg)

        with self._with_empty_clear_minibatch_fn():
            step_fns[1](model, split_target, *split_arg)

            if collect_grads:
                smp_grads = self._collect_grads(model)
            else:
                smp_grads = None

            expected_grad_counts = smp.state.model.grad_counter.get_expected_param_grad_counts()

        for h in handles:
            h.remove()
        return num_hook_calls, model, smp_grads, expected_grad_counts

    def _split_data_for_dp(self, step_fns, args_device, target):
        # split into numdpgroups batches, feed one into each dp group
        # and verify average of grads
        num_dp_groups = smp.dp_size()
        splitter = PTTensorSplitter(step_fns[1])
        split_args, _ = splitter.preprocess_args_all_mbs(args_device, {}, num_dp_groups)
        split_targets, _ = splitter.preprocess_args_all_mbs((target,), {}, num_dp_groups)
        split_arg = split_args[smp.dp_rank()]

        split_target = split_targets[smp.dp_rank()][0]
        return split_arg, split_target


class TestAllreduceDDPBase(TestAllreduceBase):
    def setUp(self):
        super(TestAllreduceDDPBase, self).setUp()
        torch.manual_seed(2)
        self.batch_dim = 80
        self.num_microbatches = 2
        self.backward_passes_per_step = 1
        self.allreduce_engine = "ddp"
        self.overlapping_allreduce = True
        self.gradient_as_bucket_view = False


class TestAllreduceDDP(TestAllreduceDDPBase, TestGrad):
    pass


class TestAllreduceDDPCheckpointing(TestAllreduceDDPBase, TestCheckpointing):
    pass


class TestAllreduceDDPGradBucketBase(TestAllreduceBase):
    def setUp(self):
        super(TestAllreduceDDPGradBucketBase, self).setUp()
        torch.manual_seed(2)
        self.batch_dim = 80
        self.num_microbatches = 2
        self.backward_passes_per_step = 1
        self.allreduce_engine = "ddp"
        self.overlapping_allreduce = True
        self.gradient_as_bucket_view = True


class TestAllreduceDDPGradBucket(TestAllreduceDDPGradBucketBase, TestGrad):
    pass


class TestAllreduceDDPGradBucketCheckpointing(TestAllreduceDDPGradBucketBase, TestCheckpointing):
    pass


class TestAllreduceDDPNonoverlappingBase(TestAllreduceBase):
    def setUp(self):
        super(TestAllreduceDDPNonoverlappingBase, self).setUp()
        torch.manual_seed(2)
        self.batch_dim = 80
        self.num_microbatches = 2
        self.backward_passes_per_step = 1
        self.allreduce_engine = "ddp"
        self.overlapping_allreduce = False
        self.gradient_as_bucket_view = False


class TestAllreduceDDPNonoverlapping(TestAllreduceDDPNonoverlappingBase, TestGrad):
    pass


class TestAllreduceDDPNonoverlappingCheckpointing(
    TestAllreduceDDPNonoverlappingBase, TestCheckpointing
):
    pass


class TestAllreduceDDPGradAccumBase(TestAllreduceBase):
    def setUp(self):
        super(TestAllreduceBase, self).setUp()
        torch.manual_seed(2)
        self.batch_dim = 160
        self.num_microbatches = 4
        self.backward_passes_per_step = 3
        self.allreduce_engine = "ddp"
        self.overlapping_allreduce = True
        self.gradient_as_bucket_view = False


class TestAllreduceDDPGradAccum(TestAllreduceDDPGradAccumBase, TestGrad):
    pass


class TestAllreduceDDPGradAccumCheckpointing(TestAllreduceDDPGradAccumBase, TestCheckpointing):
    pass


class TestAllreduceDDPGradAccumNonoverlapping(TestAllreduceBase, TestGrad):
    def setUp(self):
        super(TestAllreduceBase, self).setUp()
        torch.manual_seed(2)
        self.batch_dim = 160
        self.num_microbatches = 4
        self.backward_passes_per_step = 3
        self.allreduce_engine = "ddp"
        self.overlapping_allreduce = False
        self.gradient_as_bucket_view = False


class TestAllreduceFP32GradBucketBase(TestAllreduceDDPGradBucketBase, TestGrad):
    def setUp(self):
        super(TestAllreduceDDPGradBucketBase, self).setUp()
        self.overlapping_allreduce = True
        self.fp32_grad_accumulation = True


class TestAllreduceFP32NonGradBucketBase(TestAllreduceDDPBase, TestGrad):
    def setUp(self):
        super(TestAllreduceDDPBase, self).setUp()
        self.overlapping_allreduce = True
        self.fp32_grad_accumulation = True


class TestAllreduceFP32DDPGradBucketGradAccum(TestAllreduceDDPGradAccumBase, TestGrad):
    def setUp(self):
        super(TestAllreduceDDPGradAccumBase, self).setUp()
        self.overlapping_allreduce = True
        self.fp32_grad_accumulation = True
        self.gradient_as_bucket_view = True


class TestAllreduceFP32DDPGradBucketCheckpointing(
    TestAllreduceDDPGradBucketBase, TestCheckpointing
):
    def setUp(self):
        super(TestAllreduceDDPGradBucketBase, self).setUp()
        self.module_base_name = "main/module"
        self.overlapping_allreduce = True
        self.fp32_grad_accumulation = True
        self.gradient_as_bucket_view = True


if __name__ == "__main__":
    classes_to_run = []
    if len(sys.argv) == 1:
        to_search = "testallreduceddp"

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
