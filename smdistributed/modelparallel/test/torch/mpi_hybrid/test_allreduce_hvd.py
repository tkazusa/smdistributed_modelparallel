# Standard Library
import sys
import unittest

# Third Party
import torch

# First Party
from smdistributed.modelparallel.test.torch.mpi_hybrid.test_allreduce_ddp import (
    TestAllreduceBase,
    TestCheckpointing,
    TestGrad,
)


class TestAllreduceHvdBase(TestAllreduceBase):
    def setUp(self):
        torch.manual_seed(2)
        self.batch_dim = 80
        self.num_microbatches = 4
        self.backward_passes_per_step = 1
        self.allreduce_engine = "horovod"
        self.module_base_name = "main"
        self.overlapping_allreduce = True
        self.gradient_as_bucket_view = False
        self.fp32_grad_accumulation = False


class TestAllreduceHvd(TestAllreduceHvdBase, TestGrad):
    pass


class TestAllreduceHvdCheckpointing(TestAllreduceHvdBase, TestCheckpointing):
    pass


class TestAllreduceHvdNonoverlapping(TestAllreduceHvd):
    def setUp(self):
        torch.manual_seed(2)
        self.batch_dim = 80
        self.num_microbatches = 2
        self.backward_passes_per_step = 1
        self.allreduce_engine = "horovod"
        self.module_base_name = "main"
        self.overlapping_allreduce = False
        self.gradient_as_bucket_view = False
        self.fp32_grad_accumulation = False


class TestAllreduceHvdGradAccum(TestAllreduceHvd):
    def setUp(self):
        torch.manual_seed(2)
        self.batch_dim = 160
        self.num_microbatches = 4
        self.backward_passes_per_step = 3
        self.allreduce_engine = "horovod"
        self.overlapping_allreduce = True
        self.gradient_as_bucket_view = False
        self.fp32_grad_accumulation = False


class TestAllreduceHvdGradAccumNonoverlapping(TestAllreduceHvd):
    def setUp(self):
        torch.manual_seed(2)
        self.batch_dim = 160
        self.num_microbatches = 4
        self.backward_passes_per_step = 3
        self.module_base_name = "main"
        self.allreduce_engine = "horovod"
        self.overlapping_allreduce = False
        self.gradient_as_bucket_view = False
        self.fp32_grad_accumulation = False


if __name__ == "__main__":
    classes_to_run = []
    if len(sys.argv) == 1:
        to_search = "testallreducehvd"

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
