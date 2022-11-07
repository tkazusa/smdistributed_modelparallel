# Standard Library
import unittest

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo.smp_test_model import TestModel
from smdistributed.modelparallel.test.torch.smp_test_base import SMPTestBase


def train_nosmp_one_bwd(model, optimizer, target, *args):
    output, y = model(*args)
    loss = F.nll_loss(output, target)
    if smp.state.cfg.fp16 and optimizer:
        optimizer.backward(loss)
    else:
        loss.backward()


@smp.step
def train_one_bwd(model, target, *args):
    x, y = model(*args)
    loss = F.nll_loss(x, target)
    model.backward(loss)


class TestUnused(SMPTestBase):
    def setUp(self):
        super(TestUnused, self).setUp()
        self.smp_config = {
            "microbatches": 2,
            "pipeline": "interleaved",
            "partitions": 2,
            "auto_partition": False,
            "default_partition": 0,
            "offload_activations": False,
            "_fp32_grad_accumulation": False,
            "fp16_params": False,
            "ddp": True,
        }
        self.input_sizes = [(20, 10)]
        self.verify_loss = False

    def tearDown(self):
        # to remove the barrier inherited
        pass

    def test_unused_input(self):
        # remote computed y passed to local child but unused
        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x, y):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                self.c = Nest2()
                with smp.partition(1):
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.b(x)
                x = self.c(x, y)
                x = self.d(x)
                return x

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        self.run_test([model])

    def test_unused_compute(self):
        # remote compute unused
        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                self.c = Nest2()
                with smp.partition(1):
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return x

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        self.run_test([model])

    def test_unused_compute2(self):
        # local compute unused, no error
        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.a(x)
                z = self.b(x)
                x = self.d(z)
                return x

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        self.run_test([model])

    def test_unused_compute3(self):
        # local compute unused, backward called on one output only, no error
        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.a(x)
                z = self.b(x)
                x = self.d(z)
                return x, y

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        model.set_step_function(non_smp_func=train_nosmp_one_bwd, smp_func=train_one_bwd)
        self.run_test([model])

    def test_unused_compute4(self):
        # remote compute unused, but returned as output of main module
        # backward only called on first output, so second output is technically still unused
        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                self.c = Nest2()
                with smp.partition(1):
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return x, y

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        model.set_step_function(non_smp_func=train_nosmp_one_bwd, smp_func=train_one_bwd)
        self.run_test([model])

    def test_used_compute_backward_both(self):
        # remote compute unused in rest of graph, but returned as output of main module
        # backward called on both outputs
        # of course trivial use case, but to assert that we dont fail here
        def train_nosmp_both_bwd(model, optimizer, target, *args):
            output, y = model(*args)
            loss = F.nll_loss(output, target)
            y = y.mean()
            torch.autograd.backward((loss, y), (None, None))

        @smp.step
        def train_both_bwd(model, target, *args):
            x, y = model(*args)
            y = y.mean()
            loss = F.nll_loss(x, target)
            model.backward((y, loss), (None, None))

        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                self.c = Nest2()
                with smp.partition(1):
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return x, y

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        model.set_step_function(non_smp_func=train_nosmp_both_bwd, smp_func=train_both_bwd)
        self.run_test([model])

    def test_unused_input_nograd2(self):
        # remote child produces tensor which doesn't require grad as detached
        # no error
        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()

            def forward(self, x):
                return F.relu(x).detach()

        class Nest3(nn.Module):
            def __init__(self):
                super(Nest3, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x, y):
                return self.m(x) + self.m(y)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                self.c = Nest3()
                with smp.partition(1):
                    self.e = Nest2()
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                y = self.e(x)
                x = self.c(x, y)
                x = self.d(x)
                return x

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        self.run_test([model])

    def test_unused_input_nograd(self):
        # b output is unused but is in no_grad block so no problem
        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x, y):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                self.c = Nest2()
                with smp.partition(1):
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                with torch.no_grad():
                    y = self.b(x)
                x = self.c(x, y)
                x = self.d(x)
                return x

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        self.run_test([model])

    def test_unused_input4(self):
        # remote computed y passed to local child but unused
        # c on remote rank
        # gives MissingPathFromInputToOutput for nest2 error
        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x, y):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)

                with smp.partition(1):
                    self.c = Nest2()
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.b(x)
                x = self.c(x, y)
                x = self.d(x)
                return x

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        self.run_test([model])

    def test_unused_input5_detached(self):
        # remote computed y passed to local child but unused
        # c on remote rank
        # since y is detached before passing to nest 2 it doesn't give MissingPathFromInputToOutput
        # but main module gives MissingPathFromComputationToOutput error
        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x, y):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)

                with smp.partition(1):
                    self.c = Nest2()
                    self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                y = self.b(x)
                y = y.detach()
                x = self.c(x, y)
                x = self.d(x)
                return x

        model = TestModel(Nested, input_sizes=self.input_sizes, smp_config=self.smp_config)
        self.run_test([model])


if __name__ == "__main__":
    unittest.main()
