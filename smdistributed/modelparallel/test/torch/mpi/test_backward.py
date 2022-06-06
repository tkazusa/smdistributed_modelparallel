# Standard Library
import os
import unittest
from contextlib import contextmanager

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_original
from torch.utils.checkpoint import checkpoint_sequential as checkpoint_sequential_original

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.mpi.utils import create_bert_like_model
from smdistributed.modelparallel.test.torch.utils import ATOL, RTOL, FP16_Module, add_num_grads_hook
from smdistributed.modelparallel.torch.patches.checkpoint import checkpoint, checkpoint_sequential


@smp.step
def train(model, target, *args):
    output = model(*args)
    loss = F.nll_loss(output, target)
    model.backward(loss)


@smp.step
def train_bert(model, target, *args):
    output = model(*args)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    a, b = output
    loss = loss_fn(a, target) + loss_fn(b, target)
    model.backward(loss)


def train_bert_nosmp(model, target, *args):
    output = model(*args)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    a, b = output
    loss = loss_fn(a, target) + loss_fn(b, target)
    loss.backward()


def train_nosmp(model, target, *args):
    output = model(*args)
    loss = F.nll_loss(output, target)
    loss.backward()


class TestBackwardBase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.batch_dim = 20
        self.num_microbatches = 2
        self.backward_passes_per_step = 1
        self.fp32_grad_accumulation = False
        self.module_base_name = "main"

        self.allreduce_engine = "ddp"

    def tearDown(self):
        smp.barrier()

    def _move_to_device(self, model, target, args):
        model.to(smp.local_rank())
        target = target.to(smp.local_rank())
        args_device = []
        for arg in args:
            arg = arg.to(smp.local_rank())
            args_device.append(arg)
        return target, args_device

    def _verify_grad_counts(self, model, num_hook_calls, expected_grad_counts):
        for n, p in model.named_parameters():
            if p.requires_grad:
                s = 0
                for i in range(self.num_microbatches):
                    p_name = smp.state.model.get_param_name(p)
                    s += expected_grad_counts[p_name][i]
                s *= self.backward_passes_per_step
                assert s == num_hook_calls[n], (n, s, num_hook_calls[n])

    def _collect_grads(self, model):
        grads = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                grads[n] = p.grad.clone().detach()
                p.grad.zero_()
        if isinstance(model, smp.DistributedModel):
            assert len(grads) == len(list(model.local_parameters()))
        else:
            assert len(grads) == len(list(model.parameters()))
        return grads

    def _verify_grads(self, grads, smp_grads):
        s = 0
        local_param_names = set([n for (n, p) in smp.state.model.local_named_parameters()])
        for n, grad in grads.items():
            smp_name = n
            if smp_name in local_param_names:
                s += 1
                # as smp_grads will have subset
                smp_grad = smp_grads[smp_name]
                global RTOL
                global ATOL
                if self.fp32_grad_accumulation:
                    grad = grad.to(torch.float32)
                    # Requires reduced tolerance check, not sure why since computation
                    # is exactly the same.
                    ATOL = 1e-4
                    RTOL = 1e-3
                if not torch.allclose(smp_grad.cpu(), grad.cpu(), rtol=RTOL, atol=ATOL):
                    raise ValueError(
                        smp.rank(), smp_name, grad, smp_grad, grad.sum(), smp_grad.sum()
                    )

        assert s == len(local_param_names)

    def _init_smp(self, partitions, autopartitioning, model_class, offload=False):

        if isinstance(model_class, nn.Module):
            assert (
                autopartitioning
            ), "Model needs to be created after initialization for manual partitioning"

        smp.init(
            {
                "microbatches": self.num_microbatches,
                "pipeline": "interleaved",
                "partitions": partitions,
                "auto_partition": autopartitioning,
                "default_partition": 0,
                "offload_activations": offload,
                "_fp32_grad_accumulation": self.fp32_grad_accumulation,
                "fp16_params": self.fp32_grad_accumulation,
                self.allreduce_engine: True,
            }
        )
        if self.batch_dim % (smp.dp_size() * self.num_microbatches):
            raise RuntimeError(
                "Ensure batch size is multiple of these for correct batch splitting which can affect grad check if a sample in the batch is ignored."
            )

    def _create_inputs(self, input_dim):
        x = torch.randn(self.batch_dim, input_dim)
        target = torch.randint(low=0, high=3, size=(self.batch_dim,))
        return x, target

    def _create_model(self, model):
        if not isinstance(model, nn.Module):
            model = model()
        model.to(smp.local_rank())
        if self.fp32_grad_accumulation:
            model = FP16_Module(model)
        return model

    def _run_without_smp(self, model, args_device, target, step_fns):
        # run without smp
        num_hook_calls, handles = add_num_grads_hook(model)
        step_fns[0](model, target, *args_device)
        grads = self._collect_grads(model)
        for h in handles:
            h.remove()
        return grads

    @contextmanager
    def _with_empty_clear_minibatch_fn(self, enabled=True):
        original_fn = smp.state.clear_minibatch_state
        if enabled:
            smp.state.clear_minibatch_state = lambda *x: None
        yield
        smp.state.clear_minibatch_state = original_fn
        smp.state.clear_minibatch_state()

    def _run_with_smp(
        self,
        model,
        args_device,
        target,
        step_fns,
        checkpointing=False,
        module_config_to_checkpoint_with_non_torch_api=None,
        collect_grads=True,
    ):
        if isinstance(model, smp.DistributedModel):
            model.module.module.checkpointing = checkpointing
        else:
            model.checkpointing = checkpointing
            model = smp.DistributedModel(model)

        if module_config_to_checkpoint_with_non_torch_api:
            mod, confs = module_config_to_checkpoint_with_non_torch_api
            smp.set_activation_checkpointing(smp.state.module_manager.get_module(mod), **confs)

        num_hook_calls, handles = add_num_grads_hook(model)
        with self._with_empty_clear_minibatch_fn():
            step_fns[1](model, target, *args_device)

            if collect_grads:
                smp_grads = self._collect_grads(model)
                for h in handles:
                    h.remove()
            else:
                smp_grads = None
            expected_grad_counts = smp.state.model.grad_counter.get_expected_param_grad_counts()
        return num_hook_calls, model, smp_grads, expected_grad_counts

    def _assert_num_params(self, model, partitions, num_partitions_with_params):
        if num_partitions_with_params is None:
            num_partitions_with_params = partitions

        if partitions > 1 and num_partitions_with_params == partitions:
            assert len(set([x.device for x in model.parameters()])) == partitions, len(
                set([x.device for x in model.parameters()])
            )
            assert len(set([x.device for x in model.local_parameters()])) == 1, len(
                set([x.device for x in model.local_parameters()])
            )

    def check_grads_and_counts(
        self,
        model_class,
        args,
        target,
        step_fns=(train_nosmp, train),
        autopartitioning=False,
        partitions=2,
        num_partitions_with_params=None,
    ):
        self._init_smp(partitions, autopartitioning, model_class)

        model = self._create_model(model_class)
        target, args_device = self._move_to_device(model, target, args)
        grads = self._run_without_smp(model, args_device, target, step_fns)

        num_hook_calls, model, smp_grads, expected_grad_counts = self._run_with_smp(
            model, args_device, target, step_fns
        )
        # without this test might pass even if no grad was compared
        self._assert_num_params(model, partitions, num_partitions_with_params)
        self._verify_grad_counts(model, num_hook_calls, expected_grad_counts)
        self._verify_grads(grads, smp_grads)


class TestGrad(TestBackwardBase):
    def test_grad_count_simple(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.b = nn.Linear(4, 4)
                self.c = nn.Linear(4, 4)

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                x = self.b(x)
                x = self.c(x)
                x = F.relu(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(Model, (x,), target, autopartitioning=False)

    def test_grad_count_nest0(self):
        class Nest1(nn.Module):
            def __init__(self):
                super(Nest1, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = Nest1()

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                return x

        x, target = self._create_inputs(10)
        self.check_grads_and_counts(Nested, (x,), target, autopartitioning=False)

    def test_grad_count_nest1(self):
        class Nest1(nn.Module):
            def __init__(self):
                super(Nest1, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

        class Nest2(nn.Module):
            def __init__(self):
                super(Nest2, self).__init__()
                with smp.partition(1):
                    self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

        class Nested(nn.Module):
            def __init__(self):
                super(Nested, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.nest1 = Nest1()
                self.nest2 = Nest2()

            def forward(self, x):
                x = self.a(x)
                x = self.nest1(x)
                x = self.nest2(x)
                return x

        x, target = self._create_inputs(10)
        self.check_grads_and_counts(Nested, (x,), target, autopartitioning=False)

    def test_grad_count_nest2(self):
        class Nest1(nn.Module):
            def __init__(self):
                super(Nest1, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

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
                    self.b = Nest1()
                with smp.partition(1):
                    self.c = Nest2()

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                x = self.c(x)
                return x

        x, target = self._create_inputs(10)
        self.check_grads_and_counts(Nested, (x,), target, autopartitioning=False)

    def test_grad_count_chain(self):
        class Chain(nn.Module):
            def __init__(self):
                super(Chain, self).__init__()
                self.a = nn.Linear(10, 10)
                with smp.partition(1):
                    self.b = nn.Linear(10, 10)
                with smp.partition(1):
                    self.c = nn.Linear(10, 10)
                self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return x

        x, target = self._create_inputs(10)
        self.check_grads_and_counts(Chain, (x,), target, autopartitioning=False)

    def test_grad_count_allinonepartition(self):
        class Chain(nn.Module):
            def __init__(self):
                super(Chain, self).__init__()
                self.a = nn.Linear(10, 10)
                self.b = nn.Linear(10, 10)
                self.c = nn.Linear(10, 10)
                self.d = nn.Linear(10, 10)

            def forward(self, x):
                x = self.a(x)
                x = self.b(x)
                x = self.c(x)
                x = self.d(x)
                return x

        x, target = self._create_inputs(10)
        self.check_grads_and_counts(
            Chain, (x,), target, autopartitioning=False, num_partitions_with_params=1
        )

    def test_bert_like_model(self, use_sequential=False):
        x = torch.randint(low=0, high=5, size=(self.batch_dim, 10))
        y = torch.randint(low=0, high=5, size=(self.batch_dim, 10))
        z = torch.randn((self.batch_dim, 10, 10))
        inputs = (x, y, z)
        target = torch.randint(low=0, high=3, size=(self.batch_dim, 10))
        model = create_bert_like_model(
            use_sequential=use_sequential, activation_checkpointing=False, strategy="each"
        )
        self.check_grads_and_counts(
            model, inputs, target, step_fns=(train_bert_nosmp, train_bert), autopartitioning=True
        )

    def test_bert_like_model_seq(self):
        self.test_bert_like_model(use_sequential=True)


class TestCheckpointingBase(TestBackwardBase):
    def check_grads_and_counts(
        self,
        model,
        args,
        target,
        step_fns=(train_nosmp, train),
        autopartitioning=False,
        partitions=2,
        num_partitions_with_params=None,
        module_config_to_checkpoint_with_non_torch_api=None,
    ):
        self._init_smp(partitions, autopartitioning, model)

        model = self._create_model(model)
        target, args_device = self._move_to_device(model, target, args)

        model.checkpointing = False
        grads_without_ckpt = self._run_without_smp(model, args_device, target, step_fns)

        model.checkpointing = True
        grads_with_ckpt = self._run_without_smp(model, args_device, target, step_fns)

        # Need this if condition here since without this the test tries to mark variable ready twice
        # and errors out. Its likely we are hitting case 2 mentioned below.
        # RuntimeError: Expected to mark a variable ready only once. This error is caused by one of the following reasons: 1) Use of a module parameter outside the `forward` function. Please make sure model parameters are not shared across multiple concurrent forward-backward passes2) Reused parameters in multiple reentrant backward passes. For example, if you use multiple `checkpoint` functions to wrap the same part of your model, it would result in the same set of parameters been used by different reentrant backward passes multiple times, and hence marking a variable ready multiple times. DDP does not support such use cases yet.3) Incorrect unused parameter detection. The return value of the `forward` function is inspected by the distributed data parallel wrapper to figure out if any of the module's parameters went unused. For unused parameters, DDP would not expect gradients from then. However, if an unused parameter becomes part of the autograd graph at a later point in time (e.g., in a reentrant backward when using `checkpoint`), the gradient will show up unexpectedly. If all parameters in the model participate in the backward pass, you can disable unused parameter detection by passing the keyword argument `find_unused_parameters=False` to `torch.nn.parallel.DistributedDataParallel`.

        if not self.fp32_grad_accumulation:
            for n, g in grads_without_ckpt.items():
                gc = grads_with_ckpt[n]
                assert torch.allclose(g.cpu(), gc.cpu(), rtol=RTOL, atol=ATOL), (n, g, gc)

            num_hook_calls_without_ckpt, model, smp_grads_without_ckpt, expected_grad_counts = self._run_with_smp(
                model, args_device, target, step_fns, checkpointing=False
            )
            self._verify_grad_counts(model, num_hook_calls_without_ckpt, expected_grad_counts)
            self._verify_grads(grads_without_ckpt, smp_grads_without_ckpt)

        num_hook_calls_with_ckpt, model, smp_grads_with_ckpt, expected_grad_counts = self._run_with_smp(
            model,
            args_device,
            target,
            step_fns,
            checkpointing=True,
            module_config_to_checkpoint_with_non_torch_api=module_config_to_checkpoint_with_non_torch_api,
        )
        self._verify_grad_counts(model, num_hook_calls_with_ckpt, expected_grad_counts)

        # without this test might pass even if no grad was compared
        self._assert_num_params(model, partitions, num_partitions_with_params)

        self._verify_grads(grads_with_ckpt, smp_grads_with_ckpt)


class TestCheckpointing(TestCheckpointingBase):
    def test_remote_checkpoint(self, torch_api=True):
        """
        Checkpointing a remote module
        """

        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.b = nn.Linear(4, 4)
                self.c = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.b, x)
                    else:
                        x = checkpoint_original(self.b, x)
                else:
                    x = self.b(x)
                x = self.c(x)
                x = F.relu(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Model,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/b",
                {},
            ),
        )

    def test_remote_checkpoint_nontorch_api(self):
        self.test_remote_checkpoint(False)

    def test_shared_weight_local_ckpt(self, torch_api=True):
        """
        Checkpointing a local module which shares weight.
        Also there are two paths in the graph for added complexity.
        """

        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Linear(4, 4)
                self.c = nn.Linear(4, 4)
                self.c.weight = self.a.weight
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                x1 = self.b(x)
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.c, x1)
                    else:
                        x = checkpoint_original(self.c, x1)
                else:
                    x = self.c(x1)
                x = x1 + x
                x = F.relu(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Model,
            (x,),
            target,
            autopartitioning=False,
            num_partitions_with_params=1,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/c",
                {},
            ),
        )

    def test_shared_weight_local_ckpt_nontorch_api(self):
        self.test_shared_weight_local_ckpt(False)

    def test_local_ckpt_before_remote(self, torch_api=True):
        """
        Checkpointing local module before a remote module execution
        """

        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Linear(4, 4)
                with smp.partition(1):
                    self.c = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.b, x)
                    else:
                        x = checkpoint_original(self.b, x)
                else:
                    x = self.b(x)
                x = self.c(x)
                x = F.relu(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Model,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/b",
                {},
            ),
        )

    def test_local_ckpt_before_remote_nontorch_api(self):
        self.test_local_ckpt_before_remote(False)

    def test_local_ckpt_after_remote(self, torch_api=True):
        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.b = nn.Linear(4, 4)
                self.c = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                x = self.b(x)
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.c, x)
                    else:
                        x = checkpoint_original(self.c, x)
                else:
                    x = self.c(x)
                x = F.relu(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Model,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/c",
                {},
            ),
        )

    def test_local_ckpt_after_remote_nontorch_api(self):
        self.test_local_ckpt_after_remote(False)

    def test_remote_inner_ckpt(self, torch_api=True):
        class Nest1(nn.Module):
            def __init__(self, checkpointing=False):
                super(Nest1, self).__init__()
                self.m = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.m, x)
                    else:
                        x = checkpoint_original(self.m, x)
                else:
                    x = self.m(x)
                x = F.relu(x)
                return x

        class Nested(nn.Module):
            def __init__(self, checkpointing=False):
                super(Nested, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.nest = Nest1(checkpointing)
                self.c = nn.Linear(4, 4)
                self.activation_checkpointing = checkpointing

            @property
            def checkpointing(self):
                return self.activation_checkpointing

            @checkpointing.setter
            def checkpointing(self, b):
                self.activation_checkpointing = b
                self.nest.checkpointing = b

            def forward(self, x):
                x = self.a(x)
                x = self.nest(x)
                x = self.c(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Nested,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/nest/m",
                {},
            ),
        )

    def test_remote_inner_ckpt_nontorch_api(self):
        self.test_remote_inner_ckpt(False)

    def test_remote_remote_inner_ckpt(self, torch_api=True):
        class Nest1(nn.Module):
            def __init__(self, checkpointing=False):
                super(Nest1, self).__init__()
                with smp.partition(0):
                    self.m = nn.Linear(4, 4)
                self.n = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.m, x)
                    else:
                        x = checkpoint_original(self.m, x)
                else:
                    x = self.m(x)
                x = self.n(x)
                return x

        class Nested(nn.Module):
            def __init__(self, checkpointing=False):
                super(Nested, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.nest = Nest1(checkpointing)
                self.c = nn.Linear(4, 4)
                self.activation_checkpointing = checkpointing

            @property
            def checkpointing(self):
                return self.activation_checkpointing

            @checkpointing.setter
            def checkpointing(self, b):
                self.activation_checkpointing = b
                self.nest.checkpointing = b

            def forward(self, x):
                x = self.a(x)
                x = self.nest(x)
                x = self.c(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Nested,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/nest/m",
                {},
            ),
        )

    def test_remote_remote_inner_ckpt_nontorch_api(self):
        self.test_remote_remote_inner_ckpt(False)

    def test_sequential(self, torch_api=True):
        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.b = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)

                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint_sequential(self.b, x)
                    else:
                        x = checkpoint_sequential_original(self.b, 1, x)
                else:
                    x = self.b(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Model,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/b",
                {},
            ),
        )

    def test_nograd(self):
        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.b = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                with torch.no_grad():
                    if self.checkpointing:
                        if smp.state.model is not None:
                            x = checkpoint(self.b, x)
                    else:
                        x = self.b(x)
                return x

        self._init_smp(2, False, Model)
        x, target = self._create_inputs(4)
        model = Model()
        step_fns = (train_nosmp, train)
        target, args_device = self._move_to_device(model, target, (x,))
        self._run_with_smp(
            model, args_device, target, step_fns, checkpointing=True, collect_grads=False
        )

    def test_sequential_nograd(self):
        """
        To verify that we dont fail due to backward break error when not running in grad mode
        """

        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.b = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                with torch.no_grad():
                    if self.checkpointing:
                        if smp.state.model is not None:
                            x = checkpoint_sequential(self.b, x)
                        else:
                            x = checkpoint_sequential_original(self.b, 1, x)
                    else:
                        x = self.b(x)
                return x

        self._init_smp(2, False, Model)
        x, target = self._create_inputs(4)
        model = Model()
        step_fns = (train_nosmp, train)
        target, args_device = self._move_to_device(model, target, (x,))
        self._run_with_smp(
            model, args_device, target, step_fns, checkpointing=True, collect_grads=False
        )

    def test_sequential_nontorch_api(self):
        self.test_sequential(False)

    def test_local_sequential_before_remote(self, torch_api=True):
        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
                with smp.partition(1):
                    self.c = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint_sequential(self.b, x)
                    else:
                        x = checkpoint_sequential_original(self.b, 1, x)
                else:
                    x = self.b(x)
                x = self.c(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Model,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/b",
                {},
            ),
        )

    def test_local_sequential_before_remote_nontorch_api(self):
        self.test_local_sequential_before_remote(False)

    def test_local_sequential_after_remote(self, torch_api=True):
        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.c = nn.Linear(4, 4)
                self.b = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                x = self.c(x)
                if self.checkpointing and torch_api:
                    if smp.state.model is not None:
                        x = checkpoint_sequential(self.b, x)
                    else:
                        x = checkpoint_sequential_original(self.b, 1, x)
                else:
                    x = self.b(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Model,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=(
                f"{self.module_base_name}/module/module/b",
                {},
            ),
        )

    def test_local_sequential_after_remote_nontorch_api(self):
        self.test_local_sequential_after_remote(False)

    def test_bert_like_model(
        self, use_sequential=False, strategy="each", torch_api=True, checkpoint_style="regular"
    ):
        x = torch.randint(low=0, high=5, size=(self.batch_dim, 10))
        y = torch.randint(low=0, high=5, size=(self.batch_dim, 10))
        z = torch.randn((self.batch_dim, 10, 10))
        inputs = (x, y, z)
        target = torch.randint(low=0, high=3, size=(self.batch_dim, 10))
        model = create_bert_like_model(
            use_sequential=use_sequential,
            strategy=strategy,
            torch_api=torch_api,
            checkpoint_style=checkpoint_style,
        )
        if not torch_api and use_sequential:
            module_config_to_checkpoint_with_non_torch_api = (
                f"{self.module_base_name}/module/module/bert/encoder",
                {"strategy": strategy, "pack_args_as_tuple": True},
            )
        elif not torch_api and not use_sequential:
            module_config_to_checkpoint_with_non_torch_api = (
                f"{self.module_base_name}/module/module/bert/pooler",
                {},
            )
        elif torch_api:
            module_config_to_checkpoint_with_non_torch_api = None

        self.check_grads_and_counts(
            model,
            inputs,
            target,
            step_fns=(train_bert_nosmp, train_bert),
            autopartitioning=True,
            module_config_to_checkpoint_with_non_torch_api=module_config_to_checkpoint_with_non_torch_api,
        )

    def test_bert_like_model_nontorch_api(self):
        self.test_bert_like_model(use_sequential=False, torch_api=False)

    def test_bert_like_model_seq(self):
        self.test_bert_like_model(use_sequential=True, strategy="each")

    def test_bert_like_model_seq_nontorch_api(self):
        self.test_bert_like_model(use_sequential=True, strategy="each", torch_api=False)

    def test_bert_like_model_seq_group2(self):
        self.test_bert_like_model(use_sequential=True, strategy="group_2")

    def test_bert_like_model_seq_group2_nontorch_api(self):
        self.test_bert_like_model(use_sequential=True, strategy="group_2", torch_api=False)

    def test_bert_like_model_seq_contiguous(self):
        self.test_bert_like_model(use_sequential=True, strategy="contiguous")

    def test_bert_like_model_seq_contiguous_nontorch_api(self):
        self.test_bert_like_model(use_sequential=True, strategy="contiguous", torch_api=False)

    def test_bert_like_model_seq_nonseq_ckpt(self):
        """
        This test is to verify that we don't checkpoint a module split across partitions.
        With autopartitioning, we just raise warning and skip checkpointing as user
        can't predict which module is partitioned how.
        If we actually try to checkpoint this module, it would crash as pack_args_as_tuple is not set
        """
        self.test_bert_like_model(
            use_sequential=True, strategy="each", checkpoint_style="non_sequential"
        )


class TestCheckpointingOffload(TestCheckpointing):
    def _init_smp(self, partitions, autopartitioning, model_class, offload=True):
        super(TestCheckpointingOffload, self)._init_smp(
            partitions, autopartitioning, model_class, offload
        )


@unittest.skipIf(
    int(os.getenv("RUN_FAIL_TESTS", 0)) < 1,
    "skips because this will fail. this test is run independently with execute_xfail in CI",
)
class TestCheckpointFailures(TestCheckpointingBase):
    def tearDown(self):
        # remove barrier so process can exit after failing
        pass

    def test_remote_inner_ckpt_failure(self):
        class Nest1(nn.Module):
            def __init__(self, checkpointing=False):
                super(Nest1, self).__init__()
                with smp.partition(0):
                    self.m = nn.Linear(4, 4)
                self.n = nn.Linear(4, 4)
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.m(x)
                x = F.relu(x)
                x = self.n(x)
                return x

        class Nested(nn.Module):
            def __init__(self, checkpointing=False):
                super(Nested, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.nest = Nest1(checkpointing)
                self.activation_checkpointing = checkpointing

            @property
            def checkpointing(self):
                return self.activation_checkpointing

            @checkpointing.setter
            def checkpointing(self, b):
                self.activation_checkpointing = b
                self.nest.checkpointing = b

            def forward(self, x):
                x = self.a(x)
                x = self.nest(x)
                return x

        x, target = self._create_inputs(4)
        self.check_grads_and_counts(
            Nested,
            (x,),
            target,
            autopartitioning=False,
            module_config_to_checkpoint_with_non_torch_api=("main/module/module/nest", {}),
        )

    def test_seq_backward_break(self):
        class Nest(nn.Module):
            def __init__(self):
                super(Nest, self).__init__()
                self.a = nn.Linear(4, 4)

            def forward(self, x):
                with torch.no_grad():
                    return self.a(x)

        class Model(nn.Module):
            def __init__(self, checkpointing=False):
                super(Model, self).__init__()
                self.a = nn.Linear(4, 4)
                with smp.partition(1):
                    self.b = nn.Sequential(nn.Linear(4, 4), Nest(), nn.ReLU())
                self.checkpointing = checkpointing

            def forward(self, x):
                x = self.a(x)
                x = F.relu(x)
                if self.checkpointing:
                    x = checkpoint_sequential(self.b, x, pack_args_as_tuple=False)
                else:
                    x = self.b(x)
                return x

        self._init_smp(2, False, Model)
        x, target = self._create_inputs(4)
        model = Model()
        step_fns = (train_nosmp, train)
        target, args_device = self._move_to_device(model, target, (x,))
        with self._with_empty_clear_minibatch_fn():
            num_hook_calls_with_ckpt, model, smp_grads_with_ckpt = self._run_with_smp(
                model, args_device, target, step_fns, checkpointing=True
            )


if __name__ == "__main__":
    unittest.main()
