# Third Party
import torch.nn as nn
import torch.nn.functional as F

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo.smp_test_model import (
    TestModel,
    create_bert_test_model,
)

__all__ = [
    "grad_count_simple",
    "grad_count_nest0",
    "grad_count_nest1",
    "grad_count_nest2",
    "grad_count_chain",
    "grad_count_allinonepartition",
    "bert_like_model_bwd",
    "backward_models_pp2",
]

smp_config = {
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

DEFAULT_BATCH_SIZE = 20


class GradCountSimple(nn.Module):
    def __init__(self):
        super(GradCountSimple, self).__init__()
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


grad_count_simple = TestModel(
    GradCountSimple, input_sizes=[(DEFAULT_BATCH_SIZE, 4)], smp_config=smp_config
)


class GradCountNest0(nn.Module):
    def __init__(self):
        super(GradCountNest0, self).__init__()

        class Nest1(nn.Module):
            def __init__(self):
                super(Nest1, self).__init__()
                self.m = nn.Linear(10, 10)

            def forward(self, x):
                return self.m(x)

        self.a = nn.Linear(10, 10)
        with smp.partition(1):
            self.b = Nest1()

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x


grad_count_nest0 = TestModel(
    GradCountNest0, input_sizes=[(DEFAULT_BATCH_SIZE, 10)], smp_config=smp_config
)


class GradCountNest1(nn.Module):
    def __init__(self):
        super(GradCountNest1, self).__init__()

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

        self.a = nn.Linear(10, 10)
        with smp.partition(1):
            self.nest1 = Nest1()
        self.nest2 = Nest2()

    def forward(self, x):
        x = self.a(x)
        x = self.nest1(x)
        x = self.nest2(x)
        return x


grad_count_nest1 = TestModel(
    GradCountNest1, input_sizes=[(DEFAULT_BATCH_SIZE, 10)], smp_config=smp_config
)


class GradCountNest2(nn.Module):
    def __init__(self):
        super(GradCountNest2, self).__init__()

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


grad_count_nest2 = TestModel(
    GradCountNest2, input_sizes=[(DEFAULT_BATCH_SIZE, 10)], smp_config=smp_config
)


class GradCountChain(nn.Module):
    def __init__(self):
        super(GradCountChain, self).__init__()
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


grad_count_chain = TestModel(
    GradCountChain, input_sizes=[(DEFAULT_BATCH_SIZE, 10)], smp_config=smp_config
)


class GradCountAllinonepartition(nn.Module):
    def __init__(self):
        super(GradCountAllinonepartition, self).__init__()
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


grad_count_allinonepartition = TestModel(
    GradCountAllinonepartition, input_sizes=[(DEFAULT_BATCH_SIZE, 10)], smp_config=smp_config
)

bert_like_model_bwd = create_bert_test_model(
    DEFAULT_BATCH_SIZE,
    smp_config,
    {"use_sequential": False, "strategy": "each", "activation_checkpointing": False},
    extra_smp_config={"auto_partition": True},
)

bert_like_model_seq_bwd = create_bert_test_model(
    DEFAULT_BATCH_SIZE,
    smp_config,
    {"use_sequential": True, "strategy": "each", "activation_checkpointing": False},
    extra_smp_config={"auto_partition": True},
)

backward_models_pp2 = [
    grad_count_simple,
    grad_count_nest0,
    grad_count_nest1,
    grad_count_nest2,
    grad_count_chain,
    grad_count_allinonepartition,
    bert_like_model_bwd,
    bert_like_model_seq_bwd,
]
