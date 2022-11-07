# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_original
from torch.utils.checkpoint import checkpoint_sequential as checkpoint_sequential_original

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.model_zoo.smp_test_model import (
    TestModel,
    create_bert_test_model,
)
from smdistributed.modelparallel.torch.patches.checkpoint import checkpoint, checkpoint_sequential

__all__ = [
    "checkpointing_models_pp2",
    "bert_like_model_ckpts",
    "checkpointing_models_no_grad",
    "seq_backward_break",
    "remote_inner_ckpt_failure",
]

activation_checkpointing_smp_config = {
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

ACT_DEFAULT_BATCH_SIZE = 20

"""
Checkpointing a remote module
"""


class RemoteCheckpoint(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(RemoteCheckpoint, self).__init__()
        self.a = nn.Linear(4, 4)
        with smp.partition(1):
            self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)
        self.checkpointing = checkpointing
        self.torch_api = torch_api

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)
        if self.checkpointing and self.torch_api:
            if smp.state.model is not None:
                x = checkpoint(self.b, x)
            else:
                x = checkpoint_original(self.b, x)
        else:
            x = self.b(x)
        x = self.c(x)
        x = F.relu(x)
        return x


remote_checkpoint = TestModel(
    RemoteCheckpoint,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

remote_checkpoint_nontorch_api = TestModel(
    RemoteCheckpoint,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
remote_checkpoint_nontorch_api.update_model_kwargs(torch_api=False)
remote_checkpoint_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/b"
)

"""
Checkpointing a local module which shares weight.
Also there are two paths in the graph for added complexity.
"""


class SharedWeightLocalCkpt(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(SharedWeightLocalCkpt, self).__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)
        self.c.weight = self.a.weight
        self.checkpointing = checkpointing
        self.torch_api = torch_api

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)
        x1 = self.b(x)
        if self.checkpointing and self.torch_api:
            if smp.state.model is not None:
                x = checkpoint(self.c, x1)
            else:
                x = checkpoint_original(self.c, x1)
        else:
            x = self.c(x1)
        x = x1 + x
        x = F.relu(x)
        return x


shared_weight_local_ckpt = TestModel(
    SharedWeightLocalCkpt,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

shared_weight_local_ckpt_nontorch_api = TestModel(
    SharedWeightLocalCkpt,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
shared_weight_local_ckpt_nontorch_api.update_model_kwargs(torch_api=False)
shared_weight_local_ckpt_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/c"
)


class LocalCkptBeforeRemote(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(LocalCkptBeforeRemote, self).__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        with smp.partition(1):
            self.c = nn.Linear(4, 4)
        self.checkpointing = checkpointing
        self.torch_api = torch_api

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)
        if self.checkpointing and self.torch_api:
            if smp.state.model is not None:
                x = checkpoint(self.b, x)
            else:
                x = checkpoint_original(self.b, x)
        else:
            x = self.b(x)
        x = self.c(x)
        x = F.relu(x)
        return x


local_ckpt_before_remote = TestModel(
    LocalCkptBeforeRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

local_ckpt_before_remote_nontorch_api = TestModel(
    LocalCkptBeforeRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
local_ckpt_before_remote_nontorch_api.update_model_kwargs(torch_api=False)
local_ckpt_before_remote_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/b"
)


class LocalCkptAfterRemote(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(LocalCkptAfterRemote, self).__init__()
        self.a = nn.Linear(4, 4)
        with smp.partition(1):
            self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)
        self.checkpointing = checkpointing
        self.torch_api = torch_api

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)
        x = self.b(x)
        if self.checkpointing and self.torch_api:
            if smp.state.model is not None:
                x = checkpoint(self.c, x)
            else:
                x = checkpoint_original(self.c, x)
        else:
            x = self.c(x)
        x = F.relu(x)
        return x


local_ckpt_after_remote = TestModel(
    LocalCkptAfterRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

local_ckpt_after_remote_nontorch_api = TestModel(
    LocalCkptAfterRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
local_ckpt_after_remote_nontorch_api.update_model_kwargs(torch_api=False)
local_ckpt_after_remote_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/c"
)


class RemoteInnerCkpt(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(RemoteInnerCkpt, self).__init__()
        self.a = nn.Linear(4, 4)

        class Nest1(nn.Module):
            def __init__(self, checkpointing=False, torch_api=True):
                super(Nest1, self).__init__()
                self.m = nn.Linear(4, 4)
                self.checkpointing = checkpointing
                self.torch_api = torch_api

            def forward(self, x):
                if self.checkpointing and self.torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.m, x)
                    else:
                        x = checkpoint_original(self.m, x)
                else:
                    x = self.m(x)
                x = F.relu(x)
                return x

        with smp.partition(1):
            self.nest = Nest1(checkpointing=checkpointing, torch_api=torch_api)
        self.c = nn.Linear(4, 4)
        self.activation_checkpointing = checkpointing
        self.torch_api = torch_api

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


remote_inner_ckpt = TestModel(
    RemoteInnerCkpt,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

remote_inner_ckpt_nontorch_api = TestModel(
    RemoteInnerCkpt,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
remote_inner_ckpt_nontorch_api.update_model_kwargs(torch_api=False)
remote_inner_ckpt_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/nest/m"
)


class RemoteRemoteInnerCkpt(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(RemoteRemoteInnerCkpt, self).__init__()
        self.a = nn.Linear(4, 4)

        class Nest1(nn.Module):
            def __init__(self, checkpointing=False, torch_api=True):
                super(Nest1, self).__init__()
                with smp.partition(0):
                    self.m = nn.Linear(4, 4)
                self.n = nn.Linear(4, 4)
                self.checkpointing = checkpointing
                self.torch_api = torch_api

            def forward(self, x):
                if self.checkpointing and self.torch_api:
                    if smp.state.model is not None:
                        x = checkpoint(self.m, x)
                    else:
                        x = checkpoint_original(self.m, x)
                else:
                    x = self.m(x)
                x = self.n(x)
                return x

        with smp.partition(1):
            self.nest = Nest1(checkpointing=checkpointing, torch_api=torch_api)
        self.c = nn.Linear(4, 4)
        self.activation_checkpointing = checkpointing
        self.torch_api = torch_api

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


remote_remote_inner_ckpt = TestModel(
    RemoteRemoteInnerCkpt,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

remote_remote_inner_ckpt_nontorch_api = TestModel(
    RemoteRemoteInnerCkpt,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
remote_remote_inner_ckpt_nontorch_api.update_model_kwargs(torch_api=False)
remote_remote_inner_ckpt_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/nest/m"
)


class SequentialCheckpoint(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(SequentialCheckpoint, self).__init__()
        self.a = nn.Linear(4, 4)
        with smp.partition(1):
            self.b = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.checkpointing = checkpointing
        self.torch_api = torch_api

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)

        if self.checkpointing and self.torch_api:
            if smp.state.model is not None:
                x = checkpoint_sequential(self.b, x)
            else:
                x = checkpoint_sequential_original(self.b, 1, x)
        else:
            x = self.b(x)
        return x


sequential_checkpoint = TestModel(
    SequentialCheckpoint,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

sequential_checkpoint_nontorch_api = TestModel(
    SequentialCheckpoint,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
sequential_checkpoint_nontorch_api.update_model_kwargs(torch_api=False)
sequential_checkpoint_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/b"
)


class LocalSequentialBeforeRemote(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(LocalSequentialBeforeRemote, self).__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        with smp.partition(1):
            self.c = nn.Linear(4, 4)
        self.checkpointing = checkpointing
        self.torch_api = torch_api

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)
        if self.checkpointing and self.torch_api:
            if smp.state.model is not None:
                x = checkpoint_sequential(self.b, x)
            else:
                x = checkpoint_sequential_original(self.b, 1, x)
        else:
            x = self.b(x)
        x = self.c(x)
        return x


local_sequential_before_remote = TestModel(
    LocalSequentialBeforeRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

local_sequential_before_remote_nontorch_api = TestModel(
    LocalSequentialBeforeRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
local_sequential_before_remote_nontorch_api.update_model_kwargs(torch_api=False)
local_sequential_before_remote_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/b"
)


class LocalSequentialAfterRemote(nn.Module):
    def __init__(self, checkpointing=False, torch_api=True):
        super(LocalSequentialAfterRemote, self).__init__()
        self.a = nn.Linear(4, 4)
        with smp.partition(1):
            self.c = nn.Linear(4, 4)
        self.b = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.checkpointing = checkpointing
        self.torch_api = torch_api

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)
        x = self.c(x)
        if self.checkpointing and self.torch_api:
            if smp.state.model is not None:
                x = checkpoint_sequential(self.b, x)
            else:
                x = checkpoint_sequential_original(self.b, 1, x)
        else:
            x = self.b(x)
        return x


local_sequential_after_remote = TestModel(
    LocalSequentialAfterRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

local_sequential_after_remote_nontorch_api = TestModel(
    LocalSequentialAfterRemote,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
local_sequential_after_remote_nontorch_api.update_model_kwargs(torch_api=False)
local_sequential_after_remote_nontorch_api.set_smp_activation_checkpointing_config(
    module="main/module/module/b"
)

bert_like_model_ckpt = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": False,
        "strategy": "each",
        "torch_api": True,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
)

bert_like_model_nontorch_api_ckpt = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": False,
        "strategy": "each",
        "torch_api": False,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
    activation_checkpoint_config=("main/module/module/bert/encoder", None),
)

bert_like_model_seq_ckpt = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": True,
        "strategy": "each",
        "torch_api": True,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
)

bert_like_model_seq_nontorch_api = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": True,
        "strategy": "each",
        "torch_api": False,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
    activation_checkpoint_config=("main/module/module/bert/encoder", {"strategy": "each"}),
)

bert_like_model_seq_group2 = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": True,
        "strategy": "group_2",
        "torch_api": True,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
)

bert_like_model_seq_group2_nontorch_api = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": True,
        "strategy": "group_2",
        "torch_api": False,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
    activation_checkpoint_config=("main/module/module/bert/encoder", {"strategy": "group_2"}),
)

bert_like_model_seq_contiguous = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": True,
        "strategy": "contiguous",
        "torch_api": True,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
)

bert_like_model_seq_contiguous_nontorch_api = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": True,
        "strategy": "contiguous",
        "torch_api": False,
        "checkpoint_style": "regular",
    },
    extra_smp_config={"auto_partition": True},
    activation_checkpoint_config=("main/module/module/bert/encoder", {"strategy": "contiguous"}),
)

"""
This test is to verify that we don't checkpoint a module split across partitions.
With autopartitioning, we just raise warning and skip checkpointing as user
can't predict which module is partitioned how.
"""
bert_like_model_seq_nonseq_ckpt = create_bert_test_model(
    ACT_DEFAULT_BATCH_SIZE,
    activation_checkpointing_smp_config,
    model_kwargs={
        "use_sequential": True,
        "strategy": "each",
        "torch_api": True,
        "checkpoint_style": "non_sequential",
    },
    extra_smp_config={"auto_partition": True},
)

bert_like_model_ckpts = [
    bert_like_model_ckpt,
    bert_like_model_nontorch_api_ckpt,
    bert_like_model_seq_ckpt,
    bert_like_model_seq_nontorch_api,
    bert_like_model_seq_group2,
    bert_like_model_seq_group2_nontorch_api,
    bert_like_model_seq_contiguous,
    bert_like_model_seq_contiguous_nontorch_api,
    bert_like_model_seq_nonseq_ckpt,
]

checkpointing_models_pp2 = [
    remote_checkpoint,
    remote_checkpoint_nontorch_api,
    shared_weight_local_ckpt,
    shared_weight_local_ckpt_nontorch_api,
    local_ckpt_before_remote,
    local_ckpt_before_remote_nontorch_api,
    local_ckpt_after_remote,
    local_ckpt_after_remote_nontorch_api,
    remote_inner_ckpt,
    remote_inner_ckpt_nontorch_api,
    remote_remote_inner_ckpt,
    remote_remote_inner_ckpt_nontorch_api,
    sequential_checkpoint,
    sequential_checkpoint_nontorch_api,
    local_sequential_before_remote,
    local_sequential_before_remote_nontorch_api,
    local_sequential_after_remote,
    local_sequential_after_remote_nontorch_api,
]
checkpointing_models_pp2.extend(bert_like_model_ckpts)


class NoGrad(nn.Module):
    def __init__(self, checkpointing=True):
        super(NoGrad, self).__init__()
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
                    # x = checkpoint_original(self.b, x)
                    x = self.b(x)
            else:
                x = self.b(x)
        return x


no_grad = TestModel(
    NoGrad,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

"""
To verify that we dont fail due to backward break error when not running in grad mode
"""


class SequentialNoGrad(nn.Module):
    def __init__(self, checkpointing=True):
        super(SequentialNoGrad, self).__init__()
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
                    x = self.b(x)
            else:
                x = self.b(x)
        return x


sequential_nograd = TestModel(
    SequentialNoGrad,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)

checkpointing_models_no_grad = [no_grad, sequential_nograd]


class SeqBackwardBreak(nn.Module):
    def __init__(self, checkpointing=True):
        super(SeqBackwardBreak, self).__init__()
        self.a = nn.Linear(4, 4)

        class Nest(nn.Module):
            def __init__(self):
                super(Nest, self).__init__()
                self.a = nn.Linear(4, 4)

            def forward(self, x):
                with torch.no_grad():
                    return self.a(x)

        with smp.partition(1):
            self.b = nn.Sequential(nn.Linear(4, 4), Nest(), nn.ReLU())
        self.checkpointing = checkpointing

    def forward(self, x):
        x = self.a(x)
        x = F.relu(x)
        if self.checkpointing:
            x = checkpoint_sequential(self.b, x)
        else:
            x = self.b(x)
        return x


seq_backward_break = TestModel(
    SeqBackwardBreak,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)


class RemoteInnerCkptFailure(nn.Module):
    def __init__(self, checkpointing=False):
        super(RemoteInnerCkptFailure, self).__init__()
        self.a = nn.Linear(4, 4)

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


remote_inner_ckpt_failure = TestModel(
    RemoteInnerCkptFailure,
    input_sizes=[(ACT_DEFAULT_BATCH_SIZE, 4)],
    smp_config=activation_checkpointing_smp_config,
)
remote_inner_ckpt_failure.set_smp_activation_checkpointing_config(module="main/module/module/nest")
