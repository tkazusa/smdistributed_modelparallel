# Standard Library
import copy
from functools import partial

# Third Party
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import SGD, Adam, Adamax, AdamW, RMSprop

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.mpi.utils import create_bert_like_model
from smdistributed.modelparallel.torch.apex.fp16_utils import FP16_Optimizer
from smdistributed.modelparallel.torch.fp16 import Bit16_Module


@smp.step
def train(model, target, *args):
    output = model(*args)
    loss = F.nll_loss(output, target)
    model.backward(loss)
    return loss


def train_nosmp(model, optimizer, target, *args):
    output = model(*args)
    loss = F.nll_loss(output, target)
    if smp.state.cfg.fp16 and optimizer:
        optimizer.backward(loss)
    else:
        loss.backward()
    return loss


@smp.step
def train_bert(model, target, *args):
    output = model(*args)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    a, b = output
    loss = loss_fn(a, target) + loss_fn(b, target)
    model.backward(loss)
    return loss


def train_bert_nosmp(model, optimizer, target, *args):
    output = model(*args)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    a, b = output
    loss = loss_fn(a, target) + loss_fn(b, target)
    if smp.state.cfg.fp16 and optimizer:
        optimizer.backward(loss)
    else:
        loss.backward()
    return loss


def create_bert_input(self):
    x = torch.randint(
        low=0,
        high=5,
        size=(self.input_sizes[0][0], self.input_sizes[0][1]),
        device=smp.local_rank(),
    )
    y = torch.randint(
        low=0,
        high=5,
        size=(self.input_sizes[0][0], self.input_sizes[0][1]),
        device=smp.local_rank(),
    )
    z = torch.randn(
        (self.input_sizes[0][0], self.input_sizes[0][1], self.input_sizes[0][1]),
        device=smp.local_rank(),
    )
    self.inputs = (x, y, z)
    self.target = torch.randint(
        low=0,
        high=3,
        size=(self.input_sizes[0][0], self.input_sizes[0][1]),
        device=smp.local_rank(),
    )


def create_bert_test_model(
    batch_size,
    base_smp_config,
    model_kwargs=None,
    activation_checkpoint_config=None,
    extra_smp_config=None,
):
    bert_like_model = TestModel(
        create_bert_like_model, input_sizes=[(batch_size, 10)], smp_config=base_smp_config
    )
    bert_like_model.set_step_function(non_smp_func=train_bert_nosmp, smp_func=train_bert)
    bert_like_model.create_inputs = partial(create_bert_input, bert_like_model)
    if model_kwargs != None:
        bert_like_model.update_model_kwargs(**model_kwargs)
    if extra_smp_config != None:
        bert_like_model.update_smp_config(**extra_smp_config)
    if activation_checkpoint_config != None:
        mod, confs = activation_checkpoint_config
        bert_like_model.set_smp_activation_checkpointing_config(module=mod, config=confs)
    return bert_like_model


# Adamax and RMSprop get smaller learning rate because the lack of bias correction in gradient moments
# causes very large updates in the first steps
OPTIMIZERS = {
    "adam": (Adam, {"lr": 1e-3, "weight_decay": 0.0}),
    "adamw": (AdamW, {"lr": 1e-3, "weight_decay": 0.1}),
    "sgd": (SGD, {"lr": 1e-3, "weight_decay": 0.1}),
    "adamax": (Adamax, {"lr": 1e-4, "weight_decay": 0.1}),
    "rmsprop": (RMSprop, {"lr": 1e-4, "weight_decay": 0.0}),
}


class TestModel:
    """
    Model object used as input of SMPTestBase.run_test
    Args:
        model: model class to create torch model
        input_sizes: list or tuple of all input tensor sizes, batch dim should be dim 0
        smp_config: smp initialization configure for this model
        optimizer: optimizer to create torch optimizer, can have three format:
                 - str: Use built-in opt types, including Adam, AdamW, SGD, Adamax, RMSprop
                 - tuple: (torch.optim.optimizer class, dict) where the first element is the opt class and second element is a dict for kwargs
                 - torch.optim.Optimizer class: the opt class, kwargs will be nothing
    """

    def __init__(self, model, input_sizes=None, smp_config=None, optimizer=None):
        if input_sizes == None:
            raise ValueError("input_sizes must be provided for TestModel")
        if smp_config == None:
            smp_config = {}
        self.model_cls = model
        self.optimizer = optimizer
        self.input_sizes = input_sizes
        self.smp_config = copy.copy(smp_config)
        self.non_smp_step_func = train_nosmp
        self.smp_step_func = train
        self.num_steps = 1
        self.model_kwargs = {}
        self.model_args = ()
        self.smp_dist_model_kwargs = {}
        self.smp_activation_checkpointing_config = []
        self.translate_function = None
        self.tensor_parallel_kwargs = {}
        self.reset()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_step_function(self, non_smp_func=None, smp_func=None):
        if non_smp_func != None:
            self.non_smp_step_func = non_smp_func
        if smp_func != None:
            self.smp_step_func = smp_func

    def update_smp_config(self, **kwargs):
        # set args for smp.init
        self.smp_config.update(kwargs)

    def update_batch_size(self, batch_size):
        for idx, input in enumerate(self.input_sizes):
            input = list(input)
            # Dim 0 is expected to be the batch dim
            input[0] = batch_size
            self.input_sizes[idx] = tuple(input)

    def update_num_steps(self, num_steps):
        self.num_steps = num_steps

    def update_smp_dist_model_kwargs(self, **kwargs):
        # set args for smp.DistributedModel
        self.smp_dist_model_kwargs.update(kwargs)

    def set_model_args(self, *args):
        # set args for torch model
        self.model_args = args

    def update_model_kwargs(self, **kwargs):
        self.model_kwargs.update(kwargs)

    def set_smp_activation_checkpointing_config(self, module="", config=None):
        # set args for smp activation checkpointing
        if config == None:
            config = {}
        self.smp_activation_checkpointing_config.append([module, config])

    def set_translate_function(self, translate_function):
        self.translate_function = translate_function

    def update_tensor_parallel_kwargs(self, kwargs):
        self.tensor_parallel_kwargs.update(kwargs)

    def tp_enabled(self):
        return smp.tp_size() > 1 or self.tensor_parallel_kwargs.get("tensor_parallelism", False)

    def create_smp_model(self, dtype=None):
        model_creation_args = {"tensor_parallelism": self.tp_enabled()}
        if dtype is not None:
            model_creation_args["dtype"] = dtype
        model_creation_args.update(self.tensor_parallel_kwargs)
        with smp.model_creation(**model_creation_args):
            self.smp_model = self.model_cls(*self.model_args, **self.model_kwargs)

    def _create_model(self, upscale_model_output=False):
        self.torch_model = self.model_cls(*self.model_args, **self.model_kwargs)
        self.torch_model.to(smp.local_rank())

        bit16_dtype = None
        if smp.state.cfg.fp16:
            bit16_dtype = torch.float16
        elif smp.state.cfg.bf16:
            bit16_dtype = torch.bfloat16

        self.create_smp_model(dtype=bit16_dtype)

        if bit16_dtype is not None:
            # Do not scale model output to reduce precision loss
            self.torch_model = Bit16_Module(
                self.torch_model, dtype=bit16_dtype, upscale_model_output=upscale_model_output
            )

    def match_weights(self):
        # To guarantee that smp model is running with the same weight as non-smp model
        # when tp is enabled and some modules are replaced with their tp counterparts
        state_dict = self.torch_model.state_dict()

        if self.tp_enabled() and self.translate_function != None:
            state_dict = self.translate_function(state_dict)
        if smp.state.cfg.zero2d_enabled():
            # Need to manually translate as auto-translate is only handled in smp's DistributedModel.load_state_dict
            if self.tp_enabled():
                for _, hf_to_smp in smp.state.module_manager.translate_functions:
                    state_dict = hf_to_smp(state_dict)
            # Loading the weights into the torch model for DS case
            if smp.state.cfg.delayed_parameter_initialization:
                group = next(iter(self.smp_model.parameters())).ds_param_shard_group
                state_dict = self.shard_state_dict(state_dict, group)
            self.smp_model.module.load_state_dict(state_dict, strict=True)
            smp.state.no_reinitialization = True
        else:
            # state_dict will be translated automatically inside load_state_dict if necessary
            self.smp_model.load_state_dict(state_dict, strict=True)

    def shard_state_dict(self, state_dict, group):
        def prod(x):
            p = 1
            for i in x:
                p *= i
            return p

        shard_degree = smp.state.cfg.sharded_data_parallel_degree
        sharded_state_dict = {}
        shard_group = group

        for n, p in state_dict.items():
            shard_rank = dist.get_rank(group=shard_group)
            num_elements = prod(p.shape)
            rem = num_elements % shard_degree
            aligned_size = (shard_degree - rem if rem else 0) + num_elements
            shard_size = aligned_size // shard_degree
            shard_length = min(shard_size, aligned_size - (shard_degree - 1) * shard_size)

            if (
                n.endswith(".attn.bias")
                or n.endswith(".attn.masked_bias")
                or n.endswith(".attn.attention.bias")
                or n.endswith(".attn.attention.masked_bias")
                or n.endswith(".attention.bias")
                or n.endswith(".attention.masked_bias")
                or n.endswith(".attention.rotary_emb.inv_freq")
            ):
                sharded_state_dict[n] = p
            else:
                sharded_state_dict[n] = p.reshape([-1]).narrow(
                    0, shard_rank * shard_size, shard_length
                )
        return sharded_state_dict

    def prepare_training(self, upscale_model_output=False):
        self._create_model(upscale_model_output=upscale_model_output)
        self.create_inputs()

    def _get_optimizer_cls(self):
        if isinstance(self.optimizer, str):
            self.optimizer = self.optimizer.lower()
            assert self.optimizer in OPTIMIZERS, f"{self.optimizer} is not a built-in optimizer"
            self.opt_cls, self.opt_kwargs = OPTIMIZERS[self.optimizer]
        elif isinstance(self.optimizer, tuple):
            self.opt_cls, self.opt_kwargs = self.optimizer
        elif issubclass(self.optimizer, torch.optim.Optimizer):
            self.opt_cls, self.opt_kwargs = self.optimizer, {}
        else:
            raise ValueError(f"Unsupported opt_cls {self.opt_cls}")

    def create_optimizer(self, model, run_smp=False):
        self._get_optimizer_cls()

        if not run_smp:
            self.torch_optimizer = self.opt_cls(model.parameters(), **self.opt_kwargs)
            if smp.state.cfg.fp16 or smp.state.cfg.bf16:
                static_loss_scale = 8.0 if smp.state.cfg.fp16 else 1.0
                self.torch_optimizer = FP16_Optimizer(
                    self.torch_optimizer, static_loss_scale=static_loss_scale, verbose=False
                )
        else:
            self.smp_optimizer = smp.DistributedOptimizer(
                self.opt_cls(model.parameters(), **self.opt_kwargs),
                static_loss_scale=8.0,
                dynamic_loss_scale=False,
            )

    def create_inputs(self):
        # Create the inputs and targets
        self.inputs = []
        for shape in self.input_sizes:
            assert isinstance(shape, tuple), "input size must be a tuple"
            if shape[0] % (smp.dp_size() * smp.num_microbatches()) != 0:
                raise RuntimeError(
                    f"{shape} Ensure batch size is multiple of these for correct batch splitting which can affect grad check if a sample in the batch is ignored."
                )
            self.inputs.append(torch.randn(*shape, device=smp.local_rank()))
        self.inputs = tuple(self.inputs)
        self.target = torch.randint(
            low=0, high=3, size=(self.input_sizes[0][0],), device=smp.local_rank()
        )

    def reset(self):
        if smp.is_initialized() and smp.state.cfg.zero2d_enabled():
            model = self.smp_model
            if model is not None:
                for p in model.parameters():
                    if hasattr(p, "ds_tensor"):
                        p.ds_tensor = None
                model._parameters.clear()
                model._parameters._parent_module = None
                for m in model.modules():
                    m._parameters.persistent_param_id_to_partition = None
                    m._forward_hooks.clear()
                    m._forward_pre_hooks.clear()
            from deepspeed.runtime.zero.stage3 import FWD_MODULE_STACK

            FWD_MODULE_STACK.clear()

        # torch module created by the model class
        self.torch_model = None
        # smp module created by the model class
        self.smp_model = None
        # torch optimizer created by the optimizer class
        self.torch_optimizer = None
        # smp optimizer created by the optimizer class
        self.smp_optimizer = None
        # model inputs
        self.inputs = None
        # target to calculate loss
        self.target = None
