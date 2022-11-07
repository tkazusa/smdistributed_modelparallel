# Standard Library
import contextlib
import copy
import functools
import gc
import logging
import os
import shutil
import unittest
from contextlib import contextmanager
from functools import partial

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import (
    ATOL,
    CHECKPOINT_ATOL,
    CHECKPOINT_RTOL,
    RTOL,
    add_num_grads_hook,
)
from smdistributed.modelparallel.torch.fp16 import Bit16_Module
from smdistributed.modelparallel.torch.parameter import use_torchdistx
from smdistributed.modelparallel.torch.step import PTTensorSplitter, StepOutput
from smdistributed.modelparallel.torch.utils import slice_tp_tensor


class SMPTestConfig:
    """
    This is the configuration class to store the configuration of the a test for SMPTestBase
    It is used to changing the existing configurations or adding extra configurations.
    Usage:

        >>> def test_something(self):
        >>>     config = SMPTestConfig(...)
        >>>     self.set_test_config(config)

    Args:
        grad_atol (float):
            Atol of the gradient test.
        grad_rtol (float):
            Rtol of the gradient test.
        loss_atol (float):
            Atol of the loss test.
        loss_rtol (float):
            Rtol of the loss test.
        verify_loss (bool):
            Whether or not to verify loss.
        verify_grad_counts (bool):
            Whether or not to verify gradient counts.
        verify_grads (bool):
            Whether or not to verify gradients.
        verify_memory (bool):
            Whether or not to verify memory leak.
        smp_config (dict):
            A dict of smp's config. This will update the existing smp config for the model.
        batch_size (int):
            Batch size for current model test.
        num_steps (int):
            Total steps to run for current model test.
        model_kwargs (dict):
            Kwargs to use during create the torch model. This will update the existing model kwargs.
        model_args (tuple):
            Args to use during create the torch model. This will replace the existing model args.
        smp_dist_model_kwargs (dict):
            Kwargs to use during create the smp DistributedModel. This will update the existing DistributedModel kwargs.
        smp_activation_checkpointing_config (dict):
            A dict of format {"module": str, "config": dict} for activation checkpointing config. Note the module must a str to
            indicate the module full name. if you want to use the real module, register a pre_train_hook for smp test instead. Example:

            >>>  def smp_enable_checkpointing_hook(self):
            >>>    module = self.current_run_model
            >>>    while isinstance(module, (smp.DistributedModel, FP16_Module, DistributedDataParallel)):
            >>>        module = module.module
            >>>    checkpointing_module = module.transformer.seq_layers
            >>>    smp.set_activation_checkpointing(
            >>>        checkpointing_module, pack_args_as_tuple=True, strategy="each"
            >>>    )

        translate_function (function):
            The translate function from non-smp to smp. During testing if tp_size > 1 smp model will load the state_dict for the non-smp with this translate_function to guarantee weights equal
        optimizer (str, tuple, torch.optim.Optimizer class):
            optimizer type for parameter test
        tensor_parallel_kwargs (dict):
            Kwargs to use during create the smp.tensor_parallelism context. This will update the existing smp.tensor_parallelism kwargs.
    """

    def __init__(
        self,
        grad_atol=None,
        grad_rtol=None,
        param_atol=None,
        param_rtol=None,
        loss_atol=None,
        loss_rtol=None,
        verify_loss=None,
        verify_parameters=None,
        verify_grad_counts=None,
        verify_grads=None,
        verify_memory=None,
        smp_config=None,
        batch_size=None,
        num_steps=None,
        model_kwargs=None,
        model_args=None,
        smp_dist_model_kwargs=None,
        smp_activation_checkpointing_config=None,
        translate_function=None,
        optimizer=None,
        tensor_parallel_kwargs=None,
        upscale_model_output=None,
    ):
        self.grad_atol = grad_atol
        self.grad_rtol = grad_rtol
        self.param_atol = param_atol
        self.param_rtol = param_rtol
        self.loss_atol = loss_atol
        self.loss_rtol = loss_rtol
        self.verify_loss = verify_loss
        self.verify_parameters = verify_parameters
        self.verify_grads = verify_grads
        self.verify_grad_counts = verify_grad_counts
        self.verify_memory = verify_memory
        self.smp_config = smp_config
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.model_kwargs = model_kwargs
        self.model_args = model_args
        self.smp_dist_model_kwargs = smp_dist_model_kwargs
        self.smp_activation_checkpointing_config = smp_activation_checkpointing_config
        self.translate_function = translate_function
        self.optimizer = optimizer
        self.tensor_parallel_kwargs = tensor_parallel_kwargs
        self.upscale_model_output=upscale_model_output
        self._validate_config()

    def _validate_config(self):
        if self.smp_config != None:
            assert isinstance(self.smp_config, dict), "smp_config must be a dict"
        if self.model_kwargs != None:
            assert isinstance(self.model_kwargs, dict), "model_kwargs must be a dict"
        if self.model_args != None:
            assert isinstance(self.model_args, tuple), "model_args must be a tuple"
        if self.smp_dist_model_kwargs != None:
            assert isinstance(
                self.smp_dist_model_kwargs, dict
            ), "smp_dist_model_kwargs must be a dict"
        if self.smp_activation_checkpointing_config != None:
            assert isinstance(
                self.smp_activation_checkpointing_config, dict
            ), "smp_activation_checkpointing_config must be a dict"
            assert (
                len(self.smp_activation_checkpointing_config) == 2
                and "module" in self.smp_activation_checkpointing_config
                and "config" in self.smp_activation_checkpointing_config
            ), "smp_activation_checkpointing_config must be a dict contain module and config"
            assert isinstance(
                self.smp_activation_checkpointing_config["module"], str
            ), "smp_activation_checkpointing_config's module must be a str"
        if self.optimizer != None:
            assert (
                isinstance(self.optimizer, str)
                or isinstance(self.optimizer, tuple)
                or isinstance(self.optimizer, type)
            ), "optimizer can only be str, tuple or torch.optim.Optimizer class"
        if self.tensor_parallel_kwargs != None:
            assert isinstance(
                self.tensor_parallel_kwargs, dict
            ), "tensor_parallel_kwargs must be a dict"


class SMPTestBase(unittest.TestCase):
    """
    The base class to run tests. The default way to test will run both non-smp and smp,
    compare the model outputs, gradients and potentially memory usage
    Example usage:

       >>> class MyTest(SMPTestBase):
       >>>     def setUp(self):
       >>>         super(MyTest, self).setUp()
       >>>         # Add extra configs
       >>>         ...
       >>>
       >>>     def test_function(self):
       >>>         self.run_test([model1, model2,...])

    """

    def setUp(self):
        # Basic config setup
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._reset_current_model_status()
        self._reset_current_test_status()

    def tearDown(self):
        smp.barrier()

    def _reset_current_model_status(self):
        """
        Reset the current model status
        If one test contains multiple models, this is called after the test of each model
        """
        torch.manual_seed(2)
        if "current_model" in self.__dict__ and self.current_model != None:
            self.current_model.reset()
        # current TestModel object
        self.current_model = None
        # current running model (torch module or smp.DistributedModel)
        self.current_run_model = None
        # current running step function
        self.current_step_func = None
        # model outputs
        self.current_model_outputs = [None, None]
        # model loss
        self.current_loss = [None, None]
        # model gradients
        self.current_model_grads = [[], []]
        # gradients count checking related values
        self._expected_grad_counts = None
        self._num_hook_calls = None

        self._clean_resources()
        torch.manual_seed(2)

    def _clean_resources(self):
        # clean smp resources
        smp.reset()
        # clean the resources stored in the lru cache
        # when tp is enabled lru cache is used and it will keep module reference
        gc.collect()
        gc.collect()
        gc.collect()
        objects = [i for i in gc.get_objects() if isinstance(i, functools._lru_cache_wrapper)]
        for object in objects:
            object.cache_clear()
        if hasattr(torch.nn.modules.module.Module, "_old_load_from_state_dict"):
            torch.nn.modules.module.Module._load_from_state_dict = (
                torch.nn.modules.module.Module._old_load_from_state_dict
            )

    def _reset_current_test_status(self):
        """
        Reset the current test status
        This will be only called at setUp, i.e. between each test
        """
        self.verify_loss = True
        self.verify_grads = True
        self.verify_grad_counts = True
        self.verify_grad_count = True
        self.verify_parameters = True
        self.verify_memory = True
        self._pre_train_hooks = [[], []]
        self._post_train_hooks = [[], []]
        self._begin_hooks = []
        self._end_hooks = []
        self._training_context = None
        self._run_functions = [None, None]
        self._test_functions = []
        self.grad_atol = ATOL
        self.grad_rtol = RTOL
        self.loss_atol = ATOL
        self.loss_rtol = RTOL
        self.param_atol = ATOL
        self.param_rtol = RTOL
        self.checkpoint_atol = CHECKPOINT_ATOL
        self.checkpoint_rtol = CHECKPOINT_RTOL
        self._allgather_and_average_smp_loss = True
        self.upscale_model_output = False
        self._test_config = None

    def set_test_config(self, config):
        self._test_config = config

    def apply_test_config(self):
        if self._test_config == None:
            return
        config = self._test_config
        if config.grad_atol != None:
            self.grad_atol = config.grad_atol
        if config.grad_rtol != None:
            self.grad_rtol = config.grad_rtol
        if config.param_atol != None:
            self.param_atol = config.param_atol
        if config.param_rtol != None:
            self.param_rtol = config.param_rtol
        if config.loss_atol != None:
            self.loss_atol = config.loss_atol
        if config.loss_rtol != None:
            self.loss_rtol = config.loss_rtol
        if config.verify_loss != None:
            self.verify_loss = config.verify_loss
        if config.verify_parameters != None:
            self.verify_parameters = config.verify_parameters
        if config.verify_grads != None:
            self.verify_grads = config.verify_grads
        if config.verify_grad_counts != None:
            self.verify_grad_counts = config.verify_grad_counts
        if config.verify_memory != None:
            self.verify_memory = config.verify_memory
        if config.smp_config != None:
            self.current_model.update_smp_config(**config.smp_config)
        if config.batch_size != None:
            self.current_model.update_batch_size(config.batch_size)
        if config.num_steps != None:
            self.current_model.update_num_steps(config.num_steps)
        if config.model_kwargs != None:
            self.current_model.update_model_kwargs(**config.model_kwargs)
        if config.model_args != None:
            self.current_model.set_model_args(*config.model_args)
        if config.smp_dist_model_kwargs != None:
            self.current_model.update_smp_dist_model_kwargs(**config.smp_dist_model_kwargs)
        if config.smp_activation_checkpointing_config != None:
            self.current_model.set_smp_activation_checkpointing_config(
                **config.smp_activation_checkpointing_config
            )
        if config.translate_function != None:
            self.current_model.set_translate_function(config.translate_function)
        if config.optimizer != None:
            self.current_model.set_optimizer(config.optimizer)
        if config.tensor_parallel_kwargs != None:
            self.current_model.update_tensor_parallel_kwargs(config.tensor_parallel_kwargs)
        if config.upscale_model_output != None:
            self.upscale_model_output = config.upscale_model_output

    def _init_smp(self):
        # Initialization of Rubik
        smp.init(self.current_model.smp_config)

    def _split_data_for_dp(self):
        # split into numdpgroups batches, feed one into each dp group
        # and verify average of grads
        dp_size = smp.rdp_size() if smp.state.cfg.prescaled_batch else smp.dp_size()
        dp_rank = smp.rdp_rank() if smp.state.cfg.prescaled_batch else smp.dp_rank()

        splitter = PTTensorSplitter(self.current_step_func)
        split_args, _ = splitter.preprocess_args_all_mbs(self.current_model.inputs, {}, dp_size)
        split_targets, _ = splitter.preprocess_args_all_mbs(
            (self.current_model.target,), {}, dp_size
        )
        split_arg = split_args[dp_rank]

        split_target = split_targets[dp_rank][0]
        return split_arg, split_target

    def _init_and_create_smp_model_opt(self):
        self._init_smp()
        self.current_model.create_smp_model()
        self.current_model.smp_model = smp.DistributedModel(
            self.current_model.smp_model, **self.current_model.smp_dist_model_kwargs
        )
        if self.current_model.optimizer is None:
            raise ValueError("Checkpoint test requires an optimizer")
        self.current_model.create_optimizer(self.current_model.smp_model, run_smp=True)

    def _verify_zero_opt_match(self, origin_opt):
        from deepspeed.checkpoint.constants import (
            OPTIMIZER_STATE_DICT,
            FP32_FLAT_GROUPS,
            PARTITION_COUNT,
            ZERO_STAGE,
        )

        origin_state_dict = origin_opt.orig_state_dict()
        loaded_state_dict = self.current_model.smp_optimizer.orig_state_dict()

        # zero-2d always save the loss scale and master parameter
        self.assertTrue(origin_state_dict[ZERO_STAGE] == loaded_state_dict[ZERO_STAGE])
        self.assertTrue(
            origin_state_dict["dynamic_loss_scale"] == loaded_state_dict["dynamic_loss_scale"]
        )
        self.assertTrue(origin_state_dict["overflow"] == loaded_state_dict["overflow"])
        self.assertTrue(origin_state_dict[PARTITION_COUNT] == loaded_state_dict[PARTITION_COUNT])

        self.check_opt_states_match(
            origin_state_dict[OPTIMIZER_STATE_DICT]["state"],
            loaded_state_dict[OPTIMIZER_STATE_DICT]["state"],
        )

        self.assertTrue(
            len(origin_state_dict[FP32_FLAT_GROUPS]) == len(loaded_state_dict[FP32_FLAT_GROUPS])
        )
        for idx, tensor in enumerate(origin_state_dict[FP32_FLAT_GROUPS]):
            self._check_ckpt_tensor_match(
                loaded_state_dict[FP32_FLAT_GROUPS][idx], tensor, "zero master parameter", idx
            )

    def check_opt_states_match(self, origin_states, loaded_states):
        self.assertTrue(len(origin_states) == len(loaded_states))
        for idx, states in origin_states.items():
            for n, p in states.items():
                if isinstance(p, torch.Tensor):
                    self._check_ckpt_tensor_match(
                        loaded_states[idx][n], p, "optimizer states", f"param_{idx}_{n}"
                    )
                else:
                    self.assertTrue(loaded_states[idx][n] == p)

    def _check_ckpt_tensor_match(self, loaded, origin, tensor_type, tensor_name):
        tensor_match = torch.allclose(
            loaded.cpu(), origin.cpu(), rtol=self.checkpoint_rtol, atol=self.checkpoint_atol
        )
        self.assertTrue(
            tensor_match,
            msg=f"{tensor_type} mismatch on rank {smp.rank()}. Name {tensor_name}, loaded {loaded}, origin {origin}",
        )

    def _verify_model_opt_match(self, orgin_module, origin_opt=None):
        # Check model parameters and buffers match
        origin_params = {n: p for n, p in orgin_module.named_parameters()}
        origin_buffers = {n: p for n, p in orgin_module.named_buffers()}
        current_param_names = set()
        current_buffer_names = set()
        # Check local parameter and buffer match
        for n, p in self.current_model.smp_model.module.named_parameters():
            current_param_names.add(n)
            if smp.state.cfg.zero2d_enabled() or self.current_model.smp_model.is_local_parameter(p):
                current_p = p.ds_tensor if smp.state.cfg.zero2d_enabled() else p
                origin_p = (
                    origin_params[n].ds_tensor
                    if smp.state.cfg.zero2d_enabled()
                    else origin_params[n]
                )
                self._check_ckpt_tensor_match(current_p, origin_p, "model parameter", n)
        for n, p in self.current_model.smp_model.module.named_buffers():
            current_buffer_names.add(n)
            if smp.state.cfg.zero2d_enabled() or self.current_model.smp_model.is_local_buffer(p):
                self._check_ckpt_tensor_match(p, origin_buffers[n], "model buffer", n)
        self.assertTrue(
            current_param_names == set(origin_params.keys()),
            msg=f"mismatch parameters for checkpoint loading, origin {set(origin_params.keys())}, loaded {current_param_names}",
        )
        self.assertTrue(
            current_buffer_names == set(origin_buffers.keys()),
            msg=f"mismatch buffers for checkpoint loading, origin {set(origin_buffers.keys())}, loaded {current_buffer_names}",
        )

        # Check optimizer states and master parameters match
        if origin_opt is not None:
            if smp.state.cfg.zero2d_enabled():
                self._verify_zero_opt_match(origin_opt)
            else:
                origin_states = origin_opt.local_optimizer_state_dict()["state"]
                loaded_states = self.current_model.smp_optimizer.local_optimizer_state_dict()[
                    "state"
                ]
                self.check_opt_states_match(origin_states, loaded_states)
                if smp.state.cfg.fp16:
                    for loaded_group, origin_group in zip(
                        self.current_model.smp_optimizer.fp32_from_fp16_groups,
                        origin_opt.fp32_from_fp16_groups,
                    ):
                        for idx, (loaded, origin) in enumerate(zip(loaded_group, origin_group)):
                            self._check_ckpt_tensor_match(loaded, origin, "master parameter", idx)

    def _test_checkpoint(self):
        """
        Test if the checkpoint is loaded correctly. This function is running in the following order:
        - Create a smp model, train 1 step, save both partial and full checkpoint
        - Keep the reference of the origin module and optimizer, release the smp model and reset smp
        - Create a new smp model, load the partial checkpoint and compare the model and optimizer with the original ones
        - Release the smp model and reset smp
        - Create a new smp model, load the full checkpoint and compare the model with original one
        - Delete the saved checkpoint
        This test is designed to simulate the same behavior of save/load in the real training scenario
        """
        checkpoint_folder = "/tmp/checkpoint"

        self._init_and_create_smp_model_opt()
        self.current_model.create_inputs()

        inputs, target = self.current_model.inputs, self.current_model.target

        for i in range(self.current_model.num_steps):
            self.current_model.smp_step_func(self.current_model.smp_model, target, *inputs)
            self.current_model.smp_optimizer.step()

        smp.save_checkpoint(
            checkpoint_folder,
            "test",
            model=self.current_model.smp_model,
            optimizer=self.current_model.smp_optimizer,
        )
        if not smp.state.cfg.zero2d_enabled():
            smp.save_checkpoint(
                checkpoint_folder, "test", partial=False, model=self.current_model.smp_model
            )

        orgin_module = self.current_model.smp_model.module
        origin_opt = self.current_model.smp_optimizer

        if not smp.state.cfg.zero2d_enabled():
            self.current_model.reset()
        self._clean_resources()

        # Test load partial
        self._init_and_create_smp_model_opt()
        smp.resume_from_checkpoint(checkpoint_folder, tag="test")
        if not smp.state.cfg.zero2d_enabled():
            self.current_model.smp_step_func(self.current_model.smp_model, target, *inputs)
        self._verify_model_opt_match(orgin_module, origin_opt=origin_opt)

        self.current_model.reset()
        self._clean_resources()

        # Test load full
        if not smp.state.cfg.zero2d_enabled():
            self._init_and_create_smp_model_opt()
            smp.resume_from_checkpoint(checkpoint_folder, tag="test", partial=False)
            self.current_model.smp_step_func(self.current_model.smp_model, target, *inputs)
            self._verify_model_opt_match(orgin_module)

        # Clean the checkpoints
        if smp.local_rank() == 0:
            for item in os.listdir(checkpoint_folder):
                path = os.path.join(checkpoint_folder, item)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    def _run_with_smp(self, index=1):
        """
        Default run function with smp, this function will do the following:
        - If verify_grad_counts is enabled, record the grads count for 1 step and collect the grads for final step
        - Split the input data across the data parallel ranks
        - Run step function
        - Record the memory usage after the model run
        - Collect the model outputs and average acorss data parallel ranks
        """
        # Need to wrap here otherwise the forward call will be patched
        self.current_model.smp_model = smp.DistributedModel(
            self.current_model.smp_model, **self.current_model.smp_dist_model_kwargs
        )
        # Do not scale model output to reduce precision loss
        if smp.state.cfg.fp16 or smp.state.cfg.bf16:
            iter_model = self.current_model.smp_model
            while not isinstance(iter_model, Bit16_Module):
                iter_model = iter_model.module
            iter_model.upscale_model_output = self.upscale_model_output
        self.current_model.match_weights()
        self.current_run_model = self.current_model.smp_model
        if self.current_model.optimizer != None:
            self.current_model.create_optimizer(self.current_run_model, run_smp=True)

        self.current_step_func = self.current_model.smp_step_func

        if self.verify_grad_counts:

            def get_grad_count(model, optimizer):
                model._grad_count = model.grad_counter.get_expected_param_grad_counts()

            self.current_run_model.register_post_step_hook(get_grad_count)

        # Set activation checkpointing
        for mod, confs in self.current_model.smp_activation_checkpointing_config:
            smp.set_activation_checkpointing(smp.state.module_manager.get_module(mod), **confs)

        for pre_train_hook in self._pre_train_hooks[index]:
            pre_train_hook()

        split_inputs, split_target = self._split_data_for_dp()

        if self.verify_grad_counts:
            self._num_hook_calls, handles = add_num_grads_hook(self.current_run_model)

        for i in range(self.current_model.num_steps):
            self.current_model_outputs[index] = self.current_step_func(
                self.current_run_model, split_target, *split_inputs
            )

        for post_train_hook in self._post_train_hooks[index]:
            post_train_hook()

        if self.verify_loss:
            # Post-process of the model outputs
            if isinstance(self.current_model_outputs[index], StepOutput):
                self.current_loss[index] = self.current_model_outputs[index].reduce_mean()
            else:
                # The first output is expected to be the loss
                if not isintance(self.current_model_outputs[index][0], StepOutput):
                    raise ValueError(
                        "The first output of the step function is supposed to be the loss. Set the self.verify_loss to False if you want to skip the loss test"
                    )
                self.current_loss[index] = self.current_model_outputs[index][0].reduce_mean()

            # Allgather the model outputs across data parallel ranks and average
            # this is required for loss match. Handle carefully if your model output is not only loss
            if self._allgather_and_average_smp_loss:
                self.current_loss[index] = self._allgather_and_average(self.current_loss[index])

        if self.verify_grad_counts or self.verify_grads:
            self.current_model_grads[index] = self._collect_grads(self.current_run_model)
            if self.verify_grad_counts:
                self._expected_grad_counts = self.current_run_model._grad_count
                for h in handles:
                    h.remove()

    def _run_without_smp(self, index=0):
        """
        Default run function without smp.
        """
        self.current_run_model = self.current_model.torch_model
        if self.current_model.optimizer != None:
            self.current_model.create_optimizer(self.current_run_model)
        self.current_step_func = self.current_model.non_smp_step_func

        for pre_train_hook in self._pre_train_hooks[index]:
            pre_train_hook()

        for i in range(self.current_model.num_steps):
            self.current_model_outputs[index] = self.current_step_func(
                self.current_run_model,
                self.current_model.torch_optimizer,
                self.current_model.target,
                *self.current_model.inputs,
            )

        for post_train_hook in self._post_train_hooks[index]:
            post_train_hook()

        if self.verify_loss:
            if isinstance(self.current_model_outputs[index], torch.Tensor):
                self.current_loss[index] = self.current_model_outputs[index]
            else:
                if not isintance(self.current_model_outputs[index][0], torch.Tensor):
                    raise ValueError(
                        "The first output of the step function is supposed to be the loss. Set the self.verify_loss to False if you want to skip the loss test"
                    )
                self.current_loss[index] = self.current_model_outputs[index][0]

        if self.verify_grads:
            self.current_model_grads[index] = self._collect_grads(
                self.current_run_model, remove_prefix=True
            )

    def _allgather_and_average(self, tensor):
        # Allgather the model output tensor across the data parallel ranks and averge
        # Allgather for dp group even if the prescaled_batch is enabled since the losses between tp ranks also need to be averaged due to sequence sharding
        group = smp.DP_GROUP
        with torch.no_grad():
            tensors = smp.allgather(tensor.cpu(), group)
            return torch.stack(tensors).mean()

    def _collect_grads(self, model, remove_prefix=False):
        # Collect the gradients
        grads = {}
        if not isinstance(model, smp.DistributedModel) or not smp.state.cfg.zero2d_enabled():
            for n, p in model.named_parameters():
                # remove param name prefix for FP16_Module, we should use removeprefix() when we use python3.9
                if n.startswith("module") and remove_prefix:
                    n = n[7:]
                if p.requires_grad:
                    grads[n] = p.grad.clone().detach()
                    p.grad.zero_()
            if isinstance(model, smp.DistributedModel):
                assert len(grads) == len(list(model.local_parameters()))
            else:
                assert len(grads) == len(list(model.parameters()))
        else:
            name_to_grad_partition = {}
            index = 0
            optimizer = self.current_model.smp_optimizer
            for (_, gs), params in zip(optimizer.averaged_gradients.items(), optimizer.fp16_groups):
                for g, p in zip(gs, params):
                    name = optimizer.index_to_name[index]
                    grads[name] = self._reconstitute_gradient(g, p)
                    index += 1

        return grads

    def _reconstitute_gradient(self, grad, ds_param):
        return self._reconstitute_tensor(grad, ds_param)

    def _reconstitute_param(self, ds_param):
        return self._reconstitute_tensor(ds_param.ds_tensor, ds_param)

    def _reconstitute_tensor(self, tensor, ds_param):
        def prod(x):
            p = 1
            for i in x:
                p *= i
            return p

        optimizer = self.current_model.smp_optimizer
        num_elements = prod(ds_param.ds_shape)
        shard_degree = smp.state.cfg.sharded_data_parallel_degree
        rem = num_elements % shard_degree
        aligned_size = (shard_degree - rem if rem else 0) + num_elements
        shard_size = aligned_size // shard_degree
        shard_group = optimizer.ds_param_shard_group

        full_tensor = torch.empty([aligned_size], device=tensor.device, dtype=tensor.dtype)
        shards = [full_tensor.narrow(0, r * shard_size, shard_size) for r in range(shard_degree)]
        torch.distributed.all_gather(shards, tensor, group=shard_group)

        return full_tensor.view(ds_param.ds_shape)

    def _verify_loss(self):
        # Verify the loss match between Rubik and non-Rubik
        loss_equal = torch.allclose(
            self.current_loss[0].cpu(),
            self.current_loss[1].cpu(),
            rtol=self.loss_rtol,
            atol=self.loss_atol,
        )
        self.assertTrue(
            loss_equal,
            msg=f"Loss mismatch on rank {smp.rank()}. Loss 0 {self.current_loss[0]}, Loss 1 {self.current_loss[1]}",
        )

    def _do_verify_grad_counts(self, model):
        # Verify the gradient count match between Rubik and non-Rubik
        if self._num_hook_calls != None and self._expected_grad_counts != None:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    s = 0
                    for i in range(smp.num_microbatches()):
                        p_name = model.get_param_name(p)
                        s += self._expected_grad_counts[p_name][i]
                    s *= self.current_model.num_steps
                    self.assertTrue(
                        s == self._num_hook_calls[n],
                        msg=f"Grad count not equal on rank {smp.rank()}. Param name {n}, smp count {s}, hook record {self._num_hook_calls[n]}",
                    )

    def _verify_tensor_match(
        self, non_smp_tensors, smp_tensors, non_smp_model, smp_model, tensor_type="grad"
    ):
        """
        Verify tensor match between non-smp and smp. Currently only support gradients and parameters
        """

        if tensor_type != "grad" and tensor_type != "param":
            raise ValueError(f"Unsupported tensor type {tensor_type}")

        # Set compare tolerance
        if tensor_type == "grad":
            rtol = self.grad_rtol
            atol = self.grad_atol
        else:
            rtol = self.param_rtol
            atol = self.param_atol

        # Translate the dict if necessary
        if self.current_model.translate_function != None:
            non_smp_tensors = self.current_model.translate_function(non_smp_tensors)
        else:
            for _, hf_to_smp in smp.state.module_manager.translate_functions:
                non_smp_tensors = hf_to_smp(non_smp_tensors)

        # Get the smp params for slicing
        if tensor_type == "grad":
            smp_params = {}
            for (n, p) in smp_model.local_named_parameters():
                smp_params[n] = p
        else:
            smp_params = smp_tensors

        # Run comparison
        smp_count = 0
        for name, non_smp_tensor in non_smp_tensors.items():
            # Add prefix to non_smp param names to match with smp param names for fp16
            if smp.state.cfg.fp16 or smp.state.cfg.bf16:
                name = name if name.startswith("module") else "module." + name
            if name in smp_tensors:
                smp_count += 1
                smp_tensor = smp_tensors[name]
                smp_param = smp_params[name]
                non_smp_tensor = slice_tp_tensor(smp_param, tensor_to_slice=non_smp_tensor)
                if tensor_type == "grad" and smp.core.cfg._fp32_grad_accumulation:
                    non_smp_tensor = non_smp_tensor.to(torch.float32)
                tensor_match = torch.allclose(
                    smp_tensor.cpu(), non_smp_tensor.cpu(), rtol=rtol, atol=atol
                )
                self.assertTrue(
                    tensor_match,
                    msg=f"{tensor_type} mismatch on rank {smp.rank()}. Param name {name}, non-smp {tensor_type} {non_smp_tensor}, sum {non_smp_tensor.sum()}, smp {tensor_type} {smp_tensor}, sum {smp_tensor.sum()}",
                )

        self.assertTrue(
            smp_count == len(smp_params), msg=f"Missing parameters on rank {smp.rank()}"
        )

    def _verify_parameters(self):
        # Verify the parameters match between Rubik and non-Rubik
        non_smp_model = self.current_model.torch_model
        smp_model = self.current_model.smp_model

        self.current_model.torch_optimizer.step()
        self.current_model.smp_optimizer.step()

        non_smp_params = {}
        for (n, p) in non_smp_model.named_parameters():
            # remove param name prefix for FP16_Module, we should use removeprefix() when we use python3.9
            if n.startswith("module"):
                n = n[7:]
            non_smp_params[n] = p

            smp_params = {}
            for (n, p) in smp_model.local_named_parameters():
                if smp.state.cfg.zero2d_enabled():
                    smp_params[n] = self._reconstitute_param(p)
                else:
                    smp_params[n] = p

        self._verify_tensor_match(
            non_smp_params, smp_params, non_smp_model, smp_model, tensor_type="param"
        )

    def _verify_grad_counts(self):
        # Do not verify when using torchdistx as the grad hooks are registered on the fake params
        if not (smp.state.cfg.delayed_parameter_initialization and use_torchdistx):
            self._do_verify_grad_counts(self.current_model.smp_model)

    def _verify_grads(self):
        # Verify the gradients match between Rubik and non-Rubik
        non_smp_model = self.current_model.torch_model
        smp_model = self.current_model.smp_model
        smp_grads = self.current_model_grads[1]
        non_smp_grads = self.current_model_grads[0]

        self._verify_tensor_match(non_smp_grads, smp_grads, non_smp_model, smp_model)

    def _verify_memory(self):
        # Verify there is no memory leak
        self.assertEqual(
            torch.cuda.memory_allocated(smp.local_rank()), 0, f"leak on rank {smp.local_rank()}"
        )

    def register_training_context(self, training_context):
        # A context manager during training
        self._training_context = training_context

    def register_begin_hook(self, hook_func):
        # Hook before smp is initialized
        self._begin_hooks.append(hook_func)

    def register_end_hook(self, hook_func):
        # Hook after current model test is finished
        self._end_hooks.append(hook_func)

    def register_pre_train_hook(self, non_smp_hook=None, smp_hook=None):
        # Hook before each step function
        if non_smp_hook != None:
            self._pre_train_hooks[0].append(non_smp_hook)
        if smp_hook != None:
            self._pre_train_hooks[1].append(smp_hook)

    def register_post_train_hook(self, non_smp_hook=None, smp_hook=None):
        # Hook after each step function
        if non_smp_hook != None:
            self._post_train_hooks[0].append(non_smp_hook)
        if smp_hook != None:
            self._post_train_hooks[1].append(smp_hook)

    def register_run_functions(self, run_func1=None, run_func2=None):
        # Regist non-default run funtions
        self._run_functions = [run_func1, run_func2]

    def register_test_functions(self, test_function):
        # Register extra test functions
        self._test_functions.append(test_function)

    @contextmanager
    def only_run_smp(self):
        """
        Context manager to skil non-smp ran and the grads/outputs test
        Run run_test with this context if you only want to run with smp for sanity check
        """
        orgin_verify_grads = self.verify_grads
        orgin_verify_grad_counts = self.verify_grad_counts
        origin_verify_loss = self.verify_loss
        origin_run_non_smp = self._run_without_smp

        if self._test_config != None:
            orgin_config_verify_grads = self._test_config.verify_grads
            orgin_config_verify_grad_counts = self._test_config.verify_grad_counts
            orgin_config_verify_loss = self._test_config.verify_loss
            self._test_config.verify_grads = False
            self._test_config.verify_grad_counts = False
            self._test_config.verify_loss = False
        self.verify_grads = False
        self.verify_grad_counts = False
        self.verify_loss = False
        # do not run without smp
        self._run_without_smp = partial(lambda self, index=0: None, self)

        yield

        self.verify_grads = orgin_verify_grads
        self.verify_grad_counts = orgin_verify_grad_counts
        self.verify_loss = origin_verify_loss
        self._run_without_smp = origin_run_non_smp
        if self._test_config != None:
            self._test_config.verify_grads = orgin_config_verify_grads
            self._test_config.verify_grad_counts = orgin_config_verify_grad_counts
            self._test_config.verify_loss = orgin_config_verify_loss

    def run_model_and_test(self):

        with self._training_context():
            self._run_functions[0](index=0)
            self._run_functions[1](index=1)

        for end_hook in self._end_hooks:
            end_hook()

        if self.verify_loss:
            self._verify_loss()

        if self.verify_grad_counts:
            self._verify_grad_counts()

        if self.verify_grads:
            self._verify_grads()

        # Only verify parameter if optimizer is defined
        if self.verify_parameters and self.current_model.optimizer != None:
            self._verify_parameters()

        # customized test functions
        for test in self._test_functions:
            test()

    def run_test(self, test_models):
        """
        Entry to start the test.
        """
        if self._training_context == None:
            self._training_context = contextlib.nullcontext

        if self._run_functions[0] == None:
            self._run_functions[0] = self._run_without_smp

        if self._run_functions[1] == None:
            self._run_functions[1] = self._run_with_smp

        for model in test_models:
            self.current_model = copy.deepcopy(model)
            self.apply_test_config()
            for begin_hook in self._begin_hooks:
                begin_hook()

            # Initialize Rubik
            self._init_smp()
            self.current_model.prepare_training(upscale_model_output=self.upscale_model_output)
            self.run_model_and_test()

            smp.barrier()
            self._reset_current_model_status()
            if self.verify_memory:
                self._verify_memory()

    def run_checkpoint_test(self, test_models):
        for model in test_models:
            self.current_model = copy.deepcopy(model)
            self.apply_test_config()
            for begin_hook in self._begin_hooks:
                begin_hook()

            self._test_checkpoint()

            smp.barrier()
            self._reset_current_model_status()
            if self.verify_memory:
                self._verify_memory()
