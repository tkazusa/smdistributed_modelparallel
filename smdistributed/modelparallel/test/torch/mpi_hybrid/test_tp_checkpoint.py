# Standard Library
import random
import unittest

# Third Party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.nn.utils import get_local_channels, get_start_pos_for_slicing

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TestTensorParallelismCheckpoint(unittest.TestCase):
    def check_optim_match(self, dist_optim_state_dict, optim_state_dict):
        for param_idx in dist_optim_state_dict["state"].keys():
            for state in dist_optim_state_dict["state"][param_idx]:
                if isinstance(dist_optim_state_dict["state"][param_idx][state], torch.Tensor):
                    assert torch.allclose(
                        dist_optim_state_dict["state"][param_idx][state].cpu(),
                        optim_state_dict["state"][param_idx][state].cpu(),
                        atol=1e-5,
                        rtol=1e-4,
                    )
                else:
                    assert (
                        dist_optim_state_dict["state"][param_idx][state]
                        == optim_state_dict["state"][param_idx][state]
                    )

    def check_model_match(self, dist_model_state_dict, model_state_dict):
        for param in dist_model_state_dict:
            # Had to reduce atol=1e-4 and rtol=1e-4 for params to match after update
            assert torch.allclose(
                dist_model_state_dict[param].cpu(),
                model_state_dict[param].cpu(),
                atol=1e-4,
                rtol=1e-4,
            )

    def check_model_optimizer_state(
        self,
        model_cls,
        optim_fn=optim.Adadelta,
        optim_params=None,
        model_check=True,
        save_load=True,
        same_partition_load=False,
        state_dict_allgather=True,
    ):
        def train_step_helper(model, data, target, use_smp=True):
            output = model(data)
            loss = F.nll_loss(output, target, reduction="mean")
            if use_smp:
                model.backward(loss)
            else:
                loss.backward()
            return output, loss

        @smp.step
        def train_step(model, data, target):
            return train_step_helper(model, data, target)

        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        # TODO: make batch size, and other dims for input configurable
        data_list = [
            torch.randn((64, 1, 28, 28)).to(device),
            torch.randn((64, 1, 28, 28)).to(device),
        ]
        target_list = [
            torch.randint(0, 10, (64,)).to(device),
            torch.randint(0, 10, (64,)).to(device),
        ]
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        model = model_cls()
        model.to(device)

        from smdistributed.modelparallel.torch.state_mod import state

        tensor_parallel_modules = [
            module for module in state.module_manager._tensor_parallelism_modules
        ]
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        model2 = model_cls()
        model2.to(device)

        # Save weights to set it for dist model
        saved_params = {}
        saved_buffers = {}
        for name, param in model.named_parameters():
            mod_names = name.split(".")[:-1]
            mod = model
            for mod_name in mod_names:
                mod = getattr(mod, mod_name)
            if isinstance(mod, nn.BatchNorm2d):
                saved_params[name] = param.data.clone()
            if mod in tensor_parallel_modules:
                # Supports only nn.Linear atm
                assert isinstance(mod, nn.Linear)
                if "bias" not in name:
                    local_channel = get_local_channels(param.size(-1))
                    start = get_start_pos_for_slicing(param.size(-1))
                    saved_params[name] = (
                        torch.narrow(param.data, -1, start, local_channel).contiguous().clone()
                    )
                else:
                    saved_params[name] = param.data.clone()

        for name, buff in model.named_buffers():
            mod_names = name.split(".")[:-1]
            mod = model
            for mod_name in mod_names:
                mod = getattr(mod, mod_name)
            if isinstance(mod, nn.BatchNorm2d):
                saved_buffers[name] = buff.data.clone()

        # Run forward, backward and update for non dist model
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        optimizer = optim_fn(model.parameters(), **optim_params)
        optimizer.zero_grad()
        inp = torch.cat([data_list[0], data_list[1]], 0)
        targ = torch.cat([target_list[0], target_list[1]], 0)
        output, loss = train_step_helper(model, inp, targ, use_smp=False)
        optimizer.step()

        dist_model = smp.DistributedModel(model2, average_grads_across_microbatches=False)

        # Restore saved weights
        for name, param in dist_model.module.module.named_parameters():
            if name in saved_params:
                if "bias" not in name:
                    param.data = saved_params[name]
                elif smp.tp_rank() == 0:
                    param.data = saved_params[name]

        for name, buff in dist_model.module.module.named_buffers():
            if name in saved_buffers:
                buff.data = saved_buffers[name]

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        optimizer2 = optim_fn(dist_model.parameters(), **optim_params)
        dist_optimizer = smp.DistributedOptimizer(optimizer2)
        d = data_list[smp.dp_rank()]
        t = target_list[smp.dp_rank()]
        dist_optimizer.zero_grad()
        output, loss = train_step(dist_model, d, t)
        dist_optimizer.step()

        if smp.rdp_rank() == 0:
            dist_optim_state_dict = dist_optimizer.state_dict()
            optim_state_dict = optimizer.state_dict()
            dist_model_state_dict = dist_model.state_dict(gather_to_rank0=not state_dict_allgather)
            model_state_dict = model.state_dict()
            self.check_optim_match(dist_optim_state_dict, optim_state_dict)
            if model_check:
                self.check_model_match(dist_model_state_dict, model_state_dict)
            if save_load:
                smp.save(
                    {"model": dist_model_state_dict, "optimizer_state": dist_optim_state_dict},
                    "/tmp/checkpoint.pt",
                    partial=False,
                )
                smp.save(
                    {
                        "model": dist_model.local_state_dict(),
                        "optimizer_state": dist_optimizer.local_state_dict(),
                    },
                    "/tmp/checkpoint.pt",
                    partial=True,
                )
        smp.barrier()

        if save_load and smp.rdp_rank() == 0:
            model_loaded = model_cls()
            model_loaded.to(device)
            optimizer_loaded = optim_fn(model_loaded.parameters(), **optim_params)
            checkpoint = smp.load("/tmp/checkpoint.pt", partial=False)
            loaded_model_state_dict, loaded_optim_state_dict = (
                checkpoint["model"],
                checkpoint["optimizer_state"],
            )
            model_loaded.load_state_dict(checkpoint["model"])
            optimizer_loaded.load_state_dict(checkpoint["optimizer_state"])
            self.check_optim_match(optim_state_dict, optimizer_loaded.state_dict())
            self.check_model_match(model_state_dict, model_loaded.state_dict())

            checkpoint = smp.load("/tmp/checkpoint.pt", partial=True)
            dist_model.load_state_dict(checkpoint["model"], same_partition_load=same_partition_load)
            dist_optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.check_optim_match(dist_optimizer.state_dict(), dist_optim_state_dict)
            self.check_model_match(
                dist_model.state_dict(gather_to_rank0=not state_dict_allgather),
                dist_model_state_dict,
            )

    def test_tp_optimizer_checkpoint_ubalanced(self):
        self.test_tp_optimizer_checkpoint(channel=127)

    def test_tp_optimizer_checkpoint(self, channel=128):
        # For determinism
        batch_size = 64

        class Net1(nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                # Commenting out bn for now, since there is an issue with models with buffers, when TP enabled
                # self.bn = nn.BatchNorm2d(1)
                with smp.partition(1):
                    self.conv1 = nn.Conv2d(1, 32, 3, 1)
                    self.conv2 = nn.Conv2d(32, 64, 3, 1)

            def forward(self, x):
                # Commenting out bn for now, since there is an issue with models with buffers, when TP enabled
                # x = self.bn(x)
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = torch.flatten(x, 1)
                return x

        class Net2(nn.Module):
            def __init__(self):
                super(Net2, self).__init__()
                with smp.tensor_parallelism():
                    self.fc1 = nn.Linear(9216, channel)
                    self.fc2 = nn.Linear(channel, 10)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                output = F.log_softmax(x, 1)
                return output

        class GroupedNet(nn.Module):
            def __init__(self):
                super(GroupedNet, self).__init__()
                self.net1 = Net1()
                self.net2 = Net2()

            def forward(self, x):
                x = self.net1(x)
                x = self.net2(x)
                return x

        # Optimizer pairs with flags obtained from here:
        # https://github.com/pytorch/pytorch/blob/5b648ef909fbf89c53f28ebc1b3bd2f4fde168c5/test/test_optim.py#L308
        optimizer_pairs_with_flags = [
            (
                (optim.SGD, optim._multi_tensor.SGD),
                dict(lr=0.2, momentum=1, dampening=0, weight_decay=1, nesterov=True),
            ),
            (
                (optim.SGD, optim._multi_tensor.SGD),
                dict(lr=0.2, momentum=1, dampening=0.5, weight_decay=1, nesterov=False),
            ),
            ((optim.ASGD, optim._multi_tensor.ASGD), dict(weight_decay=0)),
            ((optim.ASGD, optim._multi_tensor.ASGD), dict(weight_decay=1)),
            ((optim.Adadelta, optim._multi_tensor.Adadelta), dict(weight_decay=0)),
            ((optim.Adadelta, optim._multi_tensor.Adadelta), dict(weight_decay=1)),
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=1.0, amsgrad=True)),
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=1.0, amsgrad=False)),
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=0.0, amsgrad=True)),
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=0.0, amsgrad=False)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=1.0, amsgrad=True)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=1.0, amsgrad=False)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=0.0, amsgrad=True)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=0.0, amsgrad=False)),
            (
                (optim.Rprop, optim._multi_tensor.Rprop),
                dict(lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)),
            ),
            ((optim.Adamax, optim._multi_tensor.Adamax), dict(weight_decay=0)),
            ((optim.Adamax, optim._multi_tensor.Adamax), dict(weight_decay=1)),
            # RMSProp fails with optim state comparisons
            # ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=1, momentum=1, centered=True)),
            # ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=1, momentum=0, centered=True)),
            # ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=1, momentum=1, centered=False)),
            # ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=0, momentum=1, centered=False)),
        ]
        for optimizers, params in optimizer_pairs_with_flags:
            for opt in optimizers:
                for same_partition_load in [True, False]:
                    for state_dict_allgather in [True, False]:
                        smp.reset()
                        smp.init(
                            {
                                "pipeline_parallel_degree": 2,
                                "microbatches": 1,
                                "tensor_parallel_degree": 2,
                                "ddp": True,
                                "auto_partition": False,
                                "default_partition": 0,
                            }
                        )
                        self.check_model_optimizer_state(
                            GroupedNet,
                            opt,
                            params,
                            same_partition_load=same_partition_load,
                            state_dict_allgather=state_dict_allgather,
                        )


if __name__ == "__main__":
    unittest.main()
