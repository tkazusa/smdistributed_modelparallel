# Standard Library
import pickle
from collections import defaultdict
from contextlib import ContextDecorator
from typing import Dict
from unittest.mock import MagicMock

# Third Party
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

# First Party
import smdistributed.modelparallel.torch as smp
import smdistributed.modelparallel.torch.messages as smp_msgs
import smdistributed.modelparallel.torch.ops as smp_ops
from smdistributed.modelparallel.test.torch.mpi_4ps.utils import LinearActivation
from smdistributed.modelparallel.torch.nn.utils import get_local_channels, get_start_pos_for_slicing
from smdistributed.modelparallel.torch.tp_registry import get_weight_slice

ATOL = 1e-5
RTOL = 1e-4

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)

try:
    from smdistributed.modelparallel.torch.nn import FusedLayerNorm

    LayerNorm = FusedLayerNorm
except ImportError:
    LayerNorm = nn.LayerNorm


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val

    return conversion_helper(val, half_conversion)


def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class FP16_Module(nn.Module):
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module("module", module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


def slice_and_compare_grads(module, dist_module, partition=None, split_shapes=None, atol=1e-2):
    if partition == "input":
        if split_shapes == None:
            local_in_features = get_local_channels(module.in_features)
            start = get_start_pos_for_slicing(module.in_features)
        else:
            local_in_features = split_shapes[smp.tp_rank()]
            start = sum(split_shapes[: smp.tp_rank()])
        weight_grad_slice = module.weight.grad.narrow(1, start, local_in_features).contiguous()
        assert torch.allclose(weight_grad_slice, dist_module.weight.grad, atol=atol)
        if smp.tp_rank() == 0:
            assert torch.allclose(module.bias.grad, dist_module.bias.grad, atol=atol)
    elif partition == "output":
        if isinstance(module, (nn.Linear, LinearActivation)):
            out_features = module.out_features
        elif isinstance(module, LayerNorm):
            out_features = module.normalized_shape[0]
        if split_shapes == None:
            local_out_features = get_local_channels(out_features)
            start = get_start_pos_for_slicing(out_features)
        else:
            local_out_features = split_shapes[smp.tp_rank()]
            start = get_start_pos_for_slicing(out_features)
        weight_grad_slice = module.weight.grad.narrow(0, start, local_out_features).contiguous()
        bias_grad_slice = module.bias.grad.narrow(0, start, local_out_features).contiguous()
        assert torch.allclose(weight_grad_slice, dist_module.weight.grad, atol=atol)
        assert torch.allclose(bias_grad_slice, dist_module.bias.grad, atol=atol)
    else:
        assert torch.allclose(module.weight.grad, dist_module.weight.grad, atol=atol)
        assert torch.allclose(module.bias.grad, dist_module.bias.grad, atol=atol)


def equalize_embedding_weights(module, dist_module, split=True):
    with torch.no_grad():
        if split:
            slice_size = get_local_channels(module.embedding_dim)
            start = get_start_pos_for_slicing(module.embedding_dim)
            weight_slice = module.weight.clone().narrow(-1, start, slice_size).contiguous()
        else:
            weight_slice = module.weight
        dist_module.weight.copy_(weight_slice)


def equalize_linear_weights(module, dist_module, partition=None, split_shapes=None):
    with torch.no_grad():
        if partition == "input":
            if split_shapes == None:
                local_in_features = get_local_channels(module.in_features)
                start = get_start_pos_for_slicing(module.in_features)
            else:
                local_in_features = split_shapes[smp.tp_rank()]
                start = sum(split_shapes[: smp.tp_rank()])
            weight_slice = module.weight.narrow(1, start, local_in_features).contiguous()
            print(dist_module.weight.shape, weight_slice.shape)
            dist_module.weight.copy_(weight_slice)  # = nn.Parameter(weight_slice)

            if smp.tp_rank() == 0:
                dist_module.bias.copy_(module.bias)
        elif partition == "output":
            if isinstance(module, (nn.Linear, LinearActivation)):
                out_features = module.out_features
            elif isinstance(module, LayerNorm):
                out_features = module.normalized_shape[0]
            print(type(module))
            if split_shapes == None:
                local_out_features = get_local_channels(out_features)
                start = get_start_pos_for_slicing(out_features)
            else:
                local_out_features = split_shapes[smp.tp_rank()]
                start = sum(split_shapes[: smp.tp_rank()])

            weight_slice = module.weight.narrow(0, start, local_out_features).contiguous()
            bias_slice = module.bias.narrow(0, start, local_out_features).contiguous()
            print(dist_module.weight.shape, weight_slice.shape)
            dist_module.weight.copy_(weight_slice)
            dist_module.bias.copy_(bias_slice)
        else:
            dist_module.weight.copy_(module.weight)
            dist_module.bias.copy_(module.bias)


# below two functions match the weights with HF GPT-2 Block class
# helpful for debugging.


def match_weights_opt_mem(mod, dist_mod):
    with torch.no_grad():
        mod_q_weight = mod.attn.c_attn.weight.narrow(1, 0, 2048)
        mod_k_weight = mod.attn.c_attn.weight.narrow(1, 2048, 2048)
        mod_v_weight = mod.attn.c_attn.weight.narrow(1, 2 * 2048, 2048)

        mod_q_bias = mod.attn.c_attn.bias.narrow(0, 0, 2048)
        mod_k_bias = mod.attn.c_attn.bias.narrow(0, 2048, 2048)
        mod_v_bias = mod.attn.c_attn.bias.narrow(0, 2 * 2048, 2048)

        dist_mod.attention.query.weight.copy_(get_weight_slice(mod_q_weight, "output").t())
        dist_mod.attention.key.weight.copy_(get_weight_slice(mod_k_weight, "output").t())
        dist_mod.attention.value.weight.copy_(get_weight_slice(mod_v_weight, "output").t())
        if smp.tp_rank() == 0:
            dist_mod.attention.query.bias.copy_(get_weight_slice(mod_q_bias, None))
            dist_mod.attention.key.bias.copy_(get_weight_slice(mod_k_bias, None))
            dist_mod.attention.value.bias.copy_(get_weight_slice(mod_v_bias, None))

        dist_mod.attention.dense.weight.copy_(
            get_weight_slice(mod.attn.c_proj.weight, "output").t()
        )
        if smp.tp_rank() == 0:
            dist_mod.attention.dense.bias.copy_(mod.attn.c_proj.bias)

        dist_mod.output.dense1.weight.copy_(get_weight_slice(mod.mlp.c_fc.weight, "output").t())
        if smp.tp_rank() == 0:
            dist_mod.output.dense1.bias.copy_(get_weight_slice(mod.mlp.c_fc.bias, None))

        dist_mod.output.dense2.weight.copy_(get_weight_slice(mod.mlp.c_proj.weight, "output").t())
        if smp.tp_rank() == 0:
            dist_mod.output.dense2.bias.copy_(mod.mlp.c_proj.bias)


def match_weights_opt_speed(mod, dist_mod):
    with torch.no_grad():
        dist_mod.attention.pre_layernorm.weight.copy_(mod.ln_1.weight)
        dist_mod.attention.pre_layernorm.bias.copy_(mod.ln_1.bias)

        mod_q_weight = mod.attn.c_attn.weight.narrow(1, 0, 2048)
        mod_k_weight = mod.attn.c_attn.weight.narrow(1, 2048, 2048)
        mod_v_weight = mod.attn.c_attn.weight.narrow(1, 2 * 2048, 2048)

        mod_q_bias = mod.attn.c_attn.bias.narrow(0, 0, 2048)
        mod_k_bias = mod.attn.c_attn.bias.narrow(0, 2048, 2048)
        mod_v_bias = mod.attn.c_attn.bias.narrow(0, 2 * 2048, 2048)

        dist_mod.attention.query.weight.copy_(get_weight_slice(mod_q_weight, "input").t())
        dist_mod.attention.key.weight.copy_(get_weight_slice(mod_k_weight, "input").t())
        dist_mod.attention.value.weight.copy_(get_weight_slice(mod_v_weight, "input").t())
        dist_mod.attention.query.bias.copy_(get_weight_slice(mod_q_bias, "output"))
        dist_mod.attention.key.bias.copy_(get_weight_slice(mod_k_bias, "output"))
        dist_mod.attention.value.bias.copy_(get_weight_slice(mod_v_bias, "output"))

        dist_mod.attention.dense.weight.copy_(
            get_weight_slice(mod.attn.c_proj.weight, "output").t()
        )
        if smp.tp_rank() == 0:
            dist_mod.attention.dense.bias.copy_(mod.attn.c_proj.bias)

        dist_mod.output.pre_layernorm.weight.copy_(mod.ln_2.weight)
        dist_mod.output.pre_layernorm.bias.copy_(mod.ln_2.bias)

        dist_mod.output.dense1.weight.copy_(get_weight_slice(mod.mlp.c_fc.weight, "input").t())
        dist_mod.output.dense1.bias.copy_(get_weight_slice(mod.mlp.c_fc.bias, "output"))

        dist_mod.output.dense2.weight.copy_(get_weight_slice(mod.mlp.c_proj.weight, "output").t())
        if smp.tp_rank() == 0:
            dist_mod.output.dense2.bias.copy_(mod.mlp.c_proj.bias)


class NetTransformer(nn.Module):
    """Model to test input tensor in kwargs
    """

    def __init__(self):
        super(NetTransformer, self).__init__()
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0

        self.encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

    def forward(self, encoder_input, src_key_padding_mask=None):
        return self.encoder(src=encoder_input, src_key_padding_mask=src_key_padding_mask)


class Net1(nn.Module):
    """Dummy module used for testing
    """

    def __init__(self):
        super(Net1, self).__init__()
        self.linear1 = nn.Linear(1000, 40)
        self.linear2 = nn.Linear(40, 20)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return x2


class Net2(nn.Module):
    """Dummy module used for testing
    """

    def __init__(self):
        super(Net2, self).__init__()
        self.linear1 = nn.Linear(20, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class NetOneToMany(nn.Module):
    """Module used for testing with ops,
    takes one input, returns two outputs
    """

    def __init__(self):
        super(NetOneToMany, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x1):
        out1 = self.linear1(x1)
        out2 = self.linear2(x1)
        return out1, out2


class NetManyToOne(nn.Module):
    """Module used for testing with ops,
    takes two inputs, returns one output
    """

    def __init__(self):
        super(NetManyToOne, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x1, x2):
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x3 = x1 + x2
        return x3


class mock_send_backward(ContextDecorator):
    """Mocks the call to trigger backward request.
    """

    def __init__(self, outputs=None, backward=False):
        """Setting trigger_backward to true will trigger backward
        on outputs, with the request args. Otherwise thread_get_backward_result
        will do nothing and return self.outputs
        """
        self.outputs = outputs
        self.backward = backward

    def trigger_backward(self, request):
        for output, out_grad in zip(self.outputs, request.args):
            output.backward(out_grad)

    def __enter__(self):
        smp_ops.state.exec_thread_id = 1
        smp_ops.state.exec_server = MagicMock()
        if not self.backward:
            smp_ops.state.current_worker.thread_get_backward_result = MagicMock(
                return_value=self.outputs
            )
        else:
            smp_ops.state.current_worker.thread_get_backward_result = MagicMock(
                side_effect=self.trigger_backward
            )
        return self

    def __exit__(self, *exc):
        pass


class mock_recv_forward(ContextDecorator):
    """Mocks the call to wait on the forward request.
    """

    def __init__(self, outputs):
        """If forward is a dict, uses the dict mapping of module->output
        as return values of thread_get_forward_result"""
        self.is_dict = isinstance(outputs, dict)
        self.outputs = outputs

    def __enter__(self):
        def get_output(request):
            return self.outputs[request.module]

        smp_ops.state.exec_server = MagicMock()
        result = (
            MagicMock(side_effect=get_output)
            if self.is_dict
            else MagicMock(return_value=self.outputs)
        )
        smp_ops.state.exec_thread_id = 1
        smp_ops.state.current_worker.thread_get_forward_result = result
        return self

    def __exit__(self, *exc):
        pass


class mock_get_module_fns(ContextDecorator):
    """Mocks the call to get_parent_module, get_module_name
    to return argument proided during instantiation
    """

    def __init__(self, parent_module=None):
        """Returns the module and str(module)
        for get_parent_module and get_module_name
        """
        self.parent_module = parent_module

    def __enter__(self):
        self.get_module_name = smp_msgs.state.module_manager.get_module_name
        self.get_parent_module = smp_msgs.state.module_manager.get_parent_module
        smp_msgs.state.module_manager.get_module_name = MagicMock(
            return_value=str(self.parent_module)
        )
        smp_msgs.state.module_manager.get_parent_module = MagicMock(return_value=self.parent_module)
        return self

    def __exit__(self, *exc):
        smp_msgs.state.module_manager.get_module_name = self.get_module_name
        smp_msgs.state.module_manager.get_parent_module = self.get_parent_module


class mock_output_stack_size(ContextDecorator):
    """Mocks the call to output_stack_size
    """

    def __init__(self, stack_size=1):
        """Returns the stack size for output_stack_size
        call"""
        self.stack_size = stack_size

    def __enter__(self):
        self.output_stack_size = smp_msgs.state.module_manager.output_stack_size
        smp_msgs.state.module_manager.output_stack_size = MagicMock(return_value=self.stack_size)
        return self

    def __exit__(self, *exc):
        smp_msgs.state.module_manager.output_stack_size = self.output_stack_size


class mock_is_executor(ContextDecorator):
    """Mocks the call to is_executor
    """

    def __init__(self, module_dict={}, is_executor=False, is_parent_executor=False):
        """Mocks the is_executor and is_parent_executor
        functions in module_manager. If module not found in dict
        returns value in is_executor and is_parent_executor
        """
        self.module_dict = module_dict
        self.is_executor = is_executor
        self.is_parent_executor = is_parent_executor

    def __enter__(self):
        def get_executor_state(module):
            if not self.module_dict:
                return (self.is_executor, self.is_parent_executor)
            else:
                return (self.module_dict[module][0], self.module_dict[module][1])

        self.backup_is_executor = smp_ops.state.module_manager.is_executor
        self.backup_is_parent_executor = smp.ops.state.module_manager.is_parent_executor
        smp_ops.state.module_manager.is_executor = MagicMock(
            side_effect=lambda x: get_executor_state(x)[0]
        )
        smp_ops.state.module_manager.is_parent_executor = MagicMock(
            side_effect=lambda x: get_executor_state(x)[1]
        )
        return self

    def __exit__(self, *exc):
        smp_ops.state.module_manager.is_executor = self.backup_is_executor
        smp_ops.state.module_manager.is_parent_executor = self.backup_is_parent_executor


class mock_push_output(ContextDecorator):
    """Mocks the call which is used to save a single output,
    in the forward call for SMPOutput op. Saves the tensors
    passed to the push_output call to be verified later"""

    def __enter__(self):
        def push_output(microbatch, module, parent_module, tensors):
            self.output_tensors = tensors

        self.push_output = smp_ops.state.module_manager.push_output
        smp_ops.state.module_manager.push_output = MagicMock(side_effect=push_output)
        return self

    def __exit__(self, *exc):
        smp_ops.state.module_manager.push_output = self.push_output


def dump_model(model, batch_idx, identifier, output, loss):
    with open(f"{identifier}_{batch_idx}.pkl", "wb") as f:
        grads = {}
        params = {}
        for n, p in model.named_parameters():
            grads[n] = p.grad
            params[n] = p
        pickle.dump((params, grads, output, loss), f)


def verify_weights(model, batch_idx, identifier, check_all=False):
    cpu = torch.device("cpu")
    with open(f"{identifier}_{batch_idx}.pkl", "rb") as f:
        params, grads, nosmp_output, nosmp_loss = pickle.load(f)
        for n, p in model.named_parameters():
            if p.is_cuda:
                assert p.requires_grad, (smp.rank(), n, "has no requires_grad for param")
            if p.is_cuda or check_all:

                p = p.to(cpu)
                loaded_param = params[n].to(cpu)
                if not torch.allclose(p, loaded_param, rtol=1e-4, atol=1e-5):
                    raise ValueError(smp.rank(), n, f"not same for {batch_idx}")
                # else:
                # print(f"{smp.rank()} {n} weight is same for {batch_idx}")
            # else:
            # print(f"{smp.rank()} {n} is on cpu, for {batch_idx}")


def verify_grads(model, batch_idx, identifier):
    cpu = torch.device("cpu")
    notsame = set()
    total = 0
    with open(f"{identifier}_{batch_idx}.pkl", "rb") as f:
        params, grads, nosmp_output, nosmp_loss = pickle.load(f)
        for n, p in model.named_parameters():
            if p.grad is not None:
                pgrad = p.grad.to(cpu)
                non_smp_name = n.split("module.", 1)[1]
                loaded_grad = grads[non_smp_name].to(cpu)
                if p.is_cuda and not torch.allclose(pgrad, loaded_grad, rtol=1e-4, atol=1e-5):
                    notsame.add(n)
                    # raise ValueError(f"{smp.rank(), n, 'not same for grad of ', batch_idx}")
                # else:
                # print(f"{smp.rank()} has same grad for {n} of {batch_idx}")
                total += 1
            # else:
            # print(f"{smp.rank()} has no grad for {n} in {batch_idx}")
        assert total == len(list(model.local_parameters()))
        if notsame:
            print(f"{smp.rank()} {notsame} grads out of {total} are not same for {batch_idx}")
        else:
            print(f"{smp.rank()} Grads matched for {batch_idx}")


def verify_outputs_loss(model, batch_idx, identifier, output, loss):
    cpu = torch.device("cpu")
    with open(f"{identifier}_{batch_idx}.pkl", "rb") as f:
        params, grads, nosmp_output, nosmp_loss = pickle.load(f)
        if output is not None:
            if torch.is_tensor(output):
                output = [output]
            if torch.is_tensor(nosmp_output):
                nosmp_output = [nosmp_output]
            assert len(output) == len(nosmp_output)
            for i in range(len(output)):
                if not torch.allclose(
                    output[i].to(cpu), nosmp_output[i].to(cpu), rtol=1e-4, atol=1e-5
                ):
                    raise ValueError(
                        "outputs not same",
                        output[i],
                        output[i].sum(),
                        nosmp_output[i],
                        nosmp_output[i].sum(),
                    )
            print(f"{smp.rank()} Outputs are same for {batch_idx}")

        if loss is not None:
            if not torch.allclose(loss.to(cpu), nosmp_loss.to(cpu)):
                raise ValueError("loss not same", loss, nosmp_loss)
            else:
                print(f"{smp.rank()} Losses are same for {batch_idx}: {loss}")


def verify(model, batch_idx, identifier, output, loss):
    verify_weights(model, batch_idx, identifier)
    verify_outputs_loss(model, batch_idx, identifier, output, loss)
    verify_grads(model, batch_idx, identifier)


def add_num_grads_hook(model: nn.Module) -> Dict[str, int]:
    num_hook_calls = defaultdict(int)

    def get_hook_fn(n):
        def hook(x):
            num_hook_calls[n] += 1
            # print(f"[{smp.rank()}] Grad computed for {n}")

        return hook

    handles = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            handle = p.register_hook(get_hook_fn(n))
            handles.append(handle)
    return num_hook_calls, handles
