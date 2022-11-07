# Standard Library
import inspect

# Third Party
import torch
import torch.nn as nn

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.nn import DistributedModule
from smdistributed.modelparallel.torch.nn.embedding import DistributedEmbedding
from smdistributed.modelparallel.torch.nn.linear import DistributedLinear
from smdistributed.modelparallel.torch.nn.predefined_hooks import PredefinedHookManager
from smdistributed.modelparallel.torch.nn.utils import get_local_channels, get_start_pos_for_slicing

_SUPPORTED_TENSOR_PARALLELISM_MODULES = {
    nn.Linear: DistributedLinear,
    nn.Embedding: DistributedEmbedding,
}

logger = get_logger()


def get_weight_slice(weight, partition, split_shapes=None):  # pragma: no cover
    from smdistributed.modelparallel.torch.core import tp_rank

    if partition not in ["input", "output"]:
        return weight

    features = weight.size(1) if partition == "input" else weight.size(0)
    dim = 1 if partition == "input" else 0
    if split_shapes == None:
        local_features = get_local_channels(features)
        start = get_start_pos_for_slicing(features)
    else:
        local_features = split_shapes[tp_rank()]
        start = sum(split_shapes[: tp_rank()])

    weight_slice = weight.narrow(dim, start, local_features)
    return weight_slice


# below two functions match the weights with HF GPT-2 Block class
# helpful for debugging.


def match_weights_opt_speed(mod, dist_mod, distribute_embedding=False):  # pragma: no cover
    from smdistributed.modelparallel.torch.core import tp_rank

    with torch.no_grad():

        transformer, _ = [c for c in mod.children()]
        wte, wpe, _, h, ln_f = [c for c in transformer.children()]

        wte_slice = get_weight_slice(wte.weight, "input")
        if distribute_embedding:
            dist_mod.word_embedding.weight.copy_(get_weight_slice(wte.weight, "output"))
        else:
            dist_mod.word_embedding.weight.copy_(wte.weight)
        dist_mod.position_embedding.weight.copy_(wpe.weight)

        # don't need to worry about layernorm

        for c, dc in zip(h.children(), dist_mod.transformer.seq_layers.children()):
            attention_size = dc.attention.total_attention_size
            split_shapes = dc.attention.all_local_attention_sizes
            dc.attention.pre_layernorm.weight.copy_(c.ln_1.weight)
            dc.attention.pre_layernorm.bias.copy_(c.ln_1.bias)

            mod_q_weight = c.attn.c_attn.weight.narrow(1, 0, attention_size)
            mod_k_weight = c.attn.c_attn.weight.narrow(1, attention_size, attention_size)
            mod_v_weight = c.attn.c_attn.weight.narrow(1, 2 * attention_size, attention_size)

            mod_q_bias = c.attn.c_attn.bias.narrow(0, 0, attention_size)
            mod_k_bias = c.attn.c_attn.bias.narrow(0, attention_size, attention_size)
            mod_v_bias = c.attn.c_attn.bias.narrow(0, 2 * attention_size, attention_size)

            dc.attention.query.weight.copy_(
                get_weight_slice(mod_q_weight, "input", split_shapes=split_shapes).t()
            )
            dc.attention.key.weight.copy_(
                get_weight_slice(mod_k_weight, "input", split_shapes=split_shapes).t()
            )
            dc.attention.value.weight.copy_(
                get_weight_slice(mod_v_weight, "input", split_shapes=split_shapes).t()
            )
            dc.attention.query.bias.copy_(
                get_weight_slice(mod_q_bias, "output", split_shapes=split_shapes)
            )
            dc.attention.key.bias.copy_(
                get_weight_slice(mod_k_bias, "output", split_shapes=split_shapes)
            )
            dc.attention.value.bias.copy_(
                get_weight_slice(mod_v_bias, "output", split_shapes=split_shapes)
            )

            dc.attention.dense.weight.copy_(
                get_weight_slice(c.attn.c_proj.weight, "output", split_shapes=split_shapes).t()
            )
            if tp_rank() == 0:
                dc.attention.dense.bias.copy_(c.attn.c_proj.bias)

            dc.output.pre_layernorm.weight.copy_(c.ln_2.weight)
            dc.output.pre_layernorm.bias.copy_(c.ln_2.bias)

            dc.output.dense1.weight.copy_(get_weight_slice(c.mlp.c_fc.weight, "input").t())
            dc.output.dense1.bias.copy_(get_weight_slice(c.mlp.c_fc.bias, "output"))

            dc.output.dense2.weight.copy_(get_weight_slice(c.mlp.c_proj.weight, "output").t())
            if tp_rank() == 0:
                dc.output.dense2.bias.copy_(c.mlp.c_proj.bias)


def match_weights_opt_memory(mod, dist_mod, distribute_embedding=False):  # pragma: no cover
    from smdistributed.modelparallel.torch.core import tp_rank

    with torch.no_grad():

        transformer, _ = [c for c in mod.children()]
        wte, wpe, _, h, ln_f = [c for c in transformer.children()]

        if distribute_embedding:
            dist_mod.word_embedding.weight.copy_(get_weight_slice(wte.weight, "input"))
            dist_mod.position_embedding.weight.copy_(get_weight_slice(wpe.weight, "input"))
        else:
            dist_mod.word_embedding.weight.copy_(wte.weight)
            dist_mod.position_embedding.weight.copy_(wpe.weight)

        for c, dc in zip(h.children(), dist_mod.transformer.seq_layers.children()):
            attention_size = dc.attention.total_attention_size
            split_shapes = dc.attention.all_local_attention_sizes

            mod_q_weight = c.attn.c_attn.weight.narrow(1, 0, attention_size)
            mod_k_weight = c.attn.c_attn.weight.narrow(1, attention_size, attention_size)
            mod_v_weight = c.attn.c_attn.weight.narrow(1, 2 * attention_size, attention_size)

            mod_q_bias = c.attn.c_attn.bias.narrow(0, 0, attention_size)
            mod_k_bias = c.attn.c_attn.bias.narrow(0, attention_size, attention_size)
            mod_v_bias = c.attn.c_attn.bias.narrow(0, 2 * attention_size, attention_size)

            dc.attention.query.weight.copy_(get_weight_slice(mod_q_weight, "output").t())
            dc.attention.key.weight.copy_(get_weight_slice(mod_k_weight, "output").t())
            dc.attention.value.weight.copy_(get_weight_slice(mod_v_weight, "output").t())
            if tp_rank() == 0:
                dc.attention.query.bias.copy_(get_weight_slice(mod_q_bias, None))
                dc.attention.key.bias.copy_(get_weight_slice(mod_k_bias, None))
                dc.attention.value.bias.copy_(get_weight_slice(mod_v_bias, None))

            dc.attention.dense.weight.copy_(
                get_weight_slice(c.attn.c_proj.weight, "output", split_shapes=split_shapes).t()
            )
            if tp_rank() == 0:
                dc.attention.dense.bias.copy_(c.attn.c_proj.bias)

            dc.output.dense1.weight.copy_(get_weight_slice(c.mlp.c_fc.weight, "output").t())
            if tp_rank() == 0:
                dc.output.dense1.bias.copy_(get_weight_slice(c.mlp.c_fc.bias, None))

            dc.output.dense2.weight.copy_(get_weight_slice(c.mlp.c_proj.weight, "output").t())
            if tp_rank() == 0:
                dc.output.dense2.bias.copy_(c.mlp.c_proj.bias)


class TensorParallelismRegistry:
    def __init__(self):  # pragma: no cover
        self.reset()
        self.register_distributed_modules()

    def register_distributed_modules(self):  # pragma: no cover
        self.tp_modules = {}

        # keyed by original module class
        self.init_hooks = {}
        self.translate_functions = {}

        # keyed by (original module class, distributed module class)
        self._forward_hooks_per_class_pair = {}
        self._return_hooks_per_class_pair = {}

        self.predefined_hook_manager = PredefinedHookManager()

        for mod, dist_mod in _SUPPORTED_TENSOR_PARALLELISM_MODULES.items():
            setattr(nn, mod.__name__, self.register(dist_mod)(mod))

        for _, mapping in self.predefined_hook_manager.all_mappings():
            self.register_with_module(*mapping)

    def clear_module_init_args(self):
        self.module_init_args = {}

    def reset(self):
        self.clear_module_init_args()

        # keyed by actual distributed module objects
        self.forward_hooks = {}
        self.return_hooks = {}

    def is_supported(self, module_cls):
        return module_cls in self.tp_modules

    def distribute(self, module):
        from smdistributed.modelparallel.torch.state_mod import state

        if type(module) not in self.tp_modules:
            raise UnsupportedTPModuleError(type(module))

        # use the recorded (args, kwargs) to create a
        # new module instance
        args, kwargs = self.module_init_args[module]

        if type(module) in self.init_hooks:
            hook_output = self.init_hooks[type(module)](*args, **kwargs)
            if hook_output is not None:
                converted_args, converted_kwargs = hook_output
            else:
                logger.warning(
                    f"Unable to distribute instance of {type(module)} since not all of its __init__ arguments are supported."
                )
                return module
        else:
            converted_args, converted_kwargs = args, kwargs

        dist_cls = self.tp_modules[type(module)]

        # override the kwargs with input from tp_config
        signature = inspect.signature(dist_cls.__init__).parameters

        tp_config = state.module_manager.get_tp_config(module)
        for k, v in tp_config.items():
            if k in signature:
                converted_kwargs[k] = v
            elif hasattr(dist_cls, "_KEYS") and k in dist_cls._KEYS:
                converted_kwargs[k] = v

        if dist_cls.can_distribute(*converted_args, **converted_kwargs):
            dist_mod = dist_cls(*converted_args, **converted_kwargs)
            if state.cfg._match_weights:
                distribute_embedding = converted_kwargs.get("distribute_embedding", False)
                if state.cfg.optimize == "speed":
                    match_weights_opt_speed(module, dist_mod, distribute_embedding)
                else:
                    match_weights_opt_memory(module, dist_mod, distribute_embedding)

            # register the forward and return hooks for the module object, if any
            if (type(module), dist_cls) in self._forward_hooks_per_class_pair:
                self.forward_hooks[dist_mod] = self._forward_hooks_per_class_pair[
                    (type(module), dist_cls)
                ]
            if (type(module), dist_cls) in self._return_hooks_per_class_pair:
                self.return_hooks[dist_mod] = self._return_hooks_per_class_pair[
                    (type(module), dist_cls)
                ]

            if (
                type(module) in self.translate_functions
                and self.translate_functions[type(module)] is not None
            ):
                state.module_manager.register_translate_function(
                    self.translate_functions[type(module)]
                )
            return dist_mod
        else:
            # if the module cannot be distributed with the given arguments, then do not distribute
            return module

    def _handle_hooks(self, module_cls, dist_module, init_hook, forward_hook, return_hook):

        if init_hook is not None:
            self.init_hooks[module_cls] = init_hook
        if forward_hook is not None:
            self._forward_hooks_per_class_pair[(module_cls, dist_module)] = forward_hook
        if return_hook is not None:
            self._return_hooks_per_class_pair[(module_cls, dist_module)] = return_hook

    def get_tracking_init(self, org_init):
        def tracking_init(module_obj, *args, **kwargs):
            self.module_init_args[module_obj] = (args, kwargs)
            return org_init(module_obj, *args, **kwargs)

        return tracking_init

    def register(self, dist_module, init_hook=None, forward_hook=None, return_hook=None):
        if not issubclass(dist_module, DistributedModule):
            raise TPModuleRegisterError(dist_module.__name__)

        def patch_module_cls_init(module_cls):
            # patch __init__ so that we track the (args, kwargs) the original module was created with
            module_cls.__init__ = self.get_tracking_init(module_cls.__init__)
            self.tp_modules[module_cls] = dist_module
            self._handle_hooks(module_cls, dist_module, init_hook, forward_hook, return_hook)
            return module_cls

        return patch_module_cls_init

    def register_with_module(
        self,
        module_cls,
        dist_module,
        init_hook=None,
        forward_hook=None,
        return_hook=None,
        translate_functions=None,
    ):
        if not issubclass(dist_module, DistributedModule):
            raise TPModuleRegisterError(dist_module.__name__)

        module_cls.__init__ = self.get_tracking_init(module_cls.__init__)
        self.tp_modules[module_cls] = dist_module
        self._handle_hooks(module_cls, dist_module, init_hook, forward_hook, return_hook)
        self.translate_functions[module_cls] = translate_functions
