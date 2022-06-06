# Third Party
import torch.nn as nn


class DistributedModule(nn.Module):
    def __call__(self, *args, **kwargs):
        from smdistributed.modelparallel.torch.state_mod import state

        if self in state.tp_registry.forward_hooks:
            translated_args, translated_kwargs = state.tp_registry.forward_hooks[self](
                *args, **kwargs
            )
        else:
            translated_args, translated_kwargs = args, kwargs

        module_outputs = super(DistributedModule, self).__call__(
            *translated_args, **translated_kwargs
        )

        if self in state.tp_registry.return_hooks:
            translated_outputs = state.tp_registry.return_hooks[self](module_outputs)
        else:
            translated_outputs = module_outputs

        return translated_outputs

    @staticmethod
    def can_distribute(*args, **kwargs):
        return True

    def can_shard_activation_offloading(self):
        return False
