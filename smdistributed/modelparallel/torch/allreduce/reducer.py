# Standard Library
from abc import ABCMeta, abstractmethod
from typing import Dict, Set

# Third Party
import torch
from torch.nn import Parameter

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.exceptions import SMPUnsupportedError

logger = get_logger()

# This is inherited by all the reducers except the ddp_reducer.cc.
# For gradscaling when there's no data parallelism(GradScaler) inherits this and bypasses the allreduce calls.
# Horovod, DDP without overlap, and Herring also inherit this and implement
# the core allreduce call in _hook_internal, and the _synchronize_internal methods.

# TorchDDPReducer in cpp does not inherit this but we need to keep functionality same in both.
# Currently ddp_reducer.cc is not used when overlapping_allreduce is False.
# We fallback to the reducer in ddp.py for that.
# Inheritance across languages might be harder, but haven't investigated.


class GradReducer:
    __metaclass__ = ABCMeta

    def __init__(
        self,
        named_parameters: Dict[str, Parameter],
        grad_counter,
        overlapping_allreduce,
        average_grads_across_microbatches,
        num_microbatches,
        scaled_batch,
        tp_size,
    ):
        self.overlapping_allreduce = overlapping_allreduce
        self.average_grads_across_microbatches = average_grads_across_microbatches
        self.grad_counter = grad_counter
        self.num_microbatches = num_microbatches
        self.scaled_batch = scaled_batch
        self.tp_size = tp_size

        self.named_parameters = named_parameters

        self._grad_accs = set()
        self._registered_hooks: Set[Parameter] = set()
        self._to_sync_grads = False

        # might be set to False only in some unit tests
        self.enable_grad_counting = True

        self.register_hooks()

    # this set grad is needed when user first runs test before train for example to see initial loss
    # first step will have no grad and this will crash
    @torch.enable_grad()
    def register_hooks(self):
        for n, p in self.named_parameters.items():
            if p.requires_grad and p not in self._registered_hooks:
                # this is better than hook on param because the first time we get hook callback,
                # we can query p.grad, else it will be None.
                # Taken from horovod. Even torch ddp does the same thing in CPP.

                p.grad = p.data.new(p.size()).zero_()
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                self._grad_accs.add(grad_acc)
                grad_acc.register_hook(self._make_grad_hook(n, p))
                self._registered_hooks.add(p)

    def prepare_for_backward(self):
        self._to_sync_grads = True

    def synchronize(self):
        self._synchronize_internal()
        self._to_sync_grads = False

    @abstractmethod
    def _synchronize_internal(self):
        raise SMPUnsupportedError

    @abstractmethod
    def _hook_internal(self, name, param, grads):
        raise SMPUnsupportedError

    def _make_grad_hook(self, name, param):
        def hook(*grads):
            if self.enable_grad_counting:
                self.grad_counter.mark_grad_computed(name)

        return hook
