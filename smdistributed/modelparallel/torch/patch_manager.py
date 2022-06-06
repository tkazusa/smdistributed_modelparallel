# Standard Library
from contextlib import contextmanager
from typing import Callable, Type

# Third Party
import torch
from torch.nn import Module, Sequential

# TODO: support custom modules which inherit a patched module, it currently goes into a recursion loop


class PatchManager:
    def __init__(self):
        self._orig_methods = {"to": {}, "forward": {}, "init": {}}
        self._orig_no_grad = None

    def reset(self):
        """
        This helps reset all state so we can run on a different model after rest
        """
        # We keep track of original methods as there are situations when we want to reverse the patches.
        # It also helps when patching the second time in certain cases if it comes up. We dont want to
        # confused an already patched method as the original one.
        if hasattr(self, "_orig_methods"):
            self.reset_original_methods()
        self._orig_methods = {"to": {}, "forward": {}, "init": {}}

    def reset_original_methods(self):
        for mname in self._orig_methods:
            for moduleclass in self._orig_methods[mname]:
                if mname == "forward":
                    moduleclass.forward = self._orig_methods[mname][moduleclass]
                elif mname == "to":
                    moduleclass.to = self._orig_methods[mname][moduleclass]
                elif mname == "init":
                    moduleclass.__init__ = self._orig_methods[mname][moduleclass]

    def saved_original_method(self, method_name: str, moduleclass: Type[Module]):
        return moduleclass in self._orig_methods[method_name]

    def save_original_method(self, method_name: str, moduleclass: Type[Module], method: Callable):
        if not self.saved_original_method(method_name, moduleclass):
            self._orig_methods[method_name][moduleclass] = method
            setattr(moduleclass, "_orig_" + method_name, method)

    def get_original_method(self, method_name: str, moduleclass: Type[Module]) -> Callable:
        return self._orig_methods[method_name][moduleclass]

    def save_original_methods_for_model(self, model: Module):
        for m in model.modules():
            self.save_original_method("forward", m.__class__, m.__class__.forward)
            self.save_original_method("to", m.__class__, m.__class__.to)
            # init would already be patched now, but it was saved below

    def save_original_methods(self):
        self.save_original_method("init", Module, Module.__init__)
        self.save_original_method("to", Module, Module.to)
        self.save_original_method("forward", Sequential, Sequential.forward)

    def patch_to_and_moves(self, model: Module):
        from smdistributed.modelparallel.torch.patches.moves import distributed_to

        for m in model.modules():
            moduleclass = m.__class__
            if not self.saved_original_method("to", moduleclass):
                self.save_original_method("to", moduleclass, moduleclass.to)
            # patch uses the original method saved
            moduleclass.to = distributed_to

    def patch_constructor(self):
        # patching only Module is enough as it's the superclass and is definitely called,
        # else we duplicate calls to our custom constructor for each inherited module.
        from smdistributed.modelparallel.torch.patches import new_init

        Module.__init__ = new_init

    def patch_forward(self, model: Module):
        from smdistributed.modelparallel.torch.sequential import (
            sequential_distributed_forward,
            execute_chain_remotely,
            execute_chain,
        )
        from smdistributed.modelparallel.torch.patches.execution import (
            distributed_forward,
            execute_module,
        )

        for m in model.modules():
            moduleclass = m.__class__
            if not self.saved_original_method("forward", moduleclass):
                self.save_original_method("forward", moduleclass, moduleclass.forward)
            elif moduleclass == Sequential:
                Sequential.forward = sequential_distributed_forward
                Sequential.execute_chain = execute_chain
                Sequential.execute_chain_remotely = execute_chain_remotely
            else:
                moduleclass.execute_locally = execute_module
                moduleclass.forward = distributed_forward

    def patch_forward_for_trace(self, model: Module):
        from smdistributed.modelparallel.torch.patches.tracing import (
            trace_forward,
            trace_forward_seq,
        )

        for m in model.modules():
            if m.__class__ == Sequential:
                m.__class__.forward = trace_forward_seq
            else:
                m.__class__.forward = trace_forward

    def reset_to_and_moves(self, model: Module):
        # reset to original for tracing
        for moduleclass in set([m.__class__ for m in model.modules()]):
            moduleclass.to = self.get_original_method("to", moduleclass)

    @contextmanager
    def no_grad(self):
        from smdistributed.modelparallel.torch.state_mod import state

        prev_context = state.no_grad_context
        state.no_grad_context = True
        try:
            with self._orig_no_grad():
                yield
        finally:
            state.no_grad_context = prev_context

    def patch_no_grad(self):
        self._orig_no_grad = torch.no_grad
        torch.no_grad = self.no_grad

    def reset_no_grad(self):
        torch.no_grad = self._orig_no_grad

    @contextmanager
    def patch_for_trace(self, model, device):
        """
        Before tracing, reset forward and to method to original implementations for the module.
        Then move the model to CPU, execute and then reverse to use SMP's patches for these methods.
        """
        from smdistributed.modelparallel.torch.patches.moves import original_to

        self.patch_no_grad()
        self.patch_forward_for_trace(model)
        # to convert params to cpu, reset `to` method to original
        self.reset_to_and_moves(model)
        original_to(model, device)
        try:
            yield
        finally:
            self.patch_forward(model)
            self.patch_to_and_moves(model)
            self.reset_no_grad()
