# Standard Library
from collections import defaultdict
from distutils.version import LooseVersion

# Third Party
import torch
from torch.cuda.amp import GradScaler as TorchGradScaler
from torch.cuda.amp.grad_scaler import OptState, _refresh_per_optimizer_state

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.comm import CommGroup, allgather
from smdistributed.modelparallel.torch.core import local_rank, pp_rank

# TODO: update this file with changes made in PT 1.8 for typechecking
# no functionality change in 1.8

_pt_19_or_newer = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

class GradScaler(TorchGradScaler):
    def __init__(self, *args, **kwargs):
        super(GradScaler, self).__init__(*args, **kwargs)
        self._warned_custom_scaling = False
        self.local_device = torch.device("cuda", local_rank())

    def update(self, new_scale=None):
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the scale directly.

        Arguments:
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return

        self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale = torch.full(
                    (1,), new_scale, dtype=torch.float32, device=self._scale.device
                )
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale = new_scale
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=self._scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            found_inf_combined_ranks = sum(
                allgather(found_inf_combined.to(torch.device("cpu")), CommGroup.PP_GROUP)
            )
            found_inf_combined_ranks = found_inf_combined_ranks.to(self.local_device)

            if _pt_19_or_newer:
                self._scale = torch._amp_update_scale_(
                    self._scale,
                    self._growth_tracker,
                    found_inf_combined_ranks,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )
            else:
                self._scale = torch._amp_update_scale(
                    self._growth_tracker,
                    self._scale,
                    found_inf_combined_ranks,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def step(self, optimizer, *args, **kwargs):
        """
            :meth:`step` carries out the following two operations:

            1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
                earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
            2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
                gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

            ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

            Returns the return value of ``optimizer.step(*args, **kwargs)``.

            Arguments:
                optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
                args:  Any arguments.
                kwargs:  Any keyword arguments.

            .. warning::
                Closure use is not currently supported.
            """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")

        if pp_rank() != 0 and self._scale is None:
            self._lazy_init_scale_growth_tracker(torch.device("cuda", local_rank()))

        self._check_scale_growth_tracker("step")
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        retval = None

        if (
            hasattr(optimizer, "_step_supports_amp_scaling")
            and optimizer._step_supports_amp_scaling
        ):
            if not self._warned_custom_scaling:
                get_logger().warning(
                    "Optimizer with custom gradient-scale-handling logic detected. For correct results with SMP, the boolean \
                                     that represents whether there is gradient overflow must be allgathered across pp_ranks using SMP allgather API."
                )
                self._warned_custom_scaling = True

            # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
            # The contract with custom optimizers is that their step() should accept an additional,
            # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale_ on itself, etc
            retval = optimizer.step(*args, **dict(kwargs, grad_scaler=self))
            optimizer_state["stage"] = OptState.STEPPED
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        if len(optimizer_state["found_inf_per_device"]) == 0:
            from smdistributed.modelparallel.torch.state_mod import state

            if len(list(state.model.local_parameters())) == 0:
                # In SMP's case if no params were assigned to rank this can be empty
                optimizer_state["found_inf_per_device"] = {
                    self.local_device: torch.zeros((1,), device=self.local_device)
                }

        assert (
            len(optimizer_state["found_inf_per_device"]) > 0
        ), "No inf checks were recorded for this optimizer."

        local_infinite = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())

        any_infinite = sum(allgather(local_infinite, CommGroup.PP_GROUP))

        if not any_infinite:
            from smdistributed.modelparallel.torch.state_mod import state

            # some optimizers like adadelta from PT 1.8 dont like it when optimizer.step is called with no param
            if len(list(state.model.local_parameters())) > 0:
                retval = optimizer.step(*args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval
