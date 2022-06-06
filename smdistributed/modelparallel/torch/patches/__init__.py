# Standard Library
from typing import Any

# Third Party
from torch.nn import Module


def new_init(m: Module, *args: Any, **kwargs: Any):
    """
    Custom constructor which keeps track of the partition given to the context manager if used.
    If not, sets partition to None for autosplitting to fill them up.
    """
    from smdistributed.modelparallel.torch.state_mod import state

    orig_method = state.patch_manager.get_original_method("init", Module)
    orig_method(m, *args, **kwargs)
    state.module_manager.assign_partition(m)
    state.module_manager.maybe_mark_for_tensor_parallelism(m)
