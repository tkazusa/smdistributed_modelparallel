# First Party
from smdistributed.modelparallel.torch.core import pp_rank


def get_device(param_groups):
    for group in param_groups:
        device = get_device_from_param_group(group)
        if device is not None:
            return device
    raise RuntimeError(
        f"Looks like partition {pp_rank()} did not have any gradients. Please make sure some parameters are assigned to that partition."
    )


def get_device_from_param_group(param_group):
    for param in param_group["params"]:
        if param.requires_grad is False:
            continue
        return param.device
    return None
