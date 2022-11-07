# Third Party
import torch
from smdistributed.modelparallel.torch.apex.optimizers import FusedNovoGrad as ApexFusedNovoGrad

# First Party
from smdistributed.modelparallel.torch.core import local_rank
from smdistributed.modelparallel.torch.optimizers.utils import get_device_from_param_group


class FusedNovoGrad(ApexFusedNovoGrad):
    def __init__(self, *args, **kwargs):
        super(FusedNovoGrad, self).__init__(*args, **kwargs)
        self._dummy_overflow_buf = torch.cuda.IntTensor(
            [0], device=torch.device("cuda", local_rank())
        )

    def load_state_dict(self, state_dict):
        super(ApexFusedNovoGrad, self).load_state_dict(state_dict)
        # in case exp_avg_sq is not on the same device as params, move it there
        for group in self.param_groups:
            if len(group["params"]) > 0 and "exp_avg_sq" in group:
                device = get_device_from_param_group(group)
                if device is None:
                    device = torch.device("cuda", local_rank())
                group["exp_avg_sq"][0] = group["exp_avg_sq"][0].to(device)
                group["exp_avg_sq"][1] = group["exp_avg_sq"][1].to(device)
