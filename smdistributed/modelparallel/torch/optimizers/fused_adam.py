# Third Party
import torch
from smdistributed.modelparallel.torch.apex.optimizers import FusedAdam as ApexFusedAdam

# First Party
from smdistributed.modelparallel.torch import local_rank


class FusedAdam(ApexFusedAdam):
    def __init__(self, *args, **kwargs):
        torch.cuda.set_device(local_rank())
        super(FusedAdam, self).__init__(*args, **kwargs)
