# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.torch.core import tp_rank
from smdistributed.modelparallel.torch.nn.dist_module import DistributedModule
from smdistributed.modelparallel.torch.nn.utils import (
    get_local_channels,
    initialize_with_input_partition,
    parameter_creation_scope,
    reduce_scatter_for_tp,
    scatter_and_merge_for_tp,
)

logger = get_logger()


class DistributedLinear(nn.Linear, DistributedModule):
    """ Tensor-parallel implementation for nn.Linear module. input_features and output_features
    attributes refer to the global size of the layer across all tp_ranks, and not the local dimensions. """

    def __init__(self, input_features, output_features, bias=True):
        super(nn.Linear, self).__init__()

        self.in_features = get_local_channels(input_features)
        self.out_features = output_features
        self.batch_dim = 0

        with parameter_creation_scope(module=self, scaled_batch=True):
            with initialize_with_input_partition(self, exclude_from_dist=["bias"]):
                self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
                if bias and tp_rank() == 0:
                    self.bias = nn.Parameter(torch.Tensor(self.out_features))
                else:
                    self.register_parameter("bias", None)
                # reset_parameters() initializes the parameters
                self.reset_parameters()

                # need to call this after reset, otherwise reset will make the params non zero
                if bias and tp_rank() != 0:
                    self.register_parameter("bias", None)

    @staticmethod
    def can_distribute(*args, **kwargs):
        if len(args) > 0:
            in_features = args[0]
        elif "input_features" in kwargs:
            in_features = kwargs["input_features"]
        else:
            logger.info(
                f"Skipping distribution of nn.Linear because the number of input channels is not specified."
            )
            return False

        return True

    def forward(self, inp):
        full_inputs = scatter_and_merge_for_tp(inp, -1, self.batch_dim)
        linear_output = F.linear(full_inputs, self.weight, self.bias)
        return reduce_scatter_for_tp(linear_output, self.batch_dim)
