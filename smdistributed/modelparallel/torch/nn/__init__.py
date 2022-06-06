# First Party
# Third Party
import torch.nn as nn

from smdistributed.modelparallel.torch.nn.dist_module import DistributedModule
from smdistributed.modelparallel.torch.nn.embedding import DistributedEmbedding
from smdistributed.modelparallel.torch.nn.layer_norm import DistributedLayerNorm
from smdistributed.modelparallel.torch.nn.linear import DistributedLinear
from smdistributed.modelparallel.torch.nn.transformer import (
    DistributedAttentionLayer,
    DistributedTransformer,
    DistributedTransformerLayer,
    DistributedTransformerLMHead,
    DistributedTransformerOutputLayer,
)

try:
    from smdistributed.modelparallel.torch.nn.layer_norm import FusedLayerNorm
except (ImportError, ModuleNotFoundError) as e:
    if "No module named 'apex'" in e.msg:
        # ignore import if apex unavailable
        pass

__all__ = [
    "FusedLayerNorm",
    "DistributedLinear",
    "DistributedLayerNorm",
    "DistributedEmbedding",
    "DistributedTransformerLayer",
    "DistributedAttentionLayer",
    "DistributedTransformerOutputLayer",
    "DistributedTransformer",
    "DistributedTransformerLMHead",
    "DistributedModule",
]
