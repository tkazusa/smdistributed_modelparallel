# First Party

try:
    from smdistributed.modelparallel.torch.optimizers.fused_adam import FusedAdam
    from smdistributed.modelparallel.torch.optimizers.fused_lamb import FusedLAMB
    from smdistributed.modelparallel.torch.optimizers.fused_novograd import FusedNovoGrad
except ModuleNotFoundError as e:
    if "No module named 'apex'" in e.msg:
        # ignore import if apex unavailable
        pass

__all__ = ["FusedAdam", "FusedLAMB", "FusedNovoGrad"]
