#!/usr/bin/env python3

try:
    import smdistributed.modelparallel.tensorflow as smp
    from smdistributed.modelparallel.tensorflow import core
except ImportError:
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel.torch import core

smp.init({"partitions": 1})
core.lib.from_blob_tensor_test()
