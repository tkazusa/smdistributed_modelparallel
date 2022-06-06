#!/usr/bin/env python3

try:
    import smdistributed.modelparallel.tensorflow as smp
    from smdistributed.modelparallel.tensorflow import core
except ImportError:
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel.torch import core

smp.init({"partitions": 4})
core.lib.test_d2d()
