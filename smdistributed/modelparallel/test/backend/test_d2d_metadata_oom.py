#!/usr/bin/env python3

"""
Verify that SMP backend transitions seamlessly from D2D to MPI when
D2D runs out of space to store metadata.
"""

# Standard Library
import time

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.messages import ModuleExecutionRequest
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.server_comm import ServerCommunicator

try:
    import torch
except ImportError:
    print("The D2D metadata OOM test only works for PT")
    raise

smp.init({"partitions": 2})
torch.manual_seed(123)
comm = ServerCommunicator()

# Send more than the metadata size, currently set to 10k.
COUNT = 11 * 1000

if smp.rank() == 0:

    print("Preparing tensors")
    args = [
        [torch.tensor([1.5, 1.3]).to(torch.device("cuda", smp.local_rank()))] for _ in range(COUNT)
    ]
    print("Starting test")
    smp.barrier()

    for ct in range(COUNT):
        msg = ModuleExecutionRequest(
            module="test_module",
            args=args[ct],
            kwargs={},
            execution_stack=[],
            executor=1,
            requester=0,
            mb=0,
            phase=MbStatus.FWD,
            enable_grads=True,
        )
        comm.send(msg, [1])
else:
    smp.barrier()
    ct = 0
    while True:
        if not comm.has_message():
            time.sleep(0.01)
            continue

        rcvd_msg = comm.get_next_message()
        ct += 1
        if ct == COUNT:
            break

if smp.rank() == 1:
    print("D2D metadata OOM test passed!")
smp.barrier()
