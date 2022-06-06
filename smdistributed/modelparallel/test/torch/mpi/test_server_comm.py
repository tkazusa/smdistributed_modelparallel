# Standard Library
import time

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.messages import ModuleExecutionRequest
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.server_comm import ServerCommunicator


def assert_same(obj1, obj2):
    if isinstance(obj1, torch.Tensor):
        assert torch.all(obj1.cpu().eq(obj2.cpu()))
    elif isinstance(obj1, (list, tuple)):
        for item1, item2 in zip(obj1, obj2):
            assert_same(item1, item2)
    elif isinstance(obj1, dict):
        for k in obj1:
            assert_same(obj1[k], obj2[k])
    else:
        assert obj1 == obj2


smp.init({"partitions": 2})
torch.manual_seed(123)
comm = ServerCommunicator()

COUNT = 5
args = [[torch.tensor([1.2, 3.4])] for _ in range(COUNT)]
kwargs = [
    {"a": 34, "b": torch.randn([400, 400]), "c": [torch.randn([40, 40]), 34]} for _ in range(COUNT)
]

if smp.rank() == 0:
    for ct in range(COUNT):
        msg = ModuleExecutionRequest(
            module="test_module",
            args=args[ct],
            kwargs=kwargs[ct],
            execution_stack=[],
            executor=1,
            requester=0,
            mb=0,
            phase=MbStatus.FWD,
            enable_grads=True,
        )
        comm.send(msg, [1])
else:
    ct = 0
    while True:
        if not comm.has_message():
            time.sleep(0.01)
            continue

        rcvd_msg = comm.get_next_message()
        assert_same(rcvd_msg.args, args[ct])
        assert_same(rcvd_msg.kwargs, kwargs[ct])
        ct += 1
        if ct == COUNT:
            break

if smp.rank() == 1:
    print("Server communicator test passed!")
smp.barrier()
