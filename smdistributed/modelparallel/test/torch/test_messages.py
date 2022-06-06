# Standard Library
import collections
import unittest

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.utils import Net1, mock_get_module_fns
from smdistributed.modelparallel.torch.messages import (
    ForwardExecutionResult,
    MicrobatchEndResult,
    ModuleExecutionRequest,
)
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.serialization import SerializationManager, TensorStub


class TestStubbedMsg(unittest.TestCase):
    def setUp(self):
        import smdistributed.modelparallel.torch.core as core

        core.pp_rank = lambda: 0

    def test_exec_request_single_tensor(self):
        t = torch.ones(1, 2, requires_grad=True)
        m = ModuleExecutionRequest("main", (t,), {}, ["main"], 0, 0, 0, MbStatus.FWD, True)
        s = SerializationManager()
        m, tensors = s.serialize(m, False, [])
        assert isinstance(m.args[0], TensorStub), m.args[0]
        assert m.args[0].tensor_index == 0
        assert m.args[0].dtype == t.dtype
        assert m.args[0].shape == t.shape

        t_recv = torch.ones(1, 2)
        msg = s.deserialize(m, (t_recv,))
        assert isinstance(msg.args[0], torch.Tensor)
        assert msg.args[0].requires_grad == t_recv.requires_grad
        assert torch.all(msg.args[0].eq(t))

    def test_exec_request_mult_tensors(self):
        t = (torch.ones(1, 2), torch.ones(2, 3))
        m = ModuleExecutionRequest("main", t, {}, ["main"], 0, 0, 0, MbStatus.FWD, True)
        s = SerializationManager()
        m, tx_list = s.serialize(m, False, [])
        assert len(tx_list) == 2
        assert isinstance(m.args[0], TensorStub), m.args[0]
        assert isinstance(m.args[1], TensorStub), m.args[1]
        msg = s.deserialize(m, [t.tensor for t in tx_list])
        assert isinstance(msg.args[0], torch.Tensor)
        assert torch.all(msg.args[0].eq(t[0]))
        assert isinstance(msg.args[1], torch.Tensor)
        assert torch.all(msg.args[1].eq(t[1]))

    def test_exec_request_dict_tensors(self):
        t = (torch.ones(1, 2), torch.ones(2, 3))
        tt = {"a": torch.ones(2, 2), "b": torch.zeros(2, 1)}
        m = ModuleExecutionRequest("main", t, tt, ["main"], 0, 0, 0, MbStatus.FWD, True)
        s = SerializationManager()
        m, tx_list = s.serialize(m, False, [])
        assert len(tx_list) == 4
        assert isinstance(m.args[0], TensorStub), m.args[0]
        assert isinstance(m.args[1], TensorStub), m.args[1]
        assert isinstance(m.kwargs["a"], TensorStub), m.kwargs["a"]
        assert isinstance(m.kwargs["b"], TensorStub), m.kwargs["b"]

        msg = s.deserialize(m, [t.tensor for t in tx_list])
        assert isinstance(msg.args[0], torch.Tensor)
        assert torch.all(msg.args[0].eq(t[0]))
        assert isinstance(msg.args[1], torch.Tensor)
        assert torch.all(msg.args[1].eq(t[1]))
        assert isinstance(msg.kwargs["a"], torch.Tensor)
        assert torch.all(msg.kwargs["a"].eq(tt["a"]))
        assert isinstance(msg.kwargs["b"], torch.Tensor)
        assert torch.all(msg.kwargs["b"].eq(tt["b"]))

    def test_exec_request_namedtuple_tensors(self):
        PointXY = collections.namedtuple("Point", ["x", "y"])
        tt = PointXY(x=torch.ones(10), y=torch.ones((1, 10)))
        m = ModuleExecutionRequest("main", tt, {}, ["main"], 0, 0, 0, MbStatus.FWD, True)
        s = SerializationManager()
        m, tx_list = s.serialize(m, False, [])
        assert len(tx_list) == 2
        assert isinstance(m.args[0], TensorStub), m.args[0]
        assert isinstance(m.args[1], TensorStub), m.args[1]
        msg = s.deserialize(m, [t.tensor for t in tx_list])
        assert isinstance(msg.args[0], torch.Tensor)
        assert torch.all(msg.args[0].eq(tt[0]))
        assert isinstance(msg.args[1], torch.Tensor)
        assert torch.all(msg.args[1].eq(tt[1]))

    def test_exec_result_tensor(self):
        t = (torch.ones(1, 2), torch.ones(2, 3))
        tt = {"a": torch.ones(2, 2), "b": torch.zeros(2, 1)}
        m = ModuleExecutionRequest("main", t, tt, ["main"], 0, 0, 0, MbStatus.FWD, True)
        outputs = (torch.ones(2, 3),)
        r = ForwardExecutionResult(m, outputs)
        s = SerializationManager()
        r, tx_list = s.serialize(r, False, [])
        assert len(tx_list) == 1
        assert len(r.request.args) == 0
        assert len(r.request.kwargs) == 0
        assert isinstance(r.outputs[0], TensorStub), r.outputs[0]

        msg = s.deserialize(r, [t.tensor for t in tx_list])
        assert len(msg.request.args) == 0
        assert len(msg.request.kwargs) == 0
        assert isinstance(msg.outputs[0], torch.Tensor)
        assert torch.all(msg.outputs[0].eq(outputs[0]))

    def test_exec_result_dict_tensor(self):
        t = (torch.ones(1, 2), torch.ones(2, 3))
        m = ModuleExecutionRequest("main", t, {}, ["main"], 0, 0, 0, MbStatus.FWD, True)
        outputs = {"a": torch.ones(2, 3), "b": torch.zeros(2)}
        r = ForwardExecutionResult(m, outputs)
        s = SerializationManager()
        r, tx_list = s.serialize(r, False, [])
        assert len(tx_list) == 2
        assert len(r.request.args) == 0
        assert isinstance(r.outputs["a"], TensorStub), r.outputs["a"]
        assert isinstance(r.outputs["b"], TensorStub), r.outputs["b"]

        msg = s.deserialize(r, [t.tensor for t in tx_list])
        assert len(msg.request.args) == 0
        assert isinstance(msg.outputs["a"], torch.Tensor)
        assert torch.all(msg.outputs["a"].eq(outputs["a"]))
        assert isinstance(msg.outputs["b"], torch.Tensor)
        assert torch.all(msg.outputs["b"].eq(outputs["b"]))

    def test_mb_end_result_dict_tensor(self):
        outputs = {
            "a": torch.ones(2, 3),
            "b": torch.zeros(2),
            "c": [torch.zeros(2, 3), torch.zeros(2)],
        }
        sg1 = Net1()

        with mock_get_module_fns(parent_module=sg1):
            r = MicrobatchEndResult(0, outputs)
            s = SerializationManager()
            r, tx_list = s.serialize(r, False, [])
        assert len(tx_list) == 4
        assert isinstance(r.outputs["a"], TensorStub)
        assert isinstance(r.outputs["b"], TensorStub)
        assert isinstance(r.outputs["c"][0], TensorStub)
        assert isinstance(r.outputs["c"][1], TensorStub)

        with mock_get_module_fns(parent_module=sg1):
            msg = s.deserialize(r, [t.tensor for t in tx_list])
        assert isinstance(msg.outputs["a"], torch.Tensor)
        assert torch.all(msg.outputs["a"].eq(outputs["a"]))
        assert isinstance(msg.outputs["b"], torch.Tensor)
        assert torch.all(msg.outputs["b"].eq(outputs["b"]))
        assert isinstance(msg.outputs["c"][0], torch.Tensor)
        assert torch.all(msg.outputs["c"][0].eq(outputs["c"][0]))
        assert isinstance(msg.outputs["c"][1], torch.Tensor)
        assert torch.all(msg.outputs["c"][1].eq(outputs["c"][1]))


if __name__ == "__main__":
    smp.init({"microbatches": 2, "pipeline": "simple", "partitions": 1})
    unittest.main()
