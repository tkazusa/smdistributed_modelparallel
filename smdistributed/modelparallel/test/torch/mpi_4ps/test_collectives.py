# Standard Library
import unittest

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.state_mod import state


class TestPTCollectives(unittest.TestCase):
    def verify_scatter_and_merge(self, tensors, split_axis, merge_axis, expected_results):
        smp.init({"partitions": 1, "ddp": True})

        rank_expected_result = expected_results[smp.dp_rank()]

        cpu_tensor = tensors[smp.dp_rank()].to(torch.device("cpu"))
        cpu_res = state.comm.scatter_and_merge_tensor(
            cpu_tensor, split_axis, merge_axis, smp.DP_GROUP
        )
        self.assertTrue(torch.all(cpu_res.cpu().eq(rank_expected_result.cpu())))

        gpu_tensor = tensors[smp.dp_rank()].to(torch.device("cuda", smp.local_rank()))
        gpu_res = state.comm.scatter_and_merge_tensor(
            gpu_tensor, split_axis, merge_axis, smp.DP_GROUP
        )
        self.assertTrue(torch.all(gpu_res.cpu().eq(rank_expected_result.cpu())))

    def verify_allgatherv(self, tensors, counts, expected_result):
        smp.init({"partitions": 1, "ddp": True})

        gpu_tensor = tensors[smp.dp_rank()].to(torch.device("cuda", smp.local_rank()))
        state.comm.allgatherv_tensor(gpu_tensor, counts, smp.DP_GROUP)
        self.assertTrue(torch.all(gpu_tensor.cpu().eq(expected_result.cpu())))

    def test_simple_allgatherv(self):
        tensors = [r * torch.ones(16) for r in range(smp.dp_size())]
        counts = [4 for _ in range(smp.dp_size())]
        expected_result = torch.tensor([float(r // 4) for r in range(16)])

        self.verify_allgatherv(tensors, counts, expected_result)

    def test_variable_allgatherv(self):
        tensors = [r * torch.ones(16) for r in range(smp.dp_size())]
        counts = [2, 5, 3, 6]
        expected = []
        for i, ct in enumerate(counts):
            expected.extend([float(i) for _ in range(ct)])
        expected_result = torch.tensor(expected)

        self.verify_allgatherv(tensors, counts, expected_result)

    def test_gather(self):
        smp.init({"partitions": 1, "ddp": True})
        tensors = [r * torch.ones(16) for r in range(smp.dp_size())]
        other_vals = [r for r in range(smp.dp_size())]
        result = [{"a": tensors[r], "b": other_vals[r]} for r in range(smp.dp_size())]
        out = state.comm.gather_large(result[smp.dp_rank()], group=smp.DP_GROUP, rank=0)
        if smp.dp_rank() == 0:
            for item1, item2 in zip(out, result):
                self.assertTrue(torch.all(item1["a"].eq(item2["a"])))
                assert item1["b"] == item2["b"]

    def test_same_axis(self):
        tensors = [
            torch.tensor([rank for _ in range(4)]).to(torch.device("cuda", rank))
            for rank in range(4)
        ]
        split_axis = 0
        merge_axis = 0
        expected = [torch.arange(0, 4).to(torch.device("cuda", rank)) for rank in range(4)]

        self.verify_scatter_and_merge(tensors, split_axis, merge_axis, expected)

    def test_different_axis(self):
        tensors = [
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            torch.tensor([[6, 7, 8, 9], [10, 11, 12, 13]]),
            torch.tensor([[9, 8, 7, 6], [5, 4, 3, 2]]),
            torch.tensor([[2, 4, 6, 8], [10, 12, 14, 16]]),
        ]
        split_axis = 1
        merge_axis = 0
        expected = [
            torch.tensor([[1], [5], [6], [10], [9], [5], [2], [10]]),
            torch.tensor([[2], [6], [7], [11], [8], [4], [4], [12]]),
            torch.tensor([[3], [7], [8], [12], [7], [3], [6], [14]]),
            torch.tensor([[4], [8], [9], [13], [6], [2], [8], [16]]),
        ]
        tensors = [t.to(torch.device("cuda", rank)) for rank, t in enumerate(tensors)]
        expected = [e.to(torch.device("cuda", rank)) for rank, e in enumerate(expected)]

        self.verify_scatter_and_merge(tensors, split_axis, merge_axis, expected)

    def test_nondivisible(self):
        tensors = [
            torch.tensor([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10]]),
            torch.tensor([[6, 7, 8, 9, 10, 11], [10, 11, 12, 13, 14, 15]]),
            torch.tensor([[9, 8, 7, 6, 5, 4], [5, 4, 3, 2, 1, 0]]),
            torch.tensor([[2, 4, 6, 8, 10, 12], [10, 12, 14, 16, 18, 20]]),
        ]
        split_axis = 1
        merge_axis = 0
        expected = [
            torch.tensor([[1, 2], [5, 6], [6, 7], [10, 11], [9, 8], [5, 4], [2, 4], [10, 12]]),
            torch.tensor([[3, 4], [7, 8], [8, 9], [12, 13], [7, 6], [3, 2], [6, 8], [14, 16]]),
            torch.tensor([[5], [9], [10], [14], [5], [1], [10], [18]]),
            torch.tensor([[6], [10], [11], [15], [4], [0], [12], [20]]),
        ]
        tensors = [t.to(torch.device("cuda", rank)) for rank, t in enumerate(tensors)]
        expected = [e.to(torch.device("cuda", rank)) for rank, e in enumerate(expected)]

        self.verify_scatter_and_merge(tensors, split_axis, merge_axis, expected)

    def test_allgather_cpu(self):
        smp.init({"partitions": 1, "ddp": True})
        tensors = [r * torch.ones(16) for r in range(smp.dp_size())]
        result = smp.allgather(tensors[smp.dp_rank()].cpu(), group=smp.DP_GROUP)
        for item1, item2 in zip(result, tensors):
            self.assertTrue(torch.all(item1.eq(item2)))


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
