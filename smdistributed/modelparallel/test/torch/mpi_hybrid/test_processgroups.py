# Standard Library
import unittest

# Third Party
import torch
import torch.distributed as dist

# First Party
import smdistributed.modelparallel.torch as smp


class TestProcessGroups(unittest.TestCase):
    def test_groups(self):
        smp.init({"partitions": 2, "ddp": True, "microbatches": 4})
        assert smp.rank() == dist.get_rank()
        device = torch.device("cuda", smp.local_rank())
        # torch.cuda.set_device(smp.local_rank())

        dp_group = smp.get_dp_process_group()

        tensors = []
        for i in range(len(smp.get_dp_group())):
            tensors.append(torch.zeros((1,), device=device))
        dist.all_gather(tensors, torch.ones((1,), device=device) * (smp.rank() + 1), group=dp_group)
        ranks = [tensor.item() - 1 for tensor in tensors]
        assert ranks == smp.get_dp_group()
        assert ranks == smp.allgather(smp.rank(), group=smp.CommGroup.DP_GROUP)

        pp_group = smp.get_pp_process_group()
        tensors = []
        for i in range(len(smp.get_pp_group())):
            tensors.append(torch.zeros((1,), device=device))
        dist.all_gather(tensors, torch.ones((1,), device=device) * (smp.rank() + 1), group=pp_group)
        ranks = [tensor.item() - 1 for tensor in tensors]
        assert ranks == smp.get_pp_group()
        assert ranks == smp.allgather(smp.rank(), group=smp.CommGroup.PP_GROUP)

        world_group = smp.get_world_process_group()
        tensors = []
        for i in range(smp.size()):
            tensors.append(torch.zeros((1,), device=device))
        dist.all_gather(
            tensors, torch.ones((1,), device=device) * (smp.rank() + 1), group=world_group
        )
        ranks = [tensor.item() - 1 for tensor in tensors]
        assert ranks == list(range(smp.size()))
        assert ranks == smp.allgather(smp.rank(), group=smp.CommGroup.WORLD)

        # no group arg
        dist.all_gather(tensors, torch.ones((1,), device=device) * (smp.rank() + 1))
        ranks = [tensor.item() - 1 for tensor in tensors]
        assert ranks == list(range(smp.size()))
        assert ranks == smp.allgather(smp.rank(), group=smp.CommGroup.WORLD)

        assert smp.dp_rank() == dist.get_rank(group=smp.get_dp_process_group())
        assert smp.pp_rank() == dist.get_rank(group=smp.get_pp_process_group())

    def tearDown(self) -> None:
        dist.barrier()


if __name__ == "__main__":
    unittest.main()
