# Standard Library
import unittest

# First Party
from smdistributed.modelparallel.backend.collectives import TransactionIdentifier

try:
    import smdistributed.modelparallel.tensorflow as smp
    from smdistributed.modelparallel.tensorflow import state
except ImportError:
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel.torch import state


class TestClusterCollectives(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestClusterCollectives, self).__init__(*args, **kwargs)
        self.init_smp()

    def init_smp(self):
        if not state.initialized:
            self.num_partitions = 2
            smp.init(
                {
                    "partitions": self.num_partitions,
                    "horovod": True,
                    "placement_strategy": "cluster",
                }
            )

    def tearDown(self):
        smp.barrier()

    def test_world_bcast_async(self):
        transaction_ids = [TransactionIdentifier(k, False) for k in [42, 36, 111]]
        data = [3, 5, ["a", "b"]], {"a": 3, "v": 7}, "test"

        if smp.rank() == 0:
            state.comm.async_bcast(data[0], transaction_ids[0], group=smp.WORLD)
            state.comm.async_bcast(data[1], transaction_ids[1], group=smp.WORLD)
            state.comm.wait_all_transactions()
        else:
            state.comm.async_recv_from(0, transaction_ids[0], smp.RankType.WORLD_RANK)
            state.comm.async_recv_from(0, transaction_ids[1], smp.RankType.WORLD_RANK)
            rcv0 = state.comm.wait_recv(0, transaction_ids[0], smp.RankType.WORLD_RANK)
            rcv1 = state.comm.wait_recv(0, transaction_ids[1], smp.RankType.WORLD_RANK)

            assert data[0] == rcv0 and data[1] == rcv1, (rcv0, rcv1)
        smp.barrier()
        ids = [smp.rank()]
        expected_res = [[i] for i in range(smp.size())]
        all_ids = state.comm.allgather(ids, group=smp.WORLD)
        assert all_ids == expected_res, (all_ids, expected_res)

    def test_bcast_mpgroup(self):
        if smp.pp_rank() == 0:
            state.comm.broadcast(smp.dp_rank(), group=smp.PP_GROUP)
        else:
            r = state.comm.recv_from(0, smp.RankType.PP_RANK)
            assert r == smp.dp_rank()

    def test_bcast_world(self):
        if smp.rank() == 0:
            state.comm.broadcast("a", group=smp.WORLD)
        else:
            r = state.comm.recv_from(0, smp.RankType.WORLD_RANK)
            assert r == "a"

        if smp.rank() == 0:
            smp.broadcast("a", group=smp.CommGroup.WORLD)
        else:
            r = smp.recv_from(0, smp.RankType.WORLD_RANK)
            assert r == "a"

    def test_bcast_dpgroup(self):
        if smp.dp_rank() == 0:
            state.comm.broadcast(smp.pp_rank(), group=smp.DP_GROUP)
        else:
            r = state.comm.recv_from(0, smp.RankType.DP_RANK)
            assert r == smp.pp_rank()

    def test_send_world(self):
        dest_rank = (smp.rank() + 1) % smp.size()
        state.comm.send(smp.rank(), dest_rank, rank_type=smp.RankType.WORLD_RANK)

        src_rank = smp.rank() - 1 if smp.rank() > 0 else smp.size() - 1
        r = state.comm.recv_from(src_rank, rank_type=smp.RankType.WORLD_RANK)
        assert r == src_rank

        dest_rank = (smp.rank() + 1) % smp.size()
        smp.send(smp.rank(), dest_rank, rank_type=smp.RankType.WORLD_RANK)

        src_rank = smp.rank() - 1 if smp.rank() > 0 else smp.size() - 1
        r = smp.recv_from(src_rank, rank_type=smp.RankType.WORLD_RANK)
        assert r == src_rank

    def test_send_mpgroup(self):
        dest_rank = (smp.pp_rank() + 1) % smp.pp_size()
        state.comm.send(smp.pp_rank(), dest_rank, rank_type=smp.RankType.PP_RANK)

        src_rank = smp.pp_rank() - 1 if smp.pp_rank() > 0 else smp.pp_size() - 1
        r = state.comm.recv_from(src_rank, rank_type=smp.RankType.PP_RANK)
        assert r == src_rank

    def test_send_dpgroup(self):
        dest_rank = (smp.dp_rank() + 1) % smp.dp_size()
        state.comm.send(smp.dp_rank(), dest_rank, rank_type=smp.RankType.DP_RANK)

        src_rank = smp.dp_rank() - 1 if smp.dp_rank() > 0 else smp.dp_size() - 1
        r = state.comm.recv_from(src_rank, rank_type=smp.RankType.DP_RANK)
        assert r == src_rank

    def test_allgather_world(self):
        ranks = state.comm.allgather(smp.rank(), group=smp.WORLD)
        assert ranks == list(range(state.core.size())), ranks

        ranks = smp.allgather(smp.rank(), group=smp.WORLD)
        assert ranks == list(range(state.core.size())), ranks

    def test_allgather_mpgroup(self):
        ranks = state.comm.allgather(smp.rank(), group=smp.PP_GROUP)
        assert ranks == state.core.get_pp_group()

        ranks = smp.allgather(smp.rank(), group=smp.PP_GROUP)
        assert ranks == state.core.get_pp_group()

    def test_allgather_dpgroup(self):
        ranks = state.comm.allgather(smp.rank(), group=smp.CommGroup.DP_GROUP)
        assert ranks == state.core.get_dp_group()

        ranks = smp.allgather(smp.rank(), group=smp.CommGroup.DP_GROUP)
        assert ranks == state.core.get_dp_group()

    def test_pp_barrier(self):
        state.comm.pp_barrier()
        smp.pp_barrier()
        smp.barrier(group=smp.CommGroup.PP_GROUP)

    def test_dp_barrier(self):
        state.comm.dp_barrier()
        smp.dp_barrier()
        smp.barrier(group=smp.CommGroup.DP_GROUP)


class TestSpreadCollectives(TestClusterCollectives):
    def init_smp(self):
        if not state.initialized:
            self.num_partitions = 2
            smp.init(
                {"partitions": self.num_partitions, "horovod": True, "placement_strategy": "spread"}
            )


if __name__ == "__main__":
    unittest.main()
