# Standard Library
import unittest

# First Party
from smdistributed.modelparallel.backend.core import Ranker


class TestRanker(unittest.TestCase):
    def test_groups(self):
        rdp_size = 4
        tp_size = 6
        pp_size = 10
        placement_strategy = "PDT"
        ranks = [17, 103, 154, 218]
        expected_pp_groups = [
            [17, 41, 65, 89, 113, 137, 161, 185, 209, 233],
            [7, 31, 55, 79, 103, 127, 151, 175, 199, 223],
            [10, 34, 58, 82, 106, 130, 154, 178, 202, 226],
            [2, 26, 50, 74, 98, 122, 146, 170, 194, 218],
        ]
        expected_tp_groups = [
            [12, 13, 14, 15, 16, 17],
            [102, 103, 104, 105, 106, 107],
            [150, 151, 152, 153, 154, 155],
            [216, 217, 218, 219, 220, 221],
        ]
        expected_rdp_groups = [
            [5, 11, 17, 23],
            [97, 103, 109, 115],
            [148, 154, 160, 166],
            [218, 224, 230, 236],
        ]
        expected_dp_groups = [
            list(range(24)),
            list(range(96, 120)),
            list(range(144, 168)),
            list(range(216, 240)),
        ]
        expected_mp_groups = [
            [12 + 24 * k + i for k in range(10) for i in range(6)],
            [6 + 24 * k + i for k in range(10) for i in range(6)],
            [6 + 24 * k + i for k in range(10) for i in range(6)],
            [24 * k + i for k in range(10) for i in range(6)],
        ]

        ranker = Ranker(placement_strategy, rdp_size, pp_size, tp_size)

        for i, rank in enumerate(ranks):
            self.assertEqual(ranker.get_dp_group(rank), expected_dp_groups[i])
            self.assertEqual(ranker.get_rdp_group(rank), expected_rdp_groups[i])
            self.assertEqual(ranker.get_tp_group(rank), expected_tp_groups[i])
            self.assertEqual(ranker.get_pp_group(rank), expected_pp_groups[i])
            self.assertEqual(ranker.get_mp_group(rank), expected_mp_groups[i])

    def test_ranks(self):
        rdp_size = 4
        tp_size = 6
        pp_size = 10
        placement_strategy = "TPD"
        ranks = [3, 5, 17, 44, 72, 103, 118, 154, 177, 200, 218, 231]
        expected_dp_ranks = [3, 1, 1, 4, 4, 11, 10, 14, 17, 20, 22, 23]
        expected_rdp_ranks = [3, 1, 1, 0, 0, 3, 2, 2, 1, 0, 2, 3]
        expected_tp_ranks = [0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 5, 5]
        expected_pp_ranks = [0, 1, 4, 1, 8, 5, 9, 8, 4, 0, 4, 7]
        expected_mp_ranks = [0, 1, 4, 11, 18, 25, 29, 38, 44, 50, 54, 57]

        ranker = Ranker(placement_strategy, rdp_size, pp_size, tp_size)

        for i, rank in enumerate(ranks):
            self.assertEqual(ranker.get_dp_rank(rank), expected_dp_ranks[i])
            self.assertEqual(ranker.get_rdp_rank(rank), expected_rdp_ranks[i])
            self.assertEqual(ranker.get_tp_rank(rank), expected_tp_ranks[i])
            self.assertEqual(ranker.get_pp_rank(rank), expected_pp_ranks[i])
            self.assertEqual(ranker.get_mp_rank(rank), expected_mp_ranks[i])


if __name__ == "__main__":
    unittest.main()
