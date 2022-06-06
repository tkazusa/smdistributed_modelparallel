# Standard Library
import unittest

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.links import LinkManager


class TestLinkManager(unittest.TestCase):
    def test_links(self):
        smp.init({"partitions": 1})

        lm = LinkManager()
        l1 = lm.get_link(10)
        l2 = lm.get_link(5)
        l3 = lm.get_link(15)
        l4 = lm.get_link(10)
        print(l1, l2, l3, l4)
        lm.reset()
        print(lm._available_links, lm._links_in_use)
        l5 = lm.get_link(10)
        assert l5 in [l1, l4], l5
        l6 = lm.get_link(5)
        assert l2 == l6
        l7 = lm.get_link(8)
        assert l7 in [l1, l4]
        l8 = lm.get_link(15)
        assert l8 == l3
        l9 = lm.get_link(21)
        l10 = lm.get_link(21)
        assert l9 != l10


if __name__ == "__main__":
    unittest.main()
