# Standard Library
import unittest

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.pipeline import MbStatus
from smdistributed.modelparallel.torch.state_mod import state


class TestPipeline(unittest.TestCase):
    def test_interleaved(self):
        num_microbatches = 5
        smp.init({"microbatches": num_microbatches, "pipeline": "interleaved", "partitions": 1})
        p = state.pipeline
        c = 0
        none_count = 0
        while p.has_more_ticks():
            mb = p.get_next_microbatch()
            mb_status = p.get_status(mb) if mb is not None else None
            if c == 0:
                assert mb == 0, p.status
                p.promote_status(mb)
            elif c == 1:
                # mb = 0 is FWD
                assert mb == 1, p.status
                p.promote_status(mb)
                p.mark_ready_for_backward(0)
            elif c == 2:
                # mb = 0 is READY_FOR_BWD
                # mb = 1 is FWD
                assert mb == 0, p.status
                p.promote_status(mb)
            elif c == 3:
                # mb = 0 is BWD
                # mb = 1 FWD
                assert mb == 2, p.status
                p.promote_status(mb)
                p.mark_ready_for_backward(1)
                p.mark_done(0)
            elif c == 4:
                # mb = 0 is DONE
                # mb = 1 READY_FOR_BWD
                # mb = 2 FWD
                assert mb == 1, p.status
                p.promote_status(mb)
            elif c == 5:
                # mb = 0 is DONE
                # mb = 1 is BWD
                # mb = 2 is FWD
                assert mb == 3, p.status
                p.promote_status(mb)
            elif c == 6:
                # mb = 0 is DONE
                # mb = 1 is BWD
                # mb = 2 is FWD
                # mb = 3 is FWD
                assert mb == 4, p.status
                p.promote_status(mb)
                p.mark_ready_for_backward(mb)
            elif c == 7:
                # mb = 0 is DONE
                # mb = 1 is BWD
                # mb = 2 is FWD
                # mb = 3 is FWD
                # mb = 4 is READY_FOR_BWD
                # all mb are in progress, so mb is None
                assert mb == 4, p.status
                p.promote_status(mb)
                p.mark_done(1)
            elif c == 8:
                # mb = 0 is DONE
                # mb = 1 is DONE
                # mb = 2 is FWD
                # mb = 3 is FWD
                # mb = 4 is BWD
                assert mb is None, p.status
                none_count += 1
                p.mark_done(4)
                print(p.status)
            elif c == 9:
                # mb = 0 is DONE
                # mb = 1 is DONE
                # mb = 2 is FWD
                # mb = 3 is FWD
                # mb = 4 is DONE
                assert mb is None, p.status
                none_count += 1
                p.mark_ready_for_backward(2)
            elif c == 10:
                # mb = 0 is DONE
                # mb = 1 is DONE
                # mb = 2 is READY_FOR_BWD
                # mb = 3 is FWD
                # mb = 4 is DONE
                assert mb == 2, p.status
                p.promote_status(mb)
                p.mark_ready_for_backward(3)
            elif c == 11:
                # mb = 0 is DONE
                # mb = 1 is DONE
                # mb = 2 is BWD
                # mb = 3 is READY_FOR_BWD
                # mb = 4 is DONE
                assert mb == 3, p.status
                p.promote_status(mb)
            elif c == 12:
                # mb = 0 is DONE
                # mb = 1 is DONE
                # mb = 2 is BWD
                # mb = 3 is DONE
                # mb = 4 is DONE
                assert mb is None, p.status
                none_count += 1
                p.mark_done(3)
            elif c == 13:
                assert mb is None, p.status
                none_count += 1
                p.mark_done(2)
            c += 1
        assert c == num_microbatches * 2 + none_count

    def test_simple(self):
        num_microbatches = 5
        # TODO: remove active_microbatches arg after this is fixed for simple pipeline
        smp.init(
            {
                "microbatches": num_microbatches,
                "active_microbatches": num_microbatches,
                "pipeline": "simple",
                "partitions": 1,
            }
        )
        p = state.pipeline
        c = 0
        none_count = 0
        while p.has_more_ticks():
            mb = p.get_next_microbatch()
            print(c, p.status)
            mb_status = p.get_status(mb) if mb is not None else None
            if c < 5:
                assert mb_status == MbStatus.READY_FOR_FWD
            elif c in [5, 6, 7]:
                assert mb_status == None
            else:
                assert mb_status == MbStatus.READY_FOR_BWD

            if c == 0:
                assert mb == 0
                p.promote_status(mb)
                p.mark_ready_for_backward(0)
            elif c == 1:
                # mb = 0 is READY_FOR_BWD
                # simple pipeline so we want to finish mb1 fwd before bwd0
                assert mb == 1
                p.promote_status(mb)
            elif c == 2:
                # mb = 0 is READY_FOR_BWD
                # mb = 1 is FWD
                assert mb == 2
                p.promote_status(mb)
                p.mark_ready_for_backward(2)
            elif c in [3, 4]:
                p.promote_status(mb)
            elif c == 5:
                # all fwd not done yet
                assert mb is None
                none_count += 1
                p.mark_ready_for_backward(1)
            elif c == 6:
                assert mb is None
                none_count += 1
                p.mark_ready_for_backward(3)
            elif c == 7:
                assert mb is None
                none_count += 1
                p.mark_ready_for_backward(4)
            elif c == 8:
                assert mb == 0
                p.promote_status(mb)
                p.mark_done(mb)
            elif c == 9:
                assert mb == 1
                p.promote_status(mb)
            elif c == 10:
                assert mb == 2
                p.promote_status(mb)
                p.mark_done(mb)
            elif c == 11:
                assert mb == 3
                p.promote_status(mb)
                p.mark_done(1)
                p.mark_done(3)
            elif c == 12:
                assert mb == 4
                p.promote_status(mb)
                p.mark_done(mb)
            elif c > 12:
                assert False, p.status
            c += 1
        # minimum number of steps is num_microbatches * 2
        # if at any stage appropriate ready_for_backward is not done, then c will be higher
        assert c == num_microbatches * 2 + none_count


if __name__ == "__main__":
    unittest.main()
