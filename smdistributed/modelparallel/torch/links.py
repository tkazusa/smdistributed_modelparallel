# Standard Library
import bisect
import collections
from typing import NamedTuple

# First Party
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.backend.utils import find_ge
from smdistributed.modelparallel.torch.core import pp_rank
from smdistributed.modelparallel.torch.utils import rmsg

logger = get_logger()


class LinkHolder(NamedTuple):
    link_id: int
    size: int


class LinkManager:
    MAX_LINK_ID = 65536
    _id = 0

    def __init__(self):
        # has just sizes
        self._sorted_sizes = []
        # size to links dict
        self._available_links: Dict[Set[LinkHolder]] = collections.defaultdict(set)
        self._links_in_use: Set[LinkHolder] = set()

    def _get_matching_size(self, size):
        # find smallest greater than or equal to size in tree
        # if none found, insert

        matched_size = None
        try:
            matched_size = find_ge(self._sorted_sizes, size)
            if len(self._available_links[matched_size]) == 0:
                # create one with exact size if none is free
                matched_size = None
        except ValueError:
            pass

        if matched_size is None:
            bisect.insort(self._sorted_sizes, size)
            matched_size = find_ge(self._sorted_sizes, size)

        return matched_size

    def get_link(self, size):
        # update size based on previous links
        size = self._get_matching_size(size)
        # fetch link from size
        try:
            lh = self._available_links[size].pop()
        except KeyError:
            # no free link, create new one
            link_id = self._get_new_link_id(size)
            lh = LinkHolder(link_id, size)
        self._links_in_use.add(lh)
        return lh.link_id

    def _get_new_link_id(self, message_size):
        LinkManager._id = (LinkManager._id + 1) % LinkManager.MAX_LINK_ID
        # below ensures each pp_rank has a distinct range and
        # link ids don't conflict in the backend
        link_id = LinkManager._id + (LinkManager.MAX_LINK_ID * pp_rank())
        return link_id

    def reset(self):
        num_free_links = sum([len(links) for links in self._available_links.values()])
        if num_free_links:
            logger.debug(
                rmsg(
                    f"Resetting link ids. There were {num_free_links} free links "
                    f"summing up to {sum([len(links) * size for size, links in self._available_links.items()])}"
                    f"bytes of wasted memory"
                )
            )

        for lh in self._links_in_use:
            self._available_links[lh.size].add(lh)
        self._links_in_use.clear()
