# Standard Library

# First Party
from smdistributed.modelparallel.backend.collectives import (
    CollectiveCommunicator,
    CommGroup,
    RankType,
)
from smdistributed.modelparallel.backend.exceptions import InvalidLinkIDError
from smdistributed.modelparallel.backend.logger import get_logger
from smdistributed.modelparallel.backend.utils import bijection_2d


class ModelParallelState:
    """Structure that represents the state of SMP, and holds the globally accessible objects."""

    def __init__(self):
        self.subgraph_to_device = {}
        self.device_to_subgraph = []
        self.called_sg = set()
        self.cfg = None
        self.pipeline = None
        self.core = None
        self.link_id_map = {}
        self.logger = get_logger()
        self._num_microbatches = 1
        self.initialized = False

    def _initialize_data_parallel(self, core):
        if self.cfg.horovod:
            self.horovod_init(core)

    def initialize(self, config, core):
        self.cfg = config
        self._num_microbatches = self.cfg.microbatches
        self.core = core
        self.use_control_inputs = False
        self.partition_file = (
            "./partition.data" if self.cfg.partition_file == None else self.cfg.partition_file
        )

        self._initialize_comm()

        if not self.initialized:
            self._initialize_data_parallel(self.core)

        self.initialized = True

    def _initialize_comm(self):
        self.comm = CollectiveCommunicator(self.core)

    def num_microbatches(self):
        return self._num_microbatches

    def horovod_init(self, core):
        """Subclasses extend implementation to call framework-specific Horovod initialization."""

        # Disable hvd.shutdown API since it conflicts with SMP's own
        # call of Horovod shutdown in the backend shutdown routine.
        # We cannot delegate Horovod shutdown to Horovod, because we initialize
        # Horovod in dp_groups, and the Horovod own internal shutdown signal, which normally
        # propagates through all ranks, does not propagate across dp_groups. In our backend,
        # each rank calls horovod_shutdown individually, making sure the shutdown signal
        # covers all dp_groups.
        import horovod.common.basics as hvd_basics

        hvd_basics.HorovodBasics.shutdown = lambda x: None

    def get_transaction_id(self, subgraph, count):
        """ Return a unique int for every possible 2-D input. Subgraph can be None. """
        if subgraph is None:
            sg_int = len(self.subgraph_to_device.keys())
        else:
            sg_int = sorted(self.subgraph_to_device.keys()).index(subgraph)
        return bijection_2d(sg_int, count)

    def get_link_id(self, input_id, output_id):
        """Return a unique id for smp input/output ops"""
        if (input_id, output_id) not in self.link_id_map:
            self.link_id_map[(input_id, output_id)] = len(self.link_id_map) + 1
        return self.link_id_map[(input_id, output_id)]

    def get_comm_link_id(self, mb, index, extra=0):
        """Return a unique id for comm ops"""
        if (mb, index, extra) not in self.link_id_map:
            self.link_id_map[(mb, index, extra)] = len(self.link_id_map) + 1
        return self.link_id_map[(mb, index, extra)]

    def get_att_from_link_id(self, link_id):
        for att, id in self.link_id_map.items():
            if id == link_id:
                return att
        raise InvalidLinkIDError(f"Link id {link_id} does not exist!")
