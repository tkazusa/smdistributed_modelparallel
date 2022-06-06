# Standard Library
import collections
import copy
import logging

# Third Party
import torch.nn as nn

# First Party
from smdistributed.modelparallel.backend.logger import get_log_level, get_logger
from smdistributed.modelparallel.torch.utils import dtype_size, product

logger = get_logger()
NodeCost = collections.namedtuple("NodeCost", "computation parameters activations")
MB = 1024 * 1024


class TreeIterator:
    """ An iterator that traverses the sub-tree of the given ModuleNode in a breadth-first manner. """

    def __init__(self, node, partition):
        self.queue = collections.deque()
        self.queue.append(node)
        self.partition = partition

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = self.queue.popleft()
            if self.partition is None or len(self._get_subtree_partitions(item)) > 1:
                self.queue.extend(item.children)
            return item
        except IndexError:
            raise StopIteration

    def _get_subtree_partitions(self, node):
        assert self.partition is not None

        subtree_partitions = {self.partition[node]}
        for c in node.children:
            subtree_partitions = subtree_partitions.union(self._get_subtree_partitions(c))
        return subtree_partitions


class ModuleNode:
    """
    A container of nn.Modules. Groups of nn.Modules that share the same nn.Parameter are grouped
    into a single ModuleNode, and ModuleNodes in turn form a tree based on the underlying parent-
    child relationships. Partitioning logic is applied on this tree, assigning sibling nn.Modules
    that are part of the same ModuleNode to the same device.
    """

    _id = 0

    def __init__(self):
        self._modules = set()
        self._children = []
        self._parent = None
        self._cost = NodeCost(0, 0, 0)
        self._normalized_cost = 0.0
        self._num_descendants = 1
        self._dummy = False

        self.id = ModuleNode._id
        ModuleNode._id += 1

        self.queue = collections.deque()

    def add_child(self, node):
        if node.parent != None:
            raise ValueError(f"Node {node.module} already has parent {node.parent.module}.")

        if node.id in [child.id for child in node.children]:
            raise ValueError(f"Child {node.module} already exists.")

        self._children.append(node)
        node.parent = self

    def add_module(self, module):
        self._modules.add(module)

    def add_modules(self, modules):
        self._modules = self._modules.union(modules)

    @property
    def modules(self):
        return self._modules

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, par):
        self._parent = par

    @property
    def num_descendants(self):
        return self._num_descendants

    @num_descendants.setter
    def num_descendants(self, desc):
        self._num_descendants = desc

    @property
    def dummy(self):
        return self._dummy

    @dummy.setter
    def dummy(self, dummy):
        self._dummy = dummy

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id_in):
        self._id = id_in

    @property
    def normalized_cost(self):
        return self._normalized_cost

    @normalized_cost.setter
    def normalized_cost(self, cost):
        self._normalized_cost = cost

    def iterate_subtree(self, partition=None):
        return TreeIterator(self, partition)

    def maybe_print_tree(self, max_level=5, level=0):
        from smdistributed.modelparallel.torch.state_mod import state

        if get_log_level() == logging.DEBUG:
            gap = " " * (2 * level)
            s = "["
            for m in self.modules:
                s += state.module_manager.get_module_name(m) + ", "
            s = s[:-2] + "]"

            logger.debug(
                gap
                + "[NODE] "
                + str(self.id)
                + ", COST: "
                + str(self.cost)
                + " "
                + str(self.normalized_cost)
                + ": "
                + s
            )

            if level < max_level:
                for c in self.children:
                    c.maybe_print_tree(max_level, level + 1)


class ModulePartitioner:
    """ An object that computes a device assignment given a model object `module`. """

    def __init__(
        self,
        module,
        num_partitions,
        trace_results=None,
        memory_weight=0.8,
        use_times=True,
        use_memory_trace=True,
    ):
        """
        Parameters:
            module           : nn.Module object to be partitioned.
            num_partitions   : Number of model partitions.
            trace_results    : A TraceResults namedtuple that contains the results of tracing including
                               module order, execution times, input and output tensor sizes
            memory_weight    : The weight assigned to the cost of memory use, vs. execution time.
                               A value of 1.0 means that we only try to balance memory, a value of
                               0.0 means that we only try to balance computation (measured by exec. times).
                               Any value in between interpolates between these two extremes.
            use_times        : Whether we should use the traced execution times in partitioning decision.
                               If set to False, the computational cost of a ModuleNode is the number of
                               its descendants. Since execution times have randomness, should be set to False
                               for deterministic results (although the degree of such randomness is small).
            use_memory_trace : Whether the traced memory usage should be used for partitioning. If False,
                               activation cost is estimated from traced module outputs, if exists.
        """

        if memory_weight < 0.0 or memory_weight > 1.0:
            raise ValueError("Memory weight must between 0.0 and 1.0.")

        self.top_module = module
        self.num_partitions = num_partitions
        self.memory_weight = memory_weight
        self.use_times = use_times
        self.use_memory_trace = use_memory_trace

        # metrics used in partitioning decision
        if trace_results:
            self.module_order = trace_results.mod_execution_order
            self.module_times = trace_results.mod_execution_times or None
            self.input_sizes = trace_results.traced_input_sizes or None  # currently unused
            self.output_sizes = trace_results.traced_output_sizes or None
            self.memory_usage = trace_results.mod_memory_usage or None
        else:
            self.module_order = []
            self.module_times = None
            self.input_sizes = None
            self.output_sizes = None
            self.memory_usage = None

        # internal state
        self.module_to_node = {}
        self.root = None
        self.current_costs = [0.0 for _ in range(self.num_partitions)]

        # internal parameters
        self.time_quantization_levels = 100
        self.epsilon = 1e-6

    def partition(self):
        """
        Entry point for partition. Creates a tree of ModuleNodes, populates the costs based on
        tracing results and tree structure, and runs the partitioning algorithm on it.
        Returns:
            partition: A dict that maps nn.Module objects to the device id they got assigned to.
        """

        # create module tree
        self.create_module_nodes()
        self.root = self.module_to_node[self.top_module]
        self.create_module_tree(self.root, set())

        self.preprocess_tracing_data()
        self.populate_cost(self.root)
        self.normalize_costs()

        self.root.maybe_print_tree()

        node_partition = self.partition_nodes(self.root)

        module_partition = {}
        for node in self.root.iterate_subtree():
            for module in node.modules:
                module_partition[module] = node_partition[node]

        return module_partition

    def display_partition(self, node_partition):
        """ Log the module assignments """
        from smdistributed.modelparallel.torch.state_mod import state

        logger.info("Partition assignments:")
        for node in self.root.iterate_subtree(node_partition):
            for module in node.modules:
                module_name = state.module_manager.get_module_name(module)
                logger.info(f"{module_name}: {node_partition[node]}")

    def preprocess_tracing_data(self):
        """ Handle missing data, normalize some data. """
        if self.module_times is not None:
            total_execution_time = self.module_times[self.top_module]
            self._prepare_execution_times(self.top_module, total_execution_time)
        else:
            self.use_times = False

        if self.memory_usage is not None:
            self._prepare_memory_usage(self.top_module)
        else:
            self.use_memory_trace = False

        if self.output_sizes is not None:
            self._prepare_tensor_sizes(self.top_module)
        else:
            self.output_sizes = collections.defaultdict(lambda: 1.0)

    def _prepare_memory_usage(self, module):
        if module not in self.memory_usage:
            self.memory_usage[module] = 0.0

        for c in module.children():
            self._prepare_memory_usage(c)

    def _prepare_execution_times(self, module, total_execution_time):
        for m in module.children():
            self._prepare_execution_times(m, total_execution_time)

        if module not in self.module_times:
            # handle nn.ModuleList and unexecuted modules
            self.module_times[module] = sum([self.module_times[c] for c in module.children()])
        else:
            # quantize to reduce the randomness in execution times
            quantization_unit = total_execution_time / self.time_quantization_levels
            self.module_times[module] = round(self.module_times[module] / quantization_unit)

    def _prepare_tensor_sizes(self, module):
        if module not in self.output_sizes:
            # this is for unused modules. it is not correct in the case nn.ModuleList,
            # but the activation costs will later be propagated from children anyway
            self.output_sizes[module] = 0.0

        if module not in self.input_sizes:
            self.input_sizes[module] = 0.0

        for c in module.children():
            self._prepare_tensor_sizes(c)

    def partition_nodes(self, root):
        """ The core of the partitioning algorithm. We traverse the ModuleNode tree in BFS manner.
        At each node, we have a subset of the available devices the current subtree needs to be
        partitioned into. The root of the current subtree always gets the first of these available
        devices. Next, each children is assigned a further subset of the available devices. If a
        child gets assigned multiple devices, BFS will later visit that child and further partition
        these devices among its own children. If a child gets assigned a single device, then that
        entire subtree is assigned to that partition. """

        to_partition = collections.deque()
        to_partition.append((root, list(range(self.num_partitions))))
        partition = {}

        while len(to_partition) > 0:
            current_node, current_partitions = to_partition.popleft()
            partition[current_node] = current_partitions[0]

            # assign children
            if len(current_partitions) > 1:
                # reorder children based on execution order
                ordered_children = self.order_nodes(current_node.children)

                if len(ordered_children) > 0:
                    child_partitions, _ = self.partition_children(
                        ordered_children,
                        current_partitions,
                        current_partitions[0],
                        self.current_costs,
                    )

                    # add nodes in the order of increasing cost, so that small children assigned to the same
                    # partition as the parent are taken into account in the later partition_children calls of
                    # larger children.
                    for ptn, child in sorted(
                        zip(child_partitions, ordered_children), key=lambda x: x[1].normalized_cost
                    ):
                        to_partition.append((child, ptn))

            else:
                self._assign_descendants_to_partition(
                    current_node, partition, current_partitions[0]
                )

            # current_costs is not currently used, but we still maintain it because it might be useful
            self._update_current_costs(
                current_node,
                current_partitions[0],
                include_descendants=(len(current_partitions) == 1),
            )

        return partition

    def _update_current_costs(self, node, partition, include_descendants):
        self.current_costs[partition] += node.normalized_cost
        if not include_descendants:
            self.current_costs[partition] -= sum([c.normalized_cost for c in node.children])

    def order_nodes(self, nodes):
        """ Order the children based on the traced module_order """

        node_set = set(nodes)
        ordered_nodes = []
        seen = set()
        for mod in self.module_order:
            node = self.module_to_node[mod]
            if node in node_set and node.id not in seen:
                ordered_nodes.append(node)
                seen.add(node.id)

        # If one of the ordered_nodes is a nn.ModuleList, it will not appear in self.module_order
        # (since it does not have a forward method), so it may not be included in ordered_children
        # at this point. We need to explicitly include it in this case so that the partitioner does
        # not miss this entire subtree.

        if len(ordered_nodes) < len(nodes):
            # this could happen because of nn.ModuleList or an uncalled nn.Module, or module_execution_order == None
            missing_nodes = node_set.difference(set(ordered_nodes))
            ordered_nodes = ordered_nodes + [n for n in nodes if n in missing_nodes]

        return ordered_nodes

    def populate_param_map(self, module, param_map):
        """
        Compute the param_map which maps a nn.Parameter to the nn.Modules that has it
        as immediate members (and not to those that have it as members of strict descendants)
        """
        from smdistributed.modelparallel.torch.state_mod import state

        if len(list(module.parameters(recurse=False))) == 0:
            # if it has no parameters, use the module itself as the key
            param_map[module].add(module)
        else:
            for param in state.module_manager.get_parameters(module, recurse=False):
                param_map[param].add(module)
        for child in module.children():
            self.populate_param_map(child, param_map)

    def create_module_nodes(self):
        """
        Group nn.Modules into a ModuleNode if they share a parameter, or if they satisfy
        the W-condition:

        Define the (bipartite) graph G which has the set of nn.Modules as its vertices, and
        there is an edge between i and j iff nn.Modules i and j share an nn.Parameter, e.g.
        module1.weight = module2.weight. Then this set of nn.Modules satisfy the W-condition
        iff graph G is connected.
        """

        param_to_module = collections.defaultdict(set)
        self.populate_param_map(self.top_module, param_to_module)

        visited = set()
        for item in param_to_module:
            modules = self._visit_modules(param_to_module, item, visited)
            if len(modules) > 0:
                node = ModuleNode()
                node.add_modules(modules)
                for mod in modules:
                    self.module_to_node[mod] = node

    def _visit_modules(self, param_to_module, item, visited):
        visited.add(item)
        found = set()
        if isinstance(item, nn.Parameter):
            for mod in param_to_module[item]:
                if mod not in visited:
                    found = found.union(self._visit_modules(param_to_module, mod, visited))
        elif isinstance(item, nn.Module):
            found.add(item)
            for param in item.parameters(recurse=False):
                if param not in visited:
                    found = found.union(self._visit_modules(param_to_module, param, visited))

        return found

    def create_module_tree(self, node, seen):
        """
        Create a tree over ModuleNodes such that ModuleNode p is parent of ModuleNode i if
        there is an nn.Module in p that is a parent of an nn.Module in i. Note that in rare
        cases such a tree is not unique, but here we return one.
        """

        for module in node.modules:
            for child in module.children():
                child_node = self.module_to_node[child]
                if child_node.id not in seen:
                    seen.add(child_node.id)
                    if node.id != child_node.id:
                        node.add_child(child_node)

        descendant_count = 1
        for child_node in node.children:
            descendant_count += self.create_module_tree(child_node, seen)

        node.num_descendants = descendant_count
        return descendant_count

    def populate_cost(self, node):
        """
        The cost of a ModuleNode is
           Cost(node) = self.memory_weight * MemoryCost(node)
                            + (1 - self.memory_weight) * ComputationCost(node),

        MemoryCost consists of ActivationCost and ParameterCost, where
        ActivationCost is the normalized sum of module output sizes in a given
        subtree, and ParameterCost is the normalized sum of sizes of nn.Parameters
        in the subtree. If execution times are available, ComputationCost is the
        traced execution time for the node. Otherwise it is the total number of
        modules in the subtree.

        This method must assign a non-zero cost to every node to guarantee the
        termination of the partitioning algorithm. Currently this is ensured by
        thresholding the costs to be larger than self.epsilon.
        """
        if self.use_times:
            computation_cost = sum([self.module_times[m] for m in node.modules])
        else:
            computation_cost = len(node.modules)

        if self.use_memory_trace:
            activation_cost = sum([self.memory_usage[m] for m in node.modules]) / MB
        else:
            activation_cost = (
                sum([self.output_sizes[m] for m in node.modules]) if len(node.children) == 0 else 0
            ) / MB

        parameter_cost = self.get_parameter_cost(node) / MB

        for child in node.children:
            self.populate_cost(child)
            parameter_cost += child.cost.parameters
            if not self.use_memory_trace:
                activation_cost += child.cost.activations
            if not self.use_times:
                computation_cost += child.cost.computation

        # we need this for both computation_cost and activation_cost to support all values of self.memory_weight
        computation_cost = max(self.epsilon, computation_cost)
        activation_cost = max(
            self.epsilon, sum([c.cost.activations for c in node.children]), activation_cost
        )

        node.cost = NodeCost(computation_cost, parameter_cost, activation_cost)

    def get_parameter_cost(self, node):
        cost = 0
        seen = set()
        for mod in node.modules:
            for param in mod.parameters(recurse=False):
                if id(param) not in seen:
                    cost += product(param.shape) * dtype_size(param.dtype)
                    seen.add(id(param))
        return cost

    def normalize_costs(self):
        """ Normalize the costs such that the total cost across the ModuleNodes sum to 1. """

        # because of optimizer state.
        # TODO: more accurately estimate this from the optimizer.
        PARAMETER_COST_MULTIPLIER = 3

        root_mem_cost = (
            self.root.cost.parameters * PARAMETER_COST_MULTIPLIER + self.root.cost.activations
        )

        for node in self.root.iterate_subtree():
            comp = node.cost.computation / self.root.cost.computation
            mem = (
                (node.cost.parameters * PARAMETER_COST_MULTIPLIER + node.cost.activations)
                / root_mem_cost
                if root_mem_cost > 0
                else 0
            )

            # ideally parameter and activation costs should contribute in the exact proportions of their
            # memory usage, but this can become complicated quickly: true parameter cost depends on the
            # optimizer used; activation cost of functionals are not captured, etc. so we simply assume
            # that parameter cost and activation cost contribute equally
            node.normalized_cost = (1 - self.memory_weight) * comp + self.memory_weight * mem

    def _assign_descendants_to_partition(self, node, partition, i):
        """ Like father, like son. """

        partition[node] = i
        for child in node.children:
            self._assign_descendants_to_partition(child, partition, i)

    def _get_segment_splittable_counts(self, segmented_children):
        """ Get the maximum number of partitions each segment can be assigned. """

        segment_splittable_counts = []
        for segment in segmented_children:
            count = 0
            for node in segment:
                if node.dummy:
                    count += 1
                else:
                    # if the number of descendants (including the node itself) is 1 or 2,
                    # it can only take 1 partition. otherwise it can take (num_descendants - 1).
                    # the former is because in the single-child + 2-partition case, the algorithm
                    # will end up allocating everything to the parent's partition anyway.
                    count += max(0, node.num_descendants - 2) + 1
            segment_splittable_counts.append(count)
        return segment_splittable_counts

    def partition_children(self, node_list, partitions, parent_partition, current_costs):
        """ Split the group of available devices among the children nodes. To do this,
        we alternate between DP-based partitioning logic to split the children into
        segments that are as balanced as possible, followed by a re-allocation of devices
        into the segments using d'Hondt method. This is because the sizes of the children
        may be imbalanced, in which case it makes more sense to assign multiple devices
        for that node (which will be divided among *its own* children). For segments
        that get assigned multiple devices, we recurse.

        Example input:
            cost_list = [0.03, 0.05, 0.07, 0.4, 0.08, 0.05, 0.17, 0.04, 0.04, 0.04, 0.08]
            num_partitions = 8

            possible output = [[1], [1], [1], [0, 2, 3], [4], [4], [5, 6], [7], [7], [7], [7]]

        Note that the algorithm allocates multiple devices to large nodes, and tries to keep
        assignments in contiguous blocks as much as possible. One additional benefit is that
        small nodes tend to get placed in the parent partition, which reduces unnecessary
        communication with children.

        Also returns the set `bound_to_parent`, representing the set of segment indices that
        are too small to be assigned an individual device, and are implicitly assigned to the
        same device/partition as the parent node.
        """

        current_partition_costs = [current_costs[p] for p in partitions]
        existing_parent_cost = current_partition_costs[0]
        parent_partition_cost = existing_parent_cost

        partitions_wrt_dummy_locations = []
        bound_to_parent_wrt_dummy_locations = []

        for dummy_loc in range(len(node_list) + 1):
            child_partitions = [[] for _ in node_list]

            # insert dummy node in node_list
            dummy_inserted_node_list = self.insert_dummy_node(
                node_list, existing_parent_cost, dummy_loc
            )

            dummy_inserted_segmented_children, dummy_inserted_segment_costs = self.get_segments(
                dummy_inserted_node_list, len(partitions)
            )

            parent_segment = self._find_parent_segment(dummy_inserted_segmented_children, dummy_loc)

            dummy_inserted_splittable_counts = self._get_segment_splittable_counts(
                dummy_inserted_segmented_children
            )

            dummy_inserted_segment_partitions = self.dhondt_allocate(
                dummy_inserted_segment_costs,
                partitions,
                dummy_inserted_splittable_counts,
                parent_segment,
            )
            segmented_children, segment_costs, segment_partitions, splittable_counts = self.remove_dummy_node(
                dummy_inserted_segmented_children,
                dummy_inserted_segment_costs,
                dummy_inserted_segment_partitions,
                dummy_inserted_splittable_counts,
                dummy_loc,
                existing_parent_cost,
            )

            child_idx = 0
            bound_to_parent = set()

            for i, (segment, seg_ptn) in enumerate(zip(segmented_children, segment_partitions)):
                if len(segment) == 0:
                    continue

                # empty segments are implicitly placed on the partition of the parent
                if len(seg_ptn) == 0:
                    for idx in range(child_idx, child_idx + len(segment)):
                        child_partitions[idx] = [parent_partition]
                        parent_partition_cost += segment[idx - child_idx].normalized_cost
                        bound_to_parent.add(idx)

                elif len(segment) == 1:
                    child_partitions[child_idx] = seg_ptn

                elif len(seg_ptn) == 1:
                    for idx in range(child_idx, child_idx + len(segment)):
                        child_partitions[idx] = seg_ptn
                else:
                    if (segment, seg_ptn) == (node_list, partitions):
                        # skip self-similar allocations to prevent infinite recursion. there must exist
                        # at least one dummy_loc for which there are no self-similar allocations.
                        # (when dummy_loc separates the segment. note len(segment) > 1 here.)
                        child_partitions = None
                        break

                    # if multiple partitions are assigned to a segment of size larger than 1,
                    # we need to do another partition/allocate cycle for this segment. this
                    # recursion is guaranteed to terminate so long as the DP step achieves the
                    # minimum of its optimization problem, and all nodes have strictly positive cost.

                    segment_results, seg_bound_to_parent = self.partition_children(
                        segment, seg_ptn, seg_ptn[0], current_costs
                    )
                    for idx in range(child_idx, child_idx + len(segment)):
                        child_partitions[idx] = segment_results[idx - child_idx]
                    for idx in seg_bound_to_parent:
                        bound_to_parent.add(child_idx + idx)

                child_idx += len(segment)

            partitions_wrt_dummy_locations.append(child_partitions)
            bound_to_parent_wrt_dummy_locations.append(bound_to_parent)

        # find the parent assignment choice that results in minimum max cost across children
        minimax_cost = float("inf")
        minimax_cost_loc = -1
        for dummy_loc, loc_partitions in enumerate(partitions_wrt_dummy_locations):
            cur_max_cost = self._compute_max_cost(
                loc_partitions, node_list, partitions, existing_parent_cost
            )

            if cur_max_cost < minimax_cost:
                minimax_cost = cur_max_cost
                minimax_cost_loc = dummy_loc

        return (
            partitions_wrt_dummy_locations[minimax_cost_loc],
            bound_to_parent_wrt_dummy_locations[minimax_cost_loc],
        )

    def _compute_max_cost(self, child_partitions, node_list, partitions, existing_parent_cost):
        if child_partitions is None:
            return float("inf")

        costs = {p: 0.0 for p in partitions}
        costs[partitions[0]] = existing_parent_cost
        for node, ptn in zip(node_list, child_partitions):
            for p in ptn:
                costs[p] += node.normalized_cost / len(ptn)
        return max([v for k, v in costs.items()])

    def _find_parent_segment(self, dummy_inserted_segmented_children, dummy_loc):
        node_ptr = 0
        seg_ptr = 0
        while node_ptr + len(dummy_inserted_segmented_children[seg_ptr]) <= dummy_loc:
            node_ptr += len(dummy_inserted_segmented_children[seg_ptr])
            seg_ptr += 1
        return seg_ptr

    def remove_dummy_node(
        self,
        dummy_inserted_segmented_children,
        dummy_inserted_segment_costs,
        dummy_inserted_segment_partitions,
        dummy_inserted_splittable_counts,
        dummy_loc,
        parent_partition_cost,
    ):
        """ Remove the inserted dummy node from the segment it was assigned to. """

        segmented_children = copy.copy(dummy_inserted_segmented_children)
        segment_costs = copy.copy(dummy_inserted_segment_costs)
        splittable_counts = copy.copy(dummy_inserted_splittable_counts)
        segment_partitions = copy.copy(dummy_inserted_segment_partitions)

        node_ptr = 0
        seg_ptr = 0
        while node_ptr + len(dummy_inserted_segmented_children[seg_ptr]) <= dummy_loc:
            node_ptr += len(dummy_inserted_segmented_children[seg_ptr])
            seg_ptr += 1

        segmented_children[seg_ptr].pop(dummy_loc - node_ptr)
        segment_costs[seg_ptr] -= parent_partition_cost
        splittable_counts[seg_ptr] -= 1

        if splittable_counts[seg_ptr] < len(segment_partitions[seg_ptr]):
            # this has to be true since the removed dummy will have splittable count = 1,
            # and splittable counts are additive
            assert splittable_counts[seg_ptr] == len(segment_partitions[seg_ptr]) - 1

            # remove the parent since the only reason it was assigned here was the dummy node
            segment_partitions[seg_ptr].pop(0)

        return segmented_children, segment_costs, segment_partitions, splittable_counts

    def insert_dummy_node(self, node_list, parent_partition_cost, dummy_loc):
        """ Insert a dummy node that represents the existing parent cost into the given location"""

        dummy_node = ModuleNode()
        dummy_node.dummy = True
        dummy_node.normalized_cost = parent_partition_cost
        return node_list[:dummy_loc] + [dummy_node] + node_list[dummy_loc:]

    def dhondt_allocate(self, cost_list, partitions, splittable_counts, parent_segment):
        """ Use d'Hondt method to allocate devices into a set of segments. d'Hondt method
        (normally used in elections to proportionally allocate parliament seats based on
        vote shares in a district) might assign multiple devices to disproportionately
        large segments. The method is known to slightly favor the parties with larger
        vote shares, which works nicely here because it will tend to leave small segments
        unallocated, which implicitly places them on the parent device, which cuts down
        on communication. https://en.wikipedia.org/wiki/D%27Hondt_method """

        if len(partitions) < len(cost_list):
            raise ValueError("Cost list cannot be larger than the number of partitions.")

        allocated_partitions = [[] for _ in range(len(cost_list))]
        current_allocations = [0 for _ in range(len(cost_list))]

        for p in partitions:
            current_costs = [cost / (q + 1) for cost, q in zip(cost_list, current_allocations)]
            decreasing_costs = sorted(
                [(i, c) for i, c in enumerate(current_costs)], key=lambda x: x[1], reverse=True
            )

            # find the highest-cost segment that did not reach its maximum splittable count
            for ind, cost in decreasing_costs:
                if current_allocations[ind] < splittable_counts[ind]:
                    allocated_partitions[ind].append(p)
                    current_allocations[ind] += 1
                    break
            else:
                raise ValueError(
                    f"Model is too small to split into {self.num_partitions} partitions!"
                )

        # if the parent is not assigned to its pre-determined segment, swap it
        if (
            len(allocated_partitions[parent_segment]) > 0
            and partitions[0] not in allocated_partitions[parent_segment]
        ):
            # find where parent partition is assigned to
            for seg_ind, seg in enumerate(allocated_partitions):
                if partitions[0] in seg:
                    swap_seg = seg_ind
                    swap_ind = seg.index(partitions[0])
                    break

            allocated_partitions[parent_segment][0], allocated_partitions[swap_seg][swap_ind] = (
                allocated_partitions[swap_seg][swap_ind],
                allocated_partitions[parent_segment][0],
            )

        return allocated_partitions

    def get_segments(self, node_list, num_partitions):
        """
        Use dynamic programming to solve

            minimize_{P} max_{i} Cost_P(i),

        where P is a partition function, i is the partition index, and Cost_P(i)
        represents the total cost of the ModuleNodes in partition i under the
        partition decision P.

        Breaks the list of ModuleNodes into num_partitions partitions using
        dynamic programming to minimize the maximum partition cost, where partition
        cost is the sum of ModuleNode costs in that partition.

        cost[k][i] = cost of partitioning node_list[:i] into k partitions

        The core recursion is the following:
            cost[k][i] = minimize_{ j \leq i} max {cost[k-1][j], total cost of node_list[j:]}

        Here j represents the index where the last partition begins. Assuming an optimal (k-1)-way
        partition is given for the sub-array node_list[:j], then the maximum partition cost
        for a given choice of j is max{cost[k-1][j], total cost of node_list[j:]}. We are
        minimizing this over the choice of j.

        Example:
            cost_list = [0.1, 0.15, 0.05, 0.15, 0.25, 0.2, 0.1], num_partitions = 2

            The partition [0.1, 0.15, 0.05, 0.15], [0.25, 0.2, 0.1] has the cost
            max(0.45, 0.55) = 0.55, and this is the minimum.
        """
        if len(node_list) == 0:
            return [[]], [0]

        cost_list = [node.normalized_cost for node in node_list]

        # prepend empty list to make indexing consistent with definition
        cost = [[], [sum(cost_list[:i]) for i in range(len(cost_list))]]
        separators = [[], [[] for i in range(len(cost_list))]]

        for k in range(2, num_partitions + 1):
            cost.append([])
            separators.append([])
            for i in range(len(node_list) + 1):
                min_cost = float("inf")
                min_separator = -1
                for j in range(i):
                    cur_cost = max(sum(cost_list[j:i]), cost[k - 1][j])
                    if cur_cost < min_cost:
                        min_cost = cur_cost
                        min_separator = j
                cost[k].append(min_cost)
                separators[k].append(separators[k - 1][min_separator] + [min_separator])

        splits = separators[-1][-1]

        # assign current and child modules
        split_nodes = []
        costs = []
        for i in range(len(splits)):
            if i == 0:
                nodes = node_list[: splits[i]]
            else:
                nodes = node_list[splits[i - 1] : splits[i]]
            split_nodes.append(nodes)
            costs.append(sum([n.normalized_cost for n in nodes]))
        split_nodes.append(node_list[splits[-1] :])
        costs.append(sum([n.normalized_cost for n in split_nodes[-1]]))

        return split_nodes, costs
