# Standard Library
import collections
import copy
import pickle
import re
from enum import Enum

# Third Party
import numpy as np
import tensorflow as tf

# First Party
import smdistributed.modelparallel.tensorflow.core_mod as core
from smdistributed.modelparallel.backend.utils import upload_metrics_to_studio
from smdistributed.modelparallel.tensorflow import graph_def_editor as gde
from smdistributed.modelparallel.tensorflow import state
from smdistributed.modelparallel.tensorflow.fusing import FusedGraph
from smdistributed.modelparallel.tensorflow.graph_def_editor.tensor import Tensor as gde_Tensor
from smdistributed.modelparallel.tensorflow.graph_def_editor.util import make_placeholder
from smdistributed.modelparallel.tensorflow.graph_utils import (
    IO,
    _get_op_name,
    get_item_in_graph,
    get_op,
    get_tensor,
    is_placeholder,
    is_var,
    make_placeholder_graph_def,
    make_tensor_shape,
    node_consumers,
    update_tensor_consumers,
)
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.ops import ALLGATHER_OP, BUNDLE_OP, INPUT_OP, OUTPUT_OP
from smdistributed.modelparallel.tensorflow.pybind import metis_partition
from smdistributed.modelparallel.tensorflow.serialization import _TF_OBJECT_PREFIX
from smdistributed.modelparallel.tensorflow.utils import (
    dtype_to_bytes,
    get_dummy_spec,
    mainify_name,
    set_to_sorted_list,
)

from smdistributed.modelparallel.tensorflow.attrs import *  # noqa isort:skip

# As we manipulate the gde.Graph, tensor names change, so we cannot keep track
# of tensors by their name. Instead, we represent them in reference to a neighboring
# node. SubgraphTensor('add', 1, IO.INPUT) would mean that we are referring to the
# tensor that is the input number 1 (second input) of the node 'add'.
SubgraphTensor = collections.namedtuple("SubgraphTensor", "node index io")


class PartitionType(Enum):
    UNIT_TEST = 0
    METIS = 1


def is_model_input(node):
    return is_placeholder(node) and node.outputs[0].name in state.serialized_graph.model_inputs


class AutoPartitioner:
    """ Object that takes a SerializedGraph, adds new attributes to it representing the
    partitioned graph in importable form."""

    def __init__(self):
        self.op_id_to_device = {}
        self.op_counter = 0
        self.metrics = {}

    def get_op_id(self, *args):
        """ Generate a unique op_id."""
        op_id = state.op_id_gen.get_op_id(*args)
        return op_id

    def _check_if_allgather_node(self, node):
        if node.op_type == ALLGATHER_OP:
            raise RuntimeError(
                "Allgather op should not be put under the smp.DistributedModel context!"
            )

    def partition(self):
        """
        Entry point for the partitioning logic. `state.op_to_device` represents the initial, user-specified
        device assignments in the form of a dict mapping op name to device id. This, if non-empty, will be
        used as a starting point for the partitioning, and only the unassigned ops will be assigned to a device."""

        self.op_id_to_device = {}

        # create a gde.Graph object that is more amenable to manipulation than raw tf.GraphDef
        g = gde.Graph(state.serialized_graph.graph_def)

        # mainify names and check invalid nodes
        for node in g.nodes:
            self._check_if_allgather_node(node)
            main_name = mainify_name(node.name)
            if main_name != node.name:
                g.rename_node(node.name, main_name)
            # 'mainify' the names of the colocation constraints
            mainified_coloc_node_names = {mainify_name(name) for name in node.colocation_groups}
            if mainified_coloc_node_names:
                node.colocation_groups = mainified_coloc_node_names

        # map each op name to a device id, and return a ModelPartition object
        model_partition = self.decide_partition(g, state.serialized_graph.op_to_device)

        partitioned_graphs = [
            [None for mb in range(state.num_microbatches())]
            for dev in range(state.cfg.pipeline_parallel_degree)
        ]

        # given ModelPartition, produce the importable GraphDef for each device and microbatch, by inserting
        # SMP ops, setting the attributes corresponding to the microbatch, and removing unassigned variables
        # and operations.
        partitioned_graphs = self.implement_partition(g, model_partition)

        # dump the results
        state.serialized_graph.partitioned_graphs = partitioned_graphs
        state.serialized_graph.op_id_to_device = self.op_id_to_device
        state.serialized_graph.save()

    def decide_partition(self, graph, op_to_device):
        """
        Given a gde.Graph object, output a ModelPartition object representing the partitioning decision.
        In TF2.x, must also partition the layers into `serialized_graph.partitioned_layers`,
        which will be used to build only the layers assigned to the current device while importing.

        The logic is using an existing library named Metis to do the partition.
        For details of Metis, please refer to: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
        """
        # (TODO) fuse the variables and user defined ops to super nodes

        # remove placeholders from op_to_device
        filtered_op_to_device = {}
        for op in op_to_device:
            if graph.contains_node(_get_op_name(op)):
                if not is_model_input(graph.get_node_by_name(_get_op_name(op))):
                    filtered_op_to_device[op] = op_to_device[op]
            else:
                get_logger().info(
                    f"Op {op} not present in graph. This is normal if re-partitioning."
                )

        op_sets = self._init_op_assignments(filtered_op_to_device)

        if state.partition_type == PartitionType.UNIT_TEST:
            stage = 0
            for node in graph.nodes:
                if is_model_input(node):
                    continue

                if all([(node.name not in op_set) for op_set in op_sets]):
                    if "first" in node.name:
                        op_sets[0].add(node.name)
                        if stage == 0:
                            stage = 1
                    elif "second" in node.name or (
                        "conv2d_1" in node.name and state.tf_version == 1
                    ):
                        op_sets[1].add(node.name)
                        if stage == 1:
                            stage = 2
                    elif node.name == "Identity" or node.name == "Identity_1":
                        op_sets[1].add(node.name)
                    elif stage < 2:
                        op_sets[0].add(node.name)
                    else:
                        op_sets[1].add(node.name)

            self._maybe_add_layers_dummy()
        else:
            # Create the fused graph based on the gde graph
            if state.tf_version == 2:
                op_to_layer = state.serialized_graph.op_to_layer
            else:
                op_to_layer = self._group_by_outermost_name_scope(state.serialized_graph.graph_def)

            if state.cfg.auto_partition == True:
                fused_g = FusedGraph(graph, op_to_layer)

                if state.cfg.load_partition:
                    with open(state.partition_file, "rb") as filehandle:
                        partition = pickle.load(filehandle)
                else:
                    var_shapes = (
                        state.serialized_graph.var_shapes if state.tf_version == 2 else None
                    )

                    if state.cfg.contiguous:
                        # if contiguous partition is requested, apply METIS contiguous partition
                        # to the largest connected component in the fused graph, and iteratively
                        # add the remaining connected components to the current smallest partition
                        # in terms of variable storage load
                        # the underlying assumption here is that the largest connected component
                        # will contain most of the nodes in the graph, and the rest are small components
                        components = fused_g.get_connected_components()
                        component_sizes = [len(comp) for comp in components]
                        get_logger().debug(f"component sizes {component_sizes}")
                        ind = component_sizes.index(max(component_sizes))

                        largest_component = sorted(components[ind])
                        other_components = [comp for i, comp in enumerate(components) if i != ind]
                    else:
                        # if contiguous partition is not requested, apply METIS partition to the entire
                        # graph at once
                        largest_component = fused_g.fused_node_names
                        other_components = []

                    num_vtxs, num_edges, adj_map, edge_map, edge_weight_map, vtx_weight, num_cons = fused_g.convert_to_metis_graph(
                        largest_component, var_shapes
                    )
                    # Run metis to get partition results
                    # partition is a list where the index for the fused node and value is the device that node belongs to
                    if state.cfg.pipeline_parallel_degree > 1:
                        partition, edge_cut = metis_partition(
                            state.cfg.pipeline_parallel_degree,
                            num_vtxs,
                            num_edges,
                            num_cons,
                            adj_map,
                            edge_map,
                            edge_weight_map,
                            vtx_weight,
                            state.cfg.contiguous,
                        )
                    else:
                        partition = [0 for _ in range(num_vtxs)]
                    get_logger().debug(f"Partition {partition}")
                    # save the partition result
                    if core.local_rank() == 0:
                        with open(state.partition_file, "wb") as filehandle:
                            pickle.dump(partition, filehandle)
            else:
                # manual partition
                layer_to_device = self.get_layer_to_device(
                    op_to_layer, filtered_op_to_device, state.cfg.default_partition
                )
                fused_g = FusedGraph(graph, op_to_layer, layer_to_device)
                largest_component = fused_g.fused_node_names
                other_components = []
                partition = []
                for component in largest_component:
                    if component in fused_g.fused_node_name_to_device:
                        device_name = fused_g.fused_node_name_to_device[component]
                        partition.append(device_name)
                    else:
                        partition.append(state.cfg.default_partition)
            # Assign gde nodes to its corresponding device
            self.assign_node_to_device(
                largest_component, other_components, partition, fused_g, op_sets
            )
            if core.rank() == 0:
                # Upload metrics to SM studio
                upload_metrics_to_studio(self.metrics)

        self._update_var_to_device(op_sets)

        asgs = [
            AutoSubgraph(idx, graph, op_sets[idx])
            for idx in range(state.cfg.pipeline_parallel_degree)
        ]
        return ModelPartition(graph, asgs)

    def _get_total_communication_volume(self, partition, fused_graph, fused_node_indices):
        """"Collect the total tensor sizes communicated between devices"""
        total_communication_volume = 0
        for node in fused_graph.fused_nodes:
            current_dev = partition[fused_node_indices[node.name]]
            for output_node_name in node.outputs:
                output_dev = partition[fused_node_indices[output_node_name]]
                if current_dev != output_dev:
                    output_node = fused_graph.get_fused_node_by_name(output_node_name)
                    total_communication_volume += fused_graph.get_communication_volume_between_nodes(
                        node, output_node
                    )
        # There are cases that we are unable to fetch the tensor shapes
        # If the total_communication_volume is too small then we don't upload
        if total_communication_volume > 1:
            self.metrics["estimated_communication_volume(MB)"] = round(
                total_communication_volume, 2
            )

    def get_layer_to_device(self, op_to_layer, op_to_device, default_partition):
        """ Generate layer to device dict using common ops between op_to_layer and op_to_device.
        For any op specified in op_to_device, every op in that op's layer will go to the same device.
       """

        layer_to_device = {}

        for op, layer in op_to_layer.items():
            if op in op_to_device:
                device_name = op_to_device[op]
            else:
                device_name = default_partition
            layer_to_device[layer] = device_name

        return layer_to_device

    def collect_node_layer_fraction(self, op_sets):
        def _collect_fraction(op_or_layer_sets, name):
            total_cnt = 0
            for dev in range(state.cfg.pipeline_parallel_degree):
                total_cnt += len(op_or_layer_sets[dev])
            for dev in range(state.cfg.pipeline_parallel_degree):
                self.metrics[f"{name}_{dev}"] = len(op_or_layer_sets[dev]) / total_cnt

        if hasattr(state.serialized_graph, "layers"):
            _collect_fraction(
                state.serialized_graph.partitioned_layers, "fraction_of_layers_on_dev"
            )
        _collect_fraction(op_sets, "fraction_of_ops_on_dev")

    def assign_node_to_device(
        self, largest_component, other_components, partition, fused_graph, op_sets
    ):
        """Given the partition result, assign each gde node into the device it belongs to.
           op_sets is a python dictionary record which maps device to a set of gde node names"""
        # var_size is the variable size we used for partition, could be the variable count if state.cfg.optimize == "speed"
        var_size = [0 for _ in range(state.cfg.pipeline_parallel_degree)]
        real_var_size = [0 for _ in range(state.cfg.pipeline_parallel_degree)]
        fused_node_list = fused_graph.fused_node_names

        full_partition = [-1 for node in fused_node_list]
        fused_node_indices = {name: i for i, name in enumerate(fused_node_list)}

        var_shapes = state.serialized_graph.var_shapes if state.tf_version == 2 else None

        def _place_fused_node_in_op_set(fused_node_name, dev):
            fused_node = fused_graph.get_fused_node_by_name(fused_node_name)
            full_partition[fused_node_indices[fused_node_name]] = dev
            var_size[dev] += fused_node.get_var_size(real_size=(state.cfg.optimize == "memory"))
            real_var_size[dev] += fused_node.get_var_size(var_shapes=var_shapes, real_size=True)
            for gde_node_name in fused_node.nodes:
                if is_model_input(fused_graph._gde_graph.get_node_by_name(gde_node_name)):
                    continue
                if all([(gde_node_name not in op_set) for op_set in op_sets]):
                    op_sets[dev].add(gde_node_name)

        op_to_device = state.serialized_graph.op_to_device

        # For repartition keep track of fused node names corresponding to already assigned variables
        fused_node_to_device = {
            self._get_op_to_fused_node_name(fused_graph, op): device
            for op, device in op_to_device.items()
        }

        # assign nodes in the largest component
        for idx, device in enumerate(partition):
            if core.rank() == 0:
                get_logger().debug(f"Placed fused node {largest_component[idx]} on device {device}")
            if largest_component[idx] not in fused_node_to_device:
                _place_fused_node_in_op_set(largest_component[idx], device)
            else:
                _place_fused_node_in_op_set(
                    largest_component[idx], fused_node_to_device[largest_component[idx]]
                )

        # iteratively assign the rest of the components
        for other_component in other_components:
            smallest_partition = var_size.index(min(var_size))
            for fused_node_name in other_component:
                if fused_node_name not in fused_node_to_device:
                    _place_fused_node_in_op_set(fused_node_name, smallest_partition)
                else:
                    _place_fused_node_in_op_set(
                        fused_node_name, fused_node_to_device[fused_node_name]
                    )

        # full_partition represents the assignment of all fused nodes - not just the largest component
        self._maybe_add_layers(full_partition, fused_graph, fused_node_indices)

        self._get_total_communication_volume(full_partition, fused_graph, fused_node_indices)

        if core.rank() == 0:
            self.collect_node_layer_fraction(op_sets)

        if core.rank() == 0:
            for i in range(state.cfg.pipeline_parallel_degree):
                # Take variable as fp32 dtype
                self.metrics[f"variable_size(GB)_on_dev_{i}"] = (
                    real_var_size[i] * dtype_to_bytes(tf.float32) // 1e9
                )
                get_logger().info(f"Total variables in device {i}: {var_size[i]}")

            for i in range(state.cfg.pipeline_parallel_degree):
                self.metrics[f"total_ops_in_device_{i}"] = len(op_sets[i])
                get_logger().info(f"Total ops in device {i}: {len(op_sets[i])}")

    def _get_identity_op_to_layer(self):
        op_to_layer = {}
        for node in state.serialized_graph.graph_def.node:
            op_to_layer[node.name] = node.name
        return op_to_layer

    def _get_op_to_fused_node_name(self, fused_graph, op):
        op_name = _get_op_name(op)
        return fused_graph.get_fused_node_name_by_gde_node_name(op_name)

    def _group_by_outermost_name_scope(self, graph_def):
        """
        Group ops by their outermost name scope, for pre-partitioning node fusion.
        It groups the ops into the outermost name scope that is itself an op, e.g.
        for the following ops:
            foo/bar/baz
            foo/bar
            foo/bar/baz/foo
            foo/foo/bar
            foo/foo/baz
        It will create three groups:
            [foo/bar/baz, foo/bar, foo/bar/baz/foo], [foo/foo/bar], [foo/foo/baz]
        since the first 3 share the name scope foo/bar, and foo/bar is itself an op.
        We are imposing the constraint that the outer name scope must itself be an op
        to prevent cases where a user-defined, large name scope results in fusing large
        parts of the graph into a single node.

        This can probably be improved in the future, but for now it performs reasonably well.
        """
        node_names = {n.name for n in graph_def.node}

        op_to_group = {}
        for node in graph_def.node:
            ind = 0
            while True:
                ind = node.name.find("/", ind + 1)
                if ind == -1:
                    op_to_group[node.name] = node.name
                    break
                elif node.name[:ind] in node_names:
                    op_to_group[node.name] = node.name[:ind]
                    break

        return op_to_group

    def _maybe_add_layers_dummy(self):
        if hasattr(state.serialized_graph, "layers") and len(state.serialized_graph.layers) > 0:
            state.serialized_graph.partitioned_layers[0] = {"conv2d", "flatten"}
            state.serialized_graph.partitioned_layers[1] = {"dense", "dense_1"}

    def _maybe_add_layers(self, partition, fused_graph, fused_node_indices):
        """Assign layers to the correct device based on partition results for fused nodes"""
        if hasattr(state.serialized_graph, "layers"):
            fused_node_names = fused_graph.fused_node_names
            for name in state.serialized_graph.layers:
                device = None
                try:
                    fused_node_name = fused_graph.layer_name_to_fused_node_name(name)
                except KeyError:
                    if name not in set(state.serialized_graph.op_to_layer.values()):
                        # if we cannot find a FusedNode because there is no op in this
                        # layer (excluding the ones in its sub-layers), we can assign
                        # it to any device because it won't matter for actual execution.
                        device = 0
                    else:
                        # if the layer name is in `op_to_layer`, but we are still getting
                        # a KeyError, it is a legitimate error
                        raise

                if device is None:
                    device = partition[fused_node_indices[fused_node_name]]

                state.serialized_graph.partitioned_layers[device].add(name)

    def _update_var_to_device(self, op_sets):
        """
        Based on the partitioning decision, update `var_to_device` mapping
        for newly created variables.
        Note that this mapping will remain unmodified for existing items, since once
        a variable is initialized on a device, it cannot be re-assigned.
        """
        for var_name, ts_name in state.serialized_graph.var_to_op.items():
            op_name = _get_op_name(ts_name)
            for dev, op_set in enumerate(op_sets):
                if op_name in op_set:
                    if var_name not in state.serialized_graph.var_to_device:
                        state.serialized_graph.var_to_device[var_name] = dev
                    else:
                        assert dev == state.serialized_graph.var_to_device[var_name]
                    break
            else:
                raise ValueError(f"Cannot find {op_name} in any op sets!")

    def _init_op_assignments(self, op_to_device):
        """ Seed the op_sets with  op_to_device, which includes user-specified
        assignments, in addition to the existing variable assignments."""
        op_sets = [set() for _ in range(state.cfg.pipeline_parallel_degree)]

        for op_name, dev in op_to_device.items():
            op_sets[dev].add(_get_op_name(op_name))

        return op_sets

    def implement_partition(self, g, model_partition):
        """ Given the ModelPartition, produce the graph to be imported for device `dev` and microbatch `mb`"""
        partitioned_graphs = [
            [None for mb in range(state.num_microbatches())]
            for dev in range(state.cfg.pipeline_parallel_degree)
        ]

        g, smp_ops, model_output_mapping = self._insert_smp_ops(g, model_partition)

        for dev in range(state.cfg.pipeline_parallel_degree):
            # make a copy
            dev_g = gde.Graph(g.to_graph_def())

            self.set_attributes_for_device(
                dev_g, dev, smp_ops, {v.name for v in model_output_mapping.values()}
            )
            graph_def = self.remove_unassigned_asgs(dev_g, model_partition, dev)

            for mb in range(state.num_microbatches()):
                dev_mb_graph_def = copy.deepcopy(graph_def)
                if mb == 0:
                    smp_op_indices = self._find_smp_op_indices(dev_mb_graph_def)
                partitioned_graphs[dev][mb] = self.set_attributes_for_mb(
                    dev_mb_graph_def, dev, mb, smp_ops, smp_op_indices
                )
                self.map_model_outputs(model_output_mapping, dev, mb)

        return partitioned_graphs

    def _find_smp_op_indices(self, graph_def):
        """ Find the NodeDef indices belonging to SMP ops """
        ind = {}
        for i, node in enumerate(graph_def.node):
            if OP_ID_ATTR in node.attr:
                ind[node.name] = i
        return ind

    def map_model_outputs(self, model_output_mapping, dev, mb):
        """ Replace Model Output names with their attached SmpOutput counterparts """
        for tensor_name, smp_output_op in model_output_mapping.items():
            ind = state.serialized_graph.model_outputs[dev][mb].index(tensor_name)
            smp_output_name = smp_output_op.outputs[0].name
            state.serialized_graph.model_outputs[dev][mb][ind] = smp_output_name

            state.serialized_graph.structured_outputs[dev][mb] = tf.nest.map_structure(
                lambda x: self.replace_tensor_name(x, tensor_name, smp_output_name),
                state.serialized_graph.structured_outputs[dev][mb],
            )

    def set_attributes_for_device(self, graph, dev, smp_ops, model_outputs):
        """ Set device-specific attributes for SMP ops """
        for asg_id, asg_ops in smp_ops.items():
            for op in asg_ops:
                if op.op_type == INPUT_OP and asg_id == dev:
                    continue
                if op.op_type == OUTPUT_OP and op.name in model_outputs:
                    continue

                node = graph.get_node_by_name(op.name)
                shape, dtype = get_dummy_spec()

                node.replace_attr(EXPECTED_SHAPE_ATTR, shape)
                node.replace_attr(OUT_TYPE_ATTR, dtype)

                node.set_outputs_from_pairs([(dtype, make_tensor_shape(shape))])

        # assign correct input dtype for smp ops (since some tensors became dummy their dtype has changed)
        for asg_id, asg_ops in smp_ops.items():
            for op in asg_ops:
                node = graph.get_node_by_name(op.name)
                dtype = node.inputs[0].dtype
                node.replace_attr(T_ATTR, dtype)

    def set_attributes_for_mb(self, graph_def, dev, mb, smp_ops, smp_op_indices):
        """ Set microbatch-specific attributes for SMP ops """
        for asg_id, asg_ops in smp_ops.items():
            for op in asg_ops:
                op_ctr = int(re.match(".*?([0-9]+)$", op.name).group(1))
                op_id = self.get_op_id(mb, asg_id, op_ctr)
                node_def = graph_def.node[smp_op_indices[op.name]]
                node_def.attr[MICROBATCH_ATTR].i = mb
                node_def.attr[OP_ID_ATTR].i = op_id

                state.serialized_graph.smp_ops[op_id] = op.name
                self.op_id_to_device[op_id] = asg_id

        return graph_def

        self.set_correct_input_output_dtype(dev_g)

    def group_by_origin_and_destination(self, graph, model_partition, sg_tensors):
        """
        Group SubgraphTensors by their origin tensor and destination device.
        Example:
            Op A has two output tensors, A:0 and A:1. A:0 is consumed by two ops in
            device 1, named B1 and C1, and one op in device 2, named B2. A:1 is consumed
            by one op in device 2, named C2.
            The input SubgraphTensors are:
            [(B1, 0, io.INPUT),  (C1, 0, io.INPUT), (B2, 0, io.INPUT), (C2, 0, io.INPUT)]

            The output groups are:
            Group 1: [(B1, 0, io.INPUT),  (C1, 0, io.INPUT)]
            Group 2: [(B2, 0, io.INPUT)]
            Group 3: [(C2, 0, io.INPUT)]

            Because the two SubgraphTensors in the first group share the same origin tensor and destination device.

            We will place exactly one pair of (SmpOutput, SmpInput) ops per computed group.
        """
        output_groups = collections.defaultdict(list)
        for sg_tensor in sg_tensors:
            # find origin tensor
            # (TODO) remove this once we added the control links back
            if sg_tensor.io == IO.CONTROL_INPUT:
                continue
            org_tensor = get_tensor(graph, sg_tensor)

            # find destination device
            dev = model_partition.get_asg_for_node(sg_tensor.node)

            key = (org_tensor.name, dev)
            output_groups[key].append(sg_tensor)

        return output_groups

    def _insert_smp_ops(self, graph, model_partition):
        """ Insert SmpInput and SmpOutput ops at AutoSubgraph boundaries and model
        inputs and outputs
        """
        asg_to_dummy = []
        smp_ops = collections.defaultdict(list)
        model_output_mapping = {}

        for asg_id, asg in enumerate(model_partition.auto_subgraphs):
            self.op_counter = 0
            smp_dummy = model_partition.dummies[asg_id]
            inserted = self.insert_smp_input(smp_dummy.outputs[0], None, asg_id, graph, dummy=True)
            smp_ops[asg_id].append(inserted)
            asg_to_dummy.append(inserted)

            #################################
            # TODO remove this block when cross-subgraph control input logic is implemented
            # for now we are breaking the control dependency
            cross_control_input_nodes = {}
            for idx, inp in enumerate(asg.inputs + asg.model_inputs):
                if inp.io == IO.CONTROL_INPUT:
                    control_input_name = (
                        graph.get_node_by_name(inp.node).control_inputs[inp.index].name
                    )
                    if inp.node in cross_control_input_nodes:
                        cross_control_input_nodes[inp.node].add(control_input_name)
                    else:
                        cross_control_input_nodes[inp.node] = {control_input_name}
            for node_name, control_input_names in cross_control_input_nodes.items():
                node = graph.get_node_by_name(node_name)
                new_control_input_names = {n.name for n in node.control_inputs}.difference(
                    control_input_names
                )
                new_control_inputs = [
                    graph.get_node_by_name(name) for name in new_control_input_names
                ]
                node.set_control_inputs(new_control_inputs)
            #################################

        for asg_id, asg in enumerate(model_partition.auto_subgraphs):
            # group output tensors, represented as SubgraphTensors, by their origin tensor and
            # destination device
            output_by_group = self.group_by_origin_and_destination(
                graph, model_partition, asg.outputs
            )

            # process asg.outputs - we place a single, shared SmpOutput op per group to save communication
            for grp_outputs in output_by_group.values():
                tensor = get_tensor(graph, grp_outputs[0])
                consumers = [out.node for out in grp_outputs]
                smp_output_op = self.insert_smp_output(
                    tensor, consumers, asg_to_dummy[asg_id].outputs[0], asg_id, graph
                )
                smp_ops[asg_id].append(smp_output_op)

            # process asg.model_outputs
            for idx, out in enumerate(asg.model_outputs):
                tensor = get_tensor(graph, out)
                smp_output_op = self.insert_smp_output(
                    tensor, None, asg_to_dummy[asg_id].outputs[0], asg_id, graph
                )
                smp_ops[asg_id].append(smp_output_op)
                model_output_mapping[tensor.name] = smp_output_op

        for asg_id, asg in enumerate(model_partition.auto_subgraphs):
            inputs_by_group = self.group_by_origin_and_destination(
                graph, model_partition, asg.inputs + asg.model_inputs
            )

            for grp_inputs in inputs_by_group.values():

                # TODO remove this when cross-subgraph control input logic is implemented
                # filter out control inputs
                grp_inputs = [inp for inp in grp_inputs if inp.io == IO.INPUT]

                if len(grp_inputs) > 0:
                    tensor = get_tensor(graph, grp_inputs[0])
                    consumers = [inp.node for inp in grp_inputs]

                    ins = self.insert_smp_input(tensor, consumers, asg_id, graph)
                    smp_ops[asg_id].append(ins)

        return graph, smp_ops, model_output_mapping

    def remove_unassigned_asgs(self, dev_g, model_partition, dev):
        """ Replace all AutoSubgraphs other than `dev` with SmpBundle ops"""
        for asg_id, asg in enumerate(model_partition.auto_subgraphs):
            if dev != asg_id:
                # this device does not own this subgraph - remove intermediate ops, replace with bundle op
                self.place_dummy_graph(dev_g, asg)

        graph_def = dev_g.to_graph_def()

        for asg_id, asg in enumerate(model_partition.auto_subgraphs):
            if dev != asg_id:
                self.remove_nodes(graph_def, asg.op_set)

        return graph_def

    def _replace_input(self, node, old_tensor, new_tensor):
        for i in range(len(node.inputs)):
            if node.inputs[i].name == old_tensor.name:
                node.replace_input(i, new_tensor)

    def insert_smp_input(self, tensor, consumers, asg_id, dev_g, dummy=False):
        smp_input_op = self.create_smp_input_op(tensor, asg_id, dev_g, dummy)
        # update_tensor_consumers(dev_g, tensor, smp_input_op, 0)

        if consumers is not None:
            for consumer in consumers:
                cons = dev_g.get_node_by_name(consumer)
                self._replace_input(cons, tensor, smp_input_op.outputs[0])

        return smp_input_op

    def insert_smp_output(self, tensor, consumers, control_node, asg_id, dev_g):
        smp_output_op = self.create_smp_output_op(tensor, control_node, asg_id, dev_g)

        if consumers is not None:
            for consumer in consumers:
                cons = dev_g.get_node_by_name(consumer)
                self._replace_input(cons, tensor, smp_output_op.outputs[0])
        # update_tensor_consumers(dev_g, tensor, smp_output_op, 0)

        return smp_output_op

    def create_smp_input_op(self, inp, asg_id, dev_g, dummy=False):
        name = f"SmpInput_asg{asg_id}_{self.op_counter}"
        # op_id = self.get_op_id(mb, asg_id, self.op_counter)

        new_node = dev_g.add_node(name, INPUT_OP)
        new_node.add_attr(OP_ID_ATTR, -1)
        new_node.add_attr(TICK_ATTR, -1)
        new_node.add_attr(LINK_ID_ATTR, -1)
        new_node.add_attr(PEER_ATTR, -1)
        new_node.add_attr(FORWARD_ATTR, True)
        new_node.add_attr(DUMMY_ATTR, dummy)
        new_node.add_attr(MICROBATCH_ATTR, -1)

        control_inputs = [
            gde.make_const(
                dev_g, f"SMPConst{i}_{asg_id}", np.array(0.0, dtype=np.float32), uniquify_name=True
            ).outputs[0]
            for i in range(2)
        ]
        outputs_of_control_inputs = [control_input for control_input in control_inputs]
        bundled_control_inputs = self.create_bundle_op(dev_g, outputs_of_control_inputs)

        input_tensors = [
            get_item_in_graph(dev_g, t) for t in [inp, bundled_control_inputs.outputs[0]]
        ]
        new_node.set_inputs(input_tensors)

        shape = [(dim if dim is not None else -1) for dim in inp.shape.as_list()]
        dtype = inp.dtype

        new_node.add_attr(EXPECTED_SHAPE_ATTR, shape)
        new_node.add_attr(OUT_TYPE_ATTR, dtype)
        new_node.add_attr(T_ATTR, inp.dtype)
        new_node.set_outputs_from_pairs([(dtype, make_tensor_shape(shape))])

        new_node.add_attr(CTRL_TYPE_ATTR, tf.float32)

        self.op_counter += 1

        return new_node

    def create_smp_output_op(self, inp, ctrl, asg_id, dev_g):
        name = f"SmpOutput_asg{asg_id}_{self.op_counter}"

        new_node = dev_g.add_node(name, OUTPUT_OP)
        new_node.add_attr(OP_ID_ATTR, -1)
        new_node.add_attr(TICK_ATTR, -1)
        new_node.add_attr(LINK_ID_ATTR, -1)
        new_node.add_attr(PEER_ATTR, -1)
        new_node.add_attr(FORWARD_ATTR, True)
        new_node.add_attr(MICROBATCH_ATTR, -1)

        input_tensors = [get_item_in_graph(dev_g, t) for t in [inp, ctrl]]
        new_node.set_inputs(input_tensors)

        new_node.add_attr(
            EXPECTED_SHAPE_ATTR, [(dim if dim is not None else -1) for dim in inp.shape.as_list()]
        )
        new_node.add_attr(OUT_TYPE_ATTR, inp.dtype)
        new_node.add_attr(T_ATTR, inp.dtype)
        new_node.add_attr(CTRL_TYPE_ATTR, tf.float32)

        new_node.set_outputs_from_pairs([(inp.dtype, make_tensor_shape(inp.shape))])
        self.op_counter += 1

        return new_node

    def place_dummy_graph(self, graph, asg):
        """ Create and insert SmpBundle ops that will "simulate" the ops in AutoSubgraph `asg`.
        To be called for AutoSubgraph's that are not assigned to the current device. """

        outputs = asg.outputs + asg.model_outputs
        for i, output in enumerate(outputs):
            output_tensor = get_tensor(graph, output)

            if i < len(asg.outputs):
                # if this is not a model output, go back 2 ops (SmpInput and SmpOutput)
                output_tensor = output_tensor.node.inputs[0].node.inputs[0]

            output_tensor_dtype = output_tensor.dtype
            output_node_name = output_tensor.node.name
            inputs = self.get_ancestors(graph, output_node_name)
            if len(inputs) == 0:
                # if this has no ancestor inputs, create a constant to feed into SmpBundle
                const_node = gde.make_const(
                    graph, f"SMPConst_input", np.array(0.0, dtype=np.float32), uniquify_name=True
                )
                inputs = [const_node.outputs[0]]

            smp_bundle_op = self.create_bundle_op(graph, inputs)

            # this.
            cast_op = self.create_cast_op(graph, smp_bundle_op.outputs[0], output_tensor_dtype)
            update_tensor_consumers(graph, output_tensor, cast_op, 0)

    def create_cast_op(self, graph, input_tensor, dtype):
        cast_op = graph.add_node("SmpBundleCast", "Cast", uniquify_name=True)
        cast_op.set_inputs([input_tensor])
        cast_op.add_attr("SrcT", input_tensor.dtype)
        cast_op.add_attr("DstT", dtype)
        cast_op.set_outputs_from_pairs([(dtype, make_tensor_shape(input_tensor.shape))])

        return cast_op

    def get_ancestors(self, graph, output_name):
        """ Get input tensors of this AutoSubgraph which are the ancestors of the node
        with the given name. """

        visited = set()
        stack = [graph.get_node_by_name(output_name)]
        ancestor_ops = set()

        while len(stack) > 0:
            node = stack.pop()
            if node.name not in visited:
                visited.add(node.name)
                if node.op_type == INPUT_OP:
                    ancestor_ops.add(node)
                else:
                    input_ops = [get_op(inp) for inp in node.inputs]
                    control_input_ops = [get_op(inp) for inp in node.control_inputs]
                    stack.extend(input_ops + control_input_ops)

        return [op.outputs[0] for op in ancestor_ops]

    def create_bundle_op(self, graph, inputs):
        new_node = graph.add_node(BUNDLE_OP, BUNDLE_OP, uniquify_name=True)
        new_node.add_attr(N_ATTR, len(inputs))
        new_node.add_attr(DTYPE_ATTR, tf.float32)
        new_node.set_inputs(inputs)
        new_node.set_outputs_from_pairs([(tf.float32, make_tensor_shape([]))])

        return new_node

    def preprocess_consumers(self, graph):
        consumer_map = collections.defaultdict(set)
        for node in graph.nodes:
            for inp in node.inputs + node.control_inputs:
                if isinstance(inp, gde_Tensor):
                    inp_node_name = inp.node.name
                else:
                    ## If inp is of type node
                    inp_node_name = inp.name
                consumer_map[inp_node_name] = consumer_map[inp_node_name].union([node])

        return consumer_map

    def get_reverse_topological_sort(self, graph):
        """Conducts the reverse toplogical sort. Resultant list consists of nodes such that node at
        position i can never have an outgoing edge to a node a position j if i < j.

        Once the dfs for a node, say A, is finished, it can be appended to the end of the rev_toposort list. This
        can be done because all the nodes that were the children of A are already in the list. Hence, when we
        append A to the rev_toposort list, we are sure that edges can only exist from A to any other node in the list
        and not the other way round.

        To check when the dfs for a node has been finished, we attach a marker to each node. While popping the nodes,
        if the node has a NODE_MARKER attached, it means that we have to conduct a simple DFS on that node.
        Before we start the DFS process for any node, we attach a START_MARKER to that node and insert it into
        the stack.
        While popping the nodes, if we come across a node that has a START_MARKER attached, then it means that all the
        children nodes to this node have been processed and that the DFS for this node is finished.
        """
        START_MARKER = 1
        NODE_MARKER = 2
        visited = set()
        rev_toposort = []
        consumer_map = self.preprocess_consumers(graph)

        def dfs(start_node):
            stack = []
            stack.append((NODE_MARKER, start_node))
            while len(stack) > 0:
                mark_type, node = stack.pop()
                if mark_type == START_MARKER:
                    rev_toposort.append(node)
                    continue
                if node in visited:
                    continue
                visited.add(node)
                consumers = []
                if node.name in consumer_map:
                    consumers = consumer_map[node.name]
                stack.append((START_MARKER, node))
                for output_node in consumers:
                    if output_node not in visited:
                        stack.append((NODE_MARKER, output_node))

        for node in graph.nodes:
            if node not in visited:
                dfs(node)

        return rev_toposort

    def remove_nodes(self, graph_def, nodes):
        """ Iterative op removal from GraphDef """
        get_logger().debug(f"Removing nodes: {nodes}")
        to_remove = []
        for i, node in enumerate(graph_def.node):
            if node.name in nodes:
                to_remove.append(i)
        for ind in reversed(to_remove):
            graph_def.node.pop(ind)

    def remove_nodes_from_gde_graph(self, graph, nodes):
        """ Topological-sort based op removal from gde.Graph """
        get_logger().debug(f"Conducting topological sort")
        reverse_toposorted_graph = self.get_reverse_topological_sort(graph)
        remaining_nodes = {name for name in nodes}
        get_logger().debug(f"Total number of nodes to remove: {len(remaining_nodes)}")

        for node in reverse_toposorted_graph:
            if node.name in remaining_nodes:
                graph.remove_node_by_name(node.name)
                remaining_nodes.remove(node.name)

        get_logger().debug(f"Remaining nodes that need to be removed: {len(remaining_nodes)}")
        if len(remaining_nodes) > 0:
            raise ValueError("Something is wrong. All nodes not removed.")

    def replace_tensor_name(self, elem, old_name, new_name):
        if (
            isinstance(elem, str)
            and elem.startswith(_TF_OBJECT_PREFIX)
            and elem == _TF_OBJECT_PREFIX + old_name
        ):
            return _TF_OBJECT_PREFIX + new_name
        else:
            return elem


class ModelPartition:
    """
    Represents a partitioning of the model graph. Maintains a list of the AutoSubgraphs,
    exactly one per device. Guarantees that the AutoSubraphs represent an exhaustive
    and non-overlapping partitioning of the ops in the model, except for model input
    placeholders.
    """

    def __init__(self, graph, asg_list):
        self.auto_subgraphs = asg_list
        self.graph = graph
        self._verify_partition()
        self.dummies = self._create_dummies()

    def _verify_partition(self):
        all_nodes = {n.name for n in self.graph.nodes if not is_model_input(n)}
        asg_nodes = set()

        for dev, asg in enumerate(self.auto_subgraphs):
            previous_size = len(asg_nodes)
            asg_nodes = asg_nodes.union(asg.op_set)

            if len(asg_nodes) < previous_size + len(asg.op_set):
                raise ValueError(
                    f"AutoSubgraph for device {dev} has ops previously assigned to another AutoSubgraph!"
                )
        if len(asg_nodes) != len(all_nodes):
            get_logger().debug(
                f"asg_nodes - all_nodes {asg_nodes.difference(all_nodes)} all_nodes - asg_nodes {all_nodes.difference(asg_nodes)}"
            )
            raise ValueError("AutoSubgraphs do not form a complete partition of the model!")

    def get_asg_for_node(self, node_name):
        for idx, asg in enumerate(self.auto_subgraphs):
            if node_name in asg.op_set:
                return idx
        raise ValueError(f"Cannot find {node_name} in any AutoSubgraph.")

    def _create_dummies(self):
        dummies = []
        for asg_id, asg in enumerate(self.auto_subgraphs):
            dummies.append(
                make_placeholder(
                    self.graph,
                    f"SMPDistributedModel/SMPDummy_{asg_id}",
                    dtype=tf.float32,
                    shape=make_tensor_shape([]),
                )
            )
        return dummies


class AutoSubgraph:
    """
    Represents a single model partition assigned to single device. Maintains model inputs and
    model outputs that belong to the current subgraph, as well as input and output tensors from
    other AutoSubgaphs.
    """

    def __init__(self, dev, graph, op_set):
        self.dev = dev
        self.graph = graph
        self.op_set = op_set
        self.inputs = []
        self.outputs = []
        self.model_inputs = []
        self.model_outputs = []
        self.vars = []
        self._find_terminals()

        get_logger().debug(
            f"ASG ID: {dev}, INPUTS: {self.inputs}, OUTPUTS: {self.outputs}, MODEL INPUTS: {self.model_inputs}, MODEL OUTPUTS: {self.model_outputs}"
        )

    def get_non_input_nodes(self):
        model_input_nodes = [node for node, _, _ in self.model_inputs]
        return {name for name in self.op_set if name not in model_input_nodes}

    def _find_terminals(self):

        if len(self.op_set) == 0:
            return

        for node_name in set_to_sorted_list(self.op_set):
            node = self.graph.get_node_by_name(node_name)

            node_inputs = {n.node.name for n in node.inputs}
            node_inputs = node_inputs.union({get_op(n).name for n in node.control_inputs})

            node_outputs = set()
            for t in node.outputs:
                node_outputs = node_outputs.union(set([cons.name for cons in t.consumers()]))

            node_model_inputs = {
                inp for inp in node_inputs if is_model_input(self.graph.get_node_by_name(inp))
            }
            node_inputs_outside_asg = node_inputs.difference(self.op_set).difference(
                node_model_inputs
            )
            node_outputs_outside_asg = node_outputs.difference(self.op_set)

            if is_var(node):
                self.vars.append(node.name)

            node_inputs_outside_asg = set_to_sorted_list(node_inputs_outside_asg)
            node_outputs_outside_asg = set_to_sorted_list(node_outputs_outside_asg)
            node_model_inputs = set_to_sorted_list(node_model_inputs)

            # inputs: tensors incoming from another AutoSubgraph
            for inp_node in node_inputs_outside_asg:
                inp_tensors = list(node.inputs) + list(node.control_inputs)
                for idx, inp_tensor in enumerate(inp_tensors):
                    if get_op(inp_tensor).name == inp_node:
                        if idx < len(node.inputs):
                            self.inputs.append(SubgraphTensor(node.name, idx, IO.INPUT))
                        else:
                            self.inputs.append(
                                SubgraphTensor(node.name, idx - len(node.inputs), IO.CONTROL_INPUT)
                            )
                        break

            # model inputs: input tensors of the entire model graph
            for inp in node_model_inputs:
                inp_node = self.graph.get_node_by_name(inp)
                self.model_inputs.append(self._get_tensor(node.name, inp_node.outputs[0]))

            # outputs: tensors outgoing into another AutoSubgraph
            for out_node in node_outputs_outside_asg:
                for idx, out_tensor in enumerate(node.outputs):
                    consumer_names = [cons.name for cons in out_tensor.consumers()]
                    if out_node in consumer_names:
                        self.outputs.append(self._get_tensor(out_node, out_tensor))
                        break

            # model outputs: output tensors of the entire model graph
            for idx, out in enumerate(node.outputs):
                if out.name in state.serialized_graph.model_outputs[0][0]:
                    self.model_outputs.append(SubgraphTensor(node.name, idx, IO.OUTPUT))

    def _get_tensor(self, node_name, tensor):
        """ Get the SubgraphTensor of tensor with reference to a consumer node_name. """

        node = self.graph.get_node_by_name(node_name)
        for idx, t in enumerate(node.inputs):
            if t.name == tensor.name:
                return SubgraphTensor(node.name, idx, IO.INPUT)
        raise ValueError(
            f"Something is wrong - tensor {tensor.name} is not an input of {node_name}."
        )
