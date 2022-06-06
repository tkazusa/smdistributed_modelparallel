# Third Party
# Standard Library
import os

import tensorflow as tf
from tensorflow.python.framework import ops

# First Party
from smdistributed.modelparallel.tensorflow import core
from smdistributed.modelparallel.tensorflow.graph_utils import is_var
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.state_mod import state
from smdistributed.modelparallel.tensorflow.utils import (
    bcast_ckpt_files,
    gather_ckpt_files,
    get_op,
    is_tensor_or_var_or_op,
    raise_if_has_cross_microbatch_dependencies,
    receive_checkpoint_files,
    send_checkpoint_files,
)


def _get_dir_list_and_prefixes(ckpt_path):
    """ Given a checkpoint dir, function returns all the directories and checkpoint prefixes in each directory
        If there is no directory, then all the prefixes are in the ckpt_path
        For example:
        The directory can contain checkpoints for all ranks in individual folders
        rank_1 : smp-600.index, smp-1200.index
        rank_2 : smp-600.index, smp-1200.index

        This will return dir_list = [ckpt_path/rank_1, ckpt_path/rank_2] , prefixes=[smp-600, smp-1200]
        """
    dir_list = []

    for x in sorted(os.listdir(ckpt_path)):
        full_path_dir = os.path.join(ckpt_path, x)
        if os.path.isdir(full_path_dir):
            dir_list.append(full_path_dir)

    prefixes = []
    if dir_list:
        for x in os.listdir(dir_list[0]):
            if x.endswith(".index"):
                prefixes.append(x.rsplit(".", 1)[0])
    else:
        dir_list.append(ckpt_path)
        # if no directories then all the ckpt files are expected in ckpt_path
        for x in os.listdir(ckpt_path):
            if x.endswith(".index"):
                prefixes.append(x.rsplit(".", 1)[0])

    return prefixes, dir_list


def get_matching_variables_from_ckpt(graphdef, ckpt_path, prefix, imported_graph=False):
    """ This function returns name of variables in checkpoint path with given prefix which are
        also present in the graphdef provided
    """

    # Getting variables from the graphdef
    graph_variables = set()
    for cur_node in graphdef.node:
        name = cur_node.name
        if is_var(cur_node):
            graph_variables.add(name)

    # Going through all the checkpoints and finding all the variables
    # that can be restored from that chckpoint file.
    var_to_restore = {}
    _, dir_list = _get_dir_list_and_prefixes(ckpt_path)

    for sub_dir in dir_list:
        ckpt_file_path = os.path.join(sub_dir, prefix)

        var_on_rank = []
        for var_tuple in tf.train.list_variables(ckpt_file_path):
            var_name = var_tuple[0]

            if imported_graph:
                # if working on imported graph, then compare with variables directly.
                # if var_name.startswith("import_") and var_name in graph_variables:
                if "SMPDummy" not in var_name and var_name in graph_variables:
                    var_on_rank.append(var_name)

            else:
                # since the graphdef was imported on each rank, the graph nodes will start with 'import_'
                # if var_name.startswith("import_") and var_name.split("/", 1)[1] in graph_variables:
                var_name_with_first = var_name.split("/", 1)
                probable_name_in_variables = (
                    var_name_with_first[1]
                    if len(var_name_with_first) > 1
                    else var_name_with_first[0]
                )
                if "SMPDummy" not in var_name and probable_name_in_variables in graph_variables:
                    var_on_rank.append(var_name)
        var_to_restore[sub_dir] = var_on_rank

    return var_to_restore


def _get_prefix_without_step(ckpt_path):
    """ Returns checkpoint prefix without  step number attached
        For example: prefix: model.ckpt-1800, returns model.ckpt
    """
    prefixes, _ = _get_dir_list_and_prefixes(ckpt_path)

    # prefix without step number will be same in all ranks, need just one prefixes
    prefix = prefixes[0]
    # prefix could be model.ckpt-600.index. model.ckpt.index
    # getting rid fo .index and then spliting on "-" will give model.ckpt
    return prefix.rsplit(".", 1)[0].rsplit("-", 1)[0]


def save_model(internal_ckpt_path, internal_prefix, ckpt_path, prefix=None):
    """
        Save the all the checkpoints in different ranks to one checkpoint per step.
    """
    if core.dp_rank() == 0 and core.pp_rank() == 0:

        graphdef = state.serialized_graph.graph_def

        var_to_restore = get_matching_variables_from_ckpt(
            graphdef, internal_ckpt_path, internal_prefix
        )

        # Resetting config to override outer context config.
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(core.pp_rank())

        added_vars = set()
        # Creating a private session to restore and then save the final ckpt.
        with tf.compat.v1.Session(graph=tf.compat.v1.Graph(), config=config) as sess:

            # importing graphdef with "import_0" prefix. This done so that node names
            # match the checkpoint variable names.
            # TODO: look to see 'init_from_checkpoint' can be used to avoid this.
            tf.compat.v1.graph_util.import_graph_def(graphdef, name="import_0")

            for dir_name, var_rank_list in var_to_restore.items():
                # dict of variable name-> tensor_name. This will be used for restore.
                filtered_var_restore = {}
                for var_name in var_rank_list:
                    var_tensor = sess.graph.get_tensor_by_name(var_name + ":0")
                    filtered_var_restore[var_name] = var_tensor

                    # adding var to collection only once.
                    if var_name not in added_vars:
                        # Adding variable to collection, this needed for final save
                        tf.compat.v1.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, var_tensor)
                        added_vars.add(var_name)

                # Restoring variables from checkpoint file
                restore_file = os.path.join(dir_name, internal_prefix)
                _restore_saver = tf.compat.v1.train.Saver(filtered_var_restore)
                _restore_saver.restore(
                    sess,
                    # tf.compat.v1.train.latest_checkpoint(os.path.join(dir_name)
                    restore_file,
                )

            if prefix is None:
                # If prefix is not provided, then use the prefix from the internal_ckpt_path.
                prefix = internal_prefix

            # final save
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            try:
                _save_saver = tf.compat.v1.train.Saver()
                _save_saver.save(sess, os.path.join(ckpt_path, prefix))
            except ValueError as ex:
                get_logger().info(ex)


def sort_prefix_on_step(prefixes):
    if prefixes and len(prefixes) and "-" in prefixes[0]:
        return sorted(prefixes, key=lambda y: int(y.rsplit("-", 1)[1]))

    return prefixes


def combine_checkpoints(src_ckpt_dir=None, dest_ckpt_dir=None):
    """ Combines all the checkpoints from all ranks to single file per checkpoint step.
        Return: Returns the path to the merged ckpt
    """
    if src_ckpt_dir is None:
        src_ckpt_dir = state.checkpoint_manager.ckpt_dir

    if dest_ckpt_dir is None:
        # set the merged_ckpt dir as default Sagemaker checkpoint dir
        dest_ckpt_dir = os.path.join("/opt/ml/checkpoint", "smp_merged_ckpt")
    else:
        if os.path.realpath(dest_ckpt_dir).startswith(os.path.realpath(src_ckpt_dir)):
            raise ValueError(
                f' Destination path: "{dest_ckpt_dir}" cannot be a subdirectory of Source path: "{src_ckpt_dir}"'
            )

    # Accumulate all ckpt files on rank 0 in a multinode case.
    gather_ckpt_files(src_ckpt_dir, src_ckpt_dir, receiver_pp_rank=0)

    if core.dp_rank() == 0 and core.pp_rank() == 0:
        prefixes, _ = _get_dir_list_and_prefixes(src_ckpt_dir)
        for prefix in sort_prefix_on_step(prefixes):
            save_model(src_ckpt_dir, prefix, dest_ckpt_dir)

        get_logger().info(f'Merged checkpoint saved in "{dest_ckpt_dir}" ')

    core.barrier()

    return dest_ckpt_dir
