# Standard Library
import collections
import os

# Third Party
import tensorflow as tf
from tensorflow.python.training.tracking import util as trackable_utils

# First Party
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.state_mod import state


def get_layers(model):
    """ Call _flatten_layers to recursively get all the nested layers given a keras model """
    model_layers = []
    for layer in model.layers:
        model_layers.extend([sl for sl in _flatten_layers(layer)])
    if len(model_layers) == 0:
        get_logger().warning("Model has no layers.")

    return model_layers


@tf.autograph.experimental.do_not_convert
def _flatten_layers(p_layer, recursive=True, include_self=True):
    """ Recursively get all the nested layers given a keras layer. Note this logic is lifted from
    tf master branch (scheduled for r2.3 release) """
    if include_self:
        yield p_layer

    # Only instantiate set and deque if needed.
    layers_or_containers = tf.nest.flatten(getattr(p_layer, "_layers", None))

    if layers_or_containers:
        seen_object_ids = set()
        deque = collections.deque(layers_or_containers)
        while deque:
            layer_or_container = deque.popleft()

            layer_or_container_id = id(layer_or_container)
            if layer_or_container_id in seen_object_ids:
                continue
            seen_object_ids.add(layer_or_container_id)

            if isinstance(layer_or_container, tf.keras.layers.Layer):
                yield layer_or_container
                # Introspect recursively through sublayers.
                if recursive:
                    sublayers = tf.nest.flatten(getattr(layer_or_container, "_layers", None))
                    if sublayers:
                        deque.extendleft(reversed(sublayers))


def load_ckpt_data_to_var(model_variables, opt_var):
    """
        Reads the variable value from checkpoint and assigns it to model variables.
    """

    if state.checkpoint_manager.restore_ckpt and os.path.exists(
        state.checkpoint_manager.restore_ckpt + ".index"
    ):

        # state.checkpoint_manager.restore_ckpt will be like: "ckpt_dir/mp_rank_0/ckpt-5"
        # splitting twice to get "ckpt_dir"
        restore_dir = os.path.realpath(state.checkpoint_manager.restore_ckpt)
        restore_dir, prefix = restore_dir.rsplit("/", 1)
        restore_dir = restore_dir.rsplit("/", 1)[0]

        rank_dirs = []
        for root, dirs, files in tf.io.gfile.walk(restore_dir):
            for name in dirs:
                rank_dirs.append(os.path.join(root, name))

        # Load the variables from checkpoint in all ranks.
        # This is done to make checkpoint forward compatible, incase,
        # the partition algorithm changes the partition variables in each partition. Making sure
        # the variable in the new partition gets the checkpointed value.
        for rank_dir in rank_dirs:
            restore_ckpt = os.path.join(rank_dir, prefix)
            # Getting all the variables in ckpt
            variables = tf.train.list_variables(restore_ckpt)

            objects = trackable_utils.object_metadata(restore_ckpt)

            # map from variable name to checkpoint key
            map_variable_name_to_ckpt_name = {}
            for obj in objects.nodes:
                for attribute in obj.attributes:
                    map_variable_name_to_ckpt_name[attribute.full_name] = attribute.checkpoint_key

            # For variables present both in model_variables and ckpt.
            # Assign value from ckpt to the model variable
            for var_list in [model_variables, opt_var]:
                if var_list:
                    for var in var_list:
                        var_name = (var.name).rsplit(":", 1)[0]
                        if var_name in map_variable_name_to_ckpt_name:
                            # Get the value from ckpt.
                            value = tf.train.load_variable(
                                restore_ckpt, map_variable_name_to_ckpt_name[var_name]
                            )

                            # Assigning the value to model variable.
                            var.assign(tf.convert_to_tensor(value))
