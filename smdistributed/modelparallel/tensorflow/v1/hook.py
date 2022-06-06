# Third Party
# Standard Library
import os
import shutil

import tensorflow as tf

# First Party
from smdistributed.modelparallel.tensorflow import core
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
from smdistributed.modelparallel.tensorflow.v1.utils import (
    get_matching_variables_from_ckpt,
    save_model,
)


class SMPCompileHook(tf.compat.v1.train.SessionRunHook):
    def begin(self):
        """Do preparation before creating the session."""

        # raise an error if there are cross microbatch dependencies
        raise_if_has_cross_microbatch_dependencies(tf.compat.v1.get_default_graph())

        # establish the control dependencies if XLA is enabled
        state.ctrl_mgr.maybe_reroute_control_inputs()

    def after_create_session(self, session, coord):
        """If we detect XLA is enabled for TF but not for SMP, raise an error."""

        try:
            tf_xla_flags = os.environ.get("TF_XLA_FLAGS")
            if tf_xla_flags and (
                "--tf_xla_auto_jit=2" in tf_xla_flags or "--tf_xla_auto_jit=1" in tf_xla_flags
            ):
                tf_xla_enabled = True
            elif session._config.graph_options.optimizer_options.global_jit_level > 0:
                tf_xla_enabled = True
            else:
                tf_xla_enabled = False

        except AttributeError:
            tf_xla_enabled = False

        # xla flag is removed from config
        # if not state.cfg.xla and tf_xla_enabled:
        #     raise ValueError(
        #         "XLA is enabled in TensorFlow but not in SMP! Set the 'xla' flag to True while initializing SMP."
        #     )

    def before_run(self, run_context):
        """Before every Session.run(), compute the counts for the ops of each behavior type, for the
           current set of fetches."""
        state.core.timeline_start_step()
        key = _create_key(state.compile_graph, run_context.original_args.fetches)

        # If this is a new key, computes io_counts and saves the compiler state into a cache.
        # Otherwise does nothing.
        state.compiler.save_state(key, run_context.original_args.fetches)

        # Loads the cached compiler state if key is different from the lasy key, otherwise does nothing.
        state.compiler.load_state(key)
        state.compiler.register_backend_state(key)

        # state.compiler.register_io_counts(run_context.original_args.fetches)
        state.core.proxy_wake_up()

    def after_run(self, run_context, run_values):
        """Before every Session.run(), run timeline postprocess to gather all records from different ranks and write file
        """
        state.core.timeline_end_step()


def _create_key(graph, fetches):
    """ Create hashable key per (TraceGraph, fetches) tuple"""
    flat_fetches = tf.nest.flatten(fetches)
    flat_fetches = [get_op(x) for x in flat_fetches if is_tensor_or_var_or_op(x)]
    sorted_fetches = tuple(sorted(flat_fetches, key=lambda x: x.name))
    return (id(graph), sorted_fetches)


class SMPCheckpointHook(tf.compat.v1.train.CheckpointSaverHook):
    """ Load and Save checkpoints  """

    def __init__(
        self,
        save_checkpoint_dir=None,
        load_checkpoint_dir=None,
        save_secs=None,
        save_steps=None,
        checkpoint_basename="model.ckpt",
        load_prefix=None,
    ):

        if not save_checkpoint_dir and not load_checkpoint_dir:
            raise ValueError(
                f"Either one of save_checkpoint_dir or load_checkpoint_dir should be provided"
            )

        # if save_checkpoint_dir is provided, we use the dir to load the checkpoint if present.
        load_checkpoint_dir = load_checkpoint_dir if load_checkpoint_dir else save_checkpoint_dir

        self._load_ckpt_path = (
            os.path.realpath(load_checkpoint_dir) if load_checkpoint_dir else None
        )
        self._save_ckpt_path = (
            os.path.realpath(save_checkpoint_dir) if save_checkpoint_dir else None
        )

        if self._save_ckpt_path:
            # Adding rank to the save path
            save_rank_dir = os.path.join(self._save_ckpt_path, "mp_rank_" + str(core.rank()))

            # Calling base class to initialize for save.
            super().__init__(
                save_rank_dir,
                save_secs=save_secs,
                save_steps=save_steps,
                checkpoint_basename=checkpoint_basename,
            )

        # Saving the checkpoint directory
        state.checkpoint_manager.ckpt_dir = save_checkpoint_dir

        # load_prefix is used to specify a given step prefix. i.e model.ckpt-1000
        self._prefix = load_prefix
        self._restore_saver = None

    def begin(self):
        if self._save_ckpt_path:
            super().begin()

        # broadcast the checkpoint from rank 0 to ranks on different node.
        # This is one time cost before training starts.
        bcast_ckpt_files(self._load_ckpt_path, sender_pp_rank=0)

        # if given path is directory:
        # Check if there is subdirectories i.e pp_rank_0, pp_rank_1:
        #    if yes, then add pp_rank to the path to reach ckpt/pp_rank_0/
        # else if no directories :
        #    then all the checkpoints are in this directory.

        if not os.path.exists(self._load_ckpt_path):
            get_logger().info(
                f' No checkpoints loaded. Directory  "{self._load_ckpt_path}" does not exist'
            )
            return

        dir_list = next(os.walk(self._load_ckpt_path))[1]

        # if the directory has subdirectories, adding rank dir to the path.
        if len(dir_list) > 0:
            self._load_ckpt_path = os.path.join(
                self._load_ckpt_path, "mp_rank_" + str(core.pp_rank())
            )

        # Adding prefix if provided, or fetch the latest from the path
        if self._prefix:
            self._latest_ckpt = os.path.join(self._load_ckpt_path, self._prefix)
        else:
            self._latest_ckpt = tf.train.latest_checkpoint(self._load_ckpt_path)

        graphdef = tf.compat.v1.get_default_graph().as_graph_def()

        # Directory is created when training starts, but there will be no checkpoints yet.
        if not self._latest_ckpt or not os.path.exists(self._latest_ckpt + ".index"):
            return

        # get the common variables in graphdef and checkpoint
        ckpt_dir, prefix = self._latest_ckpt.rsplit("/", 1)
        var_to_restore = get_matching_variables_from_ckpt(
            graphdef, ckpt_dir, prefix=prefix, imported_graph=True
        )

        # Finding the variables to restore.
        filtered_var_restore = {}

        for dir_name, var_rank_list in var_to_restore.items():
            # dict of variable name-> tensor_name. This will be used for restore.
            for var_name in var_rank_list:
                var_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(var_name + ":0")
                filtered_var_restore[var_name] = var_tensor

        # Creating a saver with var to restore.
        self._restore_saver = tf.compat.v1.train.Saver(filtered_var_restore)

    def after_create_session(self, session, coord):
        """If saver was created restore the checkpoint."""
        if self._save_ckpt_path:
            super().after_create_session(session, coord)

        if self._restore_saver:
            # Restoring after session is created.
            self._restore_saver.restore(session, self._latest_ckpt)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        if self._save_ckpt_path:
            return super().before_run(run_context)

    def after_run(self, run_context, run_values):
        if self._save_ckpt_path:
            super().after_run(run_context, run_values)

    def end(self, session):
        if self._save_ckpt_path:
            super().end(session)


class SMPSaveModelHook(tf.compat.v1.train.SessionRunHook):
    """ Saves the final checkpoint into path provided """

    def __init__(self, save_path, ckpt_prefix="smp-tf-latest"):
        self._ckpt_path = save_path
        self._ckpt_prefix = ckpt_prefix

    def begin(self):
        self._saver = tf.compat.v1.train.Saver()

    def end(self, session):
        # all ranks:
        #    -- Save a checkpoint

        _internal_ckpt_location = "/tmp/smp_ckpt/"
        _internal_prefix = "smp_tf_model"

        if core.dp_rank() == 0:
            save_file_dir = os.path.join(_internal_ckpt_location, "mp_rank_" + str(core.pp_rank()))
            if not os.path.exists(save_file_dir):
                os.makedirs(save_file_dir, exist_ok=True)
            self._saver.save(session, os.path.join(save_file_dir, _internal_prefix))

            # send ckpt from all other ranks and receive in rank 0.
            gather_ckpt_files(
                send_ckpt_path=_internal_ckpt_location,
                receive_ckpt_path=_internal_ckpt_location,
                receiver_pp_rank=0,
            )

            # Save the final checkpoint
            if core.pp_rank() == 0:
                save_model(
                    _internal_ckpt_location, _internal_prefix, self._ckpt_path, self._ckpt_prefix
                )

                # clean up
                if os.path.exists(_internal_ckpt_location):
                    shutil.rmtree(_internal_ckpt_location)

            core.barrier()
            # clean up
            if os.path.exists(_internal_ckpt_location):
                shutil.rmtree(_internal_ckpt_location)
