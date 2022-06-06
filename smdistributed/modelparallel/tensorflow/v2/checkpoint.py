# Standard Library
import os

# Third Party
import tensorflow as tf
from tensorflow.python.training import checkpoint_management

# First Party
from smdistributed.modelparallel.tensorflow import core, state
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.v2.step_mod import register_post_partition_hook


class CheckpointManager(checkpoint_management.CheckpointManager):
    def __init__(
        self, checkpoint, directory="/opt/ml/checkpoints", max_to_keep=None, checkpoint_name="ckpt"
    ):

        smp_directory = os.path.join(directory, "mp_rank_" + str(core.pp_rank()))
        super(CheckpointManager, self).__init__(
            checkpoint=checkpoint,
            directory=smp_directory,
            max_to_keep=max_to_keep,
            checkpoint_name=checkpoint_name,
        )

        self._base_directory = directory
        self._ckpt_latest_prefix = tf.train.latest_checkpoint(smp_directory)
        self._ckpt_latest_prefix = (
            self._ckpt_latest_prefix.rsplit("/", 1)[1]
            if self._ckpt_latest_prefix != None
            else self._ckpt_latest_prefix
        )

    def restore(self, restore_path=None, restore_prefix=None):
        if state._is_first_step:
            register_post_partition_hook(self._restore)(restore_path, restore_prefix)
        else:
            self._restore(restore_path, restore_prefix)

    def _restore(self, restore_path=None, restore_prefix=None):
        """
        Args:
            restore_path: str, path to restore from.
            restore_prefix: str, prefix to restore from. i.e "ckpt-5", will restore from ckpt-5 instead of latest.
        """
        restore_path = self._base_directory if restore_path == None else restore_path

        # List of all the mp rank dir.
        rank_dirs = []
        for root, dirs, files in tf.io.gfile.walk(restore_path):
            for name in dirs:
                rank_dirs.append(os.path.join(root, name))

        restore_ckpt = None
        # Load the variables from checkpoint in all ranks.
        # This is done to make checkpoint forward compatible, incase,
        # the partition algorithm changes the partition variables in each partition. Making sure
        # the variable in the new partition gets the checkpointed value.
        for rank_dir in rank_dirs:
            if restore_prefix:
                restore_ckpt = os.path.join(rank_dir, restore_prefix)
            else:
                if os.path.join(restore_path, "mp_rank_" + str(core.pp_rank())) == self.directory:
                    # This is done since restore is called after first step
                    # the 'checkpoint'  will point to checkpoint that points to
                    # initial values.
                    if self._ckpt_latest_prefix:
                        restore_ckpt = os.path.join(rank_dir, self._ckpt_latest_prefix)
                else:
                    restore_ckpt = tf.train.latest_checkpoint(rank_dir)

            if restore_ckpt and os.path.exists(restore_ckpt + ".index"):
                self.checkpoint.restore(restore_ckpt).expect_partial()
                get_logger().info(f"Checkpoint restored from: {restore_ckpt}")
            else:
                get_logger().warning(f"No checkpoints restored.")
