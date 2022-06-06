# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel.backend.logger import get_logger as backend_get_logger

if int(tf.__version__.split(".")[0]) == 2:

    @tf.autograph.experimental.do_not_convert
    def get_logger():
        return backend_get_logger()
