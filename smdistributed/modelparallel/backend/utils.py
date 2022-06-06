# Standard Library
import bisect
import math
import sysconfig

# First Party
from smdistributed.modelparallel.backend.logger import get_logger

try:
    from smexperiments.metrics import SageMakerFileMetricsWriter
except ImportError as error:
    SageMakerFileMetricsWriter = None


class ParentException(Exception):
    pass


def get_divisibility_error_str(framework, batch_size, num_mb):
    if framework == "pytorch":
        last_batch_solution = "you can set drop_last=True in your DataLoader to skip this batch, or you can manually skip this batch"
    else:
        last_batch_solution = "you can set drop_remainder=True in the .batch() method of tf.data.Dataset to skip this batch"

    return f"Batch size must be divisible by the number of microbatches. Found batch size={batch_size}, number of microbatches={num_mb}. If this is the last batch of the epoch, size of the current batch might be smaller than the regular batch size you chose. If this is the case, {last_batch_solution}. If this is a tensor with no batch dimension, you can specify its argument name in 'non_split_inputs' argument to smp.step, so that smp does not attempt to split this tensor. If the batch dimension is not 0th axis, you can specify the batch axis within 'input_split_axis' argument to smp.step function. For details, visit SageMaker distributed model parallelism documentation for smp.step: https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel.html"


def deprecated(alternative=None):
    def deprecation_wrapper(func):
        def wrapped(*args, **kwargs):
            warn_str = f"{func.__name__} is deprecated and will be removed in a future version."
            if alternative is not None:
                warn_str += f" Consider using {alternative} instead."

            get_logger().warning(warn_str)
            return func(*args, **kwargs)

        return wrapped

    return deprecation_wrapper


def flatten(list_of_lists):
    buf = []
    buf.append(len(list_of_lists))
    for i in range(len(list_of_lists)):
        buf.append(len(list_of_lists[i]))
        buf.extend(list_of_lists[i])
    return buf


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var("SO")
    if ext_suffix:
        return ext_suffix

    return ".so"


def add_to_args(args, kwargs, inp, k):
    if isinstance(k, int):
        args.append(inp)
    else:
        kwargs[k] = inp


def replace_arg(args, kwargs, inp, k):
    if isinstance(k, int):
        args[k] = inp
    else:
        kwargs[k] = inp


def bijection_2d(x, y):
    """Constructs a bijection between NxN and N, where N is the set of natural numbers. """

    diagonal_id = x + y
    sum_prev_diagonals = diagonal_id * (diagonal_id + 1) // 2
    return 1 + sum_prev_diagonals + x


def bijection_3d(x, y, z):
    """ Compose 2-D bijections recursively to construct a 3-D bijection"""

    arg = bijection_2d(x, y)
    return bijection_2d(arg, z)


def bijection_4d(x, y, z, w):
    """ Compose 2-D bijections recursively to construct a 4-D bijection"""
    arg1, arg2 = bijection_2d(x, y), bijection_2d(z, w)
    return bijection_2d(arg1, arg2)


def invert_bijection_2d(t):
    """Find the set of arguments that produced the current id."""
    tm = t - 1
    diagonal_id = int(math.floor((math.sqrt(1 + 8 * tm) - 1) / 2))
    sum_prev_diagonals = diagonal_id * (diagonal_id + 1) // 2
    return tm - sum_prev_diagonals, diagonal_id + sum_prev_diagonals - tm


def invert_bijection_4d(t):
    """Find the set of arguments that produced the current id."""
    arg1, arg2 = invert_bijection_2d(t)
    x, y = invert_bijection_2d(arg1)
    z, w = invert_bijection_2d(arg2)

    return x, y, z, w


def get_time_stamp():
    import time

    timestamp = time.time()
    return timestamp


def find_ge(a, x):
    """Find leftmost item greater than or equal to x, in a sorted list.
    Taken from bisect module documentation"""
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError


def upload_metrics_to_studio(metrics):
    """
    Upload metrics to the Sagemaker Studio
    """
    if SageMakerFileMetricsWriter == None:
        get_logger().warning(
            "Failed to import SageMakerFileMetricsWriter, metrics will not be uploaded to SageMaker Studio."
        )
        return

    writer = SageMakerFileMetricsWriter()
    try:
        for name, value in metrics.items():
            writer.log_metric(metric_name=name, value=value)
    finally:
        writer.close()
