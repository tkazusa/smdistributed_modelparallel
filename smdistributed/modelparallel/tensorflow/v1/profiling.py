# Standard Library
import time

# Third Party
import tensorflow as tf

# First Party
from smdistributed.modelparallel.tensorflow import core, state
from smdistributed.modelparallel.tensorflow.graph_utils import SMPCpuContext
from smdistributed.modelparallel.tensorflow.utils import is_tensor_or_op, is_tensor_or_var
from smdistributed.modelparallel.tensorflow.v1.serialization import TraceGraph

try:
    import smdebug.tensorflow as smd
    from smdebug.trials import create_trial
except ImportError as error:
    smd = None


def profile(
    model_func, *args, placeholders=None, path=None, aggregation="median", force_cpu=True, kwargs={}
):
    """
    Args:
        model_func: the model function which is decorated with smp.distributedmodel()
        args: The input data.
        placeholders: If user is using placeholder and feed_dict to train, provide a list of placeholder here corresponding to each input.
                      If there are python objects put a None at that index.
        aggregation_method: method to aggregate result from multiple runs. Possible values: median, mean, max, p95
        force_cpu: bool, wthether to force to everything on CPU. Set to False to only put variables on CPU
        kwargs: a dict of key arguments that passed into model function, must only be python objects
    """

    g = TraceGraph()
    with state.serialized_graph.profiling_context(smd=smd, graph=g) as should_profile:
        if should_profile:

            _cpu_context = SMPCpuContext.cpu_context if force_cpu else SMPCpuContext.cpu_var_creator
            with _cpu_context():

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.gpu_options.visible_device_list = str(core.local_rank())

                real_args, phs_to_data = _preprocess_args(
                    *args, _placeholders=placeholders, config=config
                )

                with g.trace():

                    stamp = str(int(time.time()))
                    path = f"/smdebug_outputs_{stamp}"
                    hook = smd.SessionHook(
                        path,
                        save_all=True,
                        save_config=smd.SaveConfig(save_steps=[0], save_interval=1),
                        reduction_config=smd.ReductionConfig(save_shape=True),
                    )

                    out = model_func(*real_args, **kwargs)
                    if isinstance(out, tuple):
                        out = tf.nest.flatten(out)
                        out = [item for item in out if is_tensor_or_op(item)]

                    _feed_dict = _create_feed_dict(phs_to_data, g)

                    with tf.train.MonitoredTrainingSession(hooks=[hook], config=config) as mon_sess:
                        out = mon_sess.run(out, feed_dict=_feed_dict)

                        trial = create_trial(path=path, name="training_run")
                        tensor_names = trial.tensor_names()
                        for name in tensor_names:
                            shape = trial.tensor(name).shape(step_num=0)
                            if name not in state.serialized_graph.raw_profiling_results:
                                state.serialized_graph.raw_profiling_results[name] = [shape]
                            else:
                                state.serialized_graph.raw_profiling_results[name].append(shape)
                        state.serialized_graph.aggregation_method = aggregation

    core.barrier()


def _create_feed_dict(input_data_mapping, graph):
    """Create the feed_dict for the profiling session run"""

    def _val_to_key(dict, val):
        for key, value in dict.items():
            if value == val:
                return key

    _feed_dict = {}
    for ph_name, ts_name in graph.step_input_map.items():
        if ts_name not in input_data_mapping:
            raise ValueError(
                f"Step input tensor {ts_name} is not in input_data_mapping, which contains {input_data_mapping.keys()}"
            )
        _feed_dict[ph_name] = input_data_mapping[ts_name]
    return _feed_dict


def _preprocess_args(*args, _placeholders=None, config=None):
    """
    Preprocess the input positional arguments to model function
    - If using placeholder, the placeholder will be used as arguments
    - If using tf.Data.dataset, first run the tensor to get the numpy value,
      then use the numpy value to feed the placeholder created in TraceGraph
    """
    input_data_mapping = {}

    if _placeholders == None:
        input_data = []

        def _add_tensor(item):
            if is_tensor_or_var(item):
                input_data.append(item)
            return item

        for index, arg in enumerate(args):
            if isinstance(arg, list) or isinstance(arg, tuple):
                tf.nest.map_structure(_add_tensor, arg)
            else:
                _add_tensor(arg)

        with tf.train.MonitoredTrainingSession(config=config) as mon_sess:
            input_data_py = mon_sess.run(input_data)

        for index, item in enumerate(input_data):
            input_data_mapping[item.name] = input_data_py[index]

        return args, input_data_mapping
    else:
        real_args = []
        if len(_placeholders) != len(args):
            raise ValueError("placeholders should have same number of items as input args")

        for index, item in enumerate(_placeholders):
            if item is not None:
                real_args.append(item)
                input_data_mapping[item.name] = args[index]
            else:
                real_args.append(args[index])
        return real_args, input_data_mapping
