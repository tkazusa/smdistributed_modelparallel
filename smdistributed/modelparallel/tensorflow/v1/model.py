# First Party
from smdistributed.modelparallel.tensorflow import state
from smdistributed.modelparallel.tensorflow.compile import CompileStatus


def distributed_model():
    """
    A decorator that specifies the scope over which graph partitioning will occur. Any operation
    outside the smp.DistributedModel scope will be available in all ranks, while any operation
    inside is subject to partitioning across devices.
    """

    def _model(model_func):
        def wrapper(*args, **kwargs):
            if state.serialized_graph.is_profiling:
                return model_func(*args, **kwargs)
            else:
                if state.serialized_graph.should_aggregate():
                    state.serialized_graph.aggregate_profile_results()
                    state.serialized_graph.has_aggregated = True

            # Broadcast the profiling result once
            state.serialized_graph.broadcast_profile_result()

            if state.compile_status == CompileStatus.STEP_COMPILE:
                with state.serialized_graph.track_graph():
                    outputs = model_func(*args, **kwargs)
                state.serialized_graph.finalize(outputs)
                state.partitioner.partition()
                state.serialized_graph.has_partitioned = True

            return state.serialized_graph.import_graph()

        return wrapper

    return _model
