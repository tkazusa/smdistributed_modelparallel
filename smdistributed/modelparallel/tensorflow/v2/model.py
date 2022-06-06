# Standard Library
import inspect
import os
import shutil
import time
from contextlib import contextmanager

# Third Party
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as keras_backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential as KerasSequential
from tensorflow.python.keras.utils import generic_utils as keras_generic_utils

# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup, RankType
from smdistributed.modelparallel.tensorflow import core
from smdistributed.modelparallel.tensorflow.compile import CompileStatus
from smdistributed.modelparallel.tensorflow.graph_utils import SMPCpuContext
from smdistributed.modelparallel.tensorflow.state_mod import state
from smdistributed.modelparallel.tensorflow.utils import (
    _PROFILE_MODEL_PREFIX,
    _TRACE_MODEL_PREFIX,
    GraphBuildError,
    _SpecArg,
    assert_compatible,
    convert_tensor_to_spec,
    create_and_write_to_file,
    is_tensor_or_var,
    receive_checkpoint_files,
    send_checkpoint_files,
    smdebug_name_to_layer_name,
)
from smdistributed.modelparallel.tensorflow.v2.step_mod import step
from smdistributed.modelparallel.tensorflow.v2.utils import get_layers

try:
    import smdebug.tensorflow as smd
    from smdebug.trials import create_trial
except ImportError as error:
    smd = None


class DistributedModel(Model):
    """ A class that defines a distributed model. Model partitioning will occur within the scope
    of the __call__ function. """

    _constructing_duplicate = False

    def __init__(self, *args, **kwargs):
        # 1. handle DistributedModel(inputs=, outputs=) API
        # 2. handle functional API

        super(DistributedModel, self).__init__(*args, **kwargs)
        self.dummy_layers = [SMPDummy(i) for i in range(state.cfg.pipeline_parallel_degree)]

        self.call_fn = tf.function(self._internal_call)

        # create duplicate model object at the CPU
        if core.pp_rank() == 0 and not DistributedModel._constructing_duplicate:
            state._tracing_model = DistributedModel._duplicate_model(self, _TRACE_MODEL_PREFIX)
            if smd is not None:
                state._profile_model = DistributedModel._duplicate_model(
                    self, _PROFILE_MODEL_PREFIX
                )
                state._profile_model.train_step = super(
                    DistributedModel, state._profile_model
                ).train_step
            order = 0
            for cpu_layer in get_layers(state._tracing_model):
                state.layer_order_map[id(cpu_layer)] = order
                order += 1

    @staticmethod
    def _duplicate_model(model_obj, prefix):
        """
        Create a duplicate model object. This model will be kept at CPU, and it will be used for
        partitioning.

        We duplicate the model by extracting the arguments model_obj.__init__ was called with, and
        then calling the constructor of the given model object. We take this approach because
        keras.Model objects are not deep-copyable, and Keras clone_model function does not work
        for Keras models defined through the sub-classing API.
        """

        args, kwargs = DistributedModel._get_init_args(model_obj)

        args = tuple(arg for arg in args if arg is not model_obj)

        # duplicate the model in _duplicate_context to avoid infinite recursion
        with DistributedModel._duplicate_context(prefix):
            return model_obj.__class__(*args, **kwargs)

    @staticmethod
    def _get_init_args(model_obj):
        """ Determine the argument values of the __init__ call of the sub-class. """

        argspec = inspect.getfullargspec(model_obj.__class__.__init__)
        frame = inspect.currentframe()

        # we are going 3 levels out to get to the __init__ call of the sub-class,
        # and extract the parameters it was called with
        outer_frame = inspect.getouterframes(frame)[3][0]
        argvalues = inspect.getargvalues(outer_frame)

        # construct args and kwargs
        args = []
        kwargs = {}

        # the order matters here
        for arg in argspec.args:
            args.append(argvalues.locals[arg])

        if argspec.varargs is not None:
            args.extend(list(argvalues.locals[argspec.varargs]))

        for kwarg in argspec.kwonlyargs:
            kwargs[kwarg] = argvalues.locals[kwarg]

        if argspec.varkw is not None:
            kwargs.update(argvalues.locals[argspec.varkw])

        return args, kwargs

    @contextmanager
    def patch_layer_names(prefix):
        """
        Override the layer name assignment behavior of Keras while constructing the CPU model.
        This is used to keep the layer/op names across the GPU and CPU models in sync, i.e.,
        [GPU layer/op name] = prefix + "_" + [corresponding CPU layer/op name]
        """
        org_init_set_name = Layer._init_set_name

        def _trace_init_set_name(self, name, zero_based=True):
            if name:
                self._name = prefix + "_" + name
            else:
                self._name = keras_backend.unique_object_name(
                    keras_generic_utils.to_snake_case(prefix + "_" + self.__class__.__name__),
                    zero_based=zero_based,
                )

        Layer._init_set_name = _trace_init_set_name
        yield
        Layer._init_set_name = org_init_set_name

    @staticmethod
    @contextmanager
    def _duplicate_context(layer_prefix):
        DistributedModel._constructing_duplicate = True
        with DistributedModel.patch_layer_names(layer_prefix):
            yield
        DistributedModel._constructing_duplicate = False

    def _internal_call(self, microbatch, compiling_step, compiling_model, *args, **kwargs):
        if not compiling_model:
            return state.serialized_graph.import_graph(self, args, kwargs, microbatch)
        else:
            with state.serialized_graph.track_graph(self, args, kwargs):
                return super(DistributedModel, self).__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """ Call the model. If tracing, will create a SerializedGraph representing the
        body of the model call, and then exit. If tracing is already done, it will import the
        partitioned SerializedGraph.

        The body of the __call__ function is executed as a tf.function, since this makes serialization and
        graph importing much simpler.
        """

        if state.serialized_graph.is_profiling:
            return super(DistributedModel, self).__call__(*args, **kwargs)
        else:
            if state.serialized_graph.should_aggregate():
                state.serialized_graph.aggregate_profile_results()
                state.serialized_graph.has_aggregated = True

        # Broadcast the profiling result once
        state.serialized_graph.broadcast_profile_result()

        # Identifying first step, this is done for restore, which should be executed
        # after first step
        state.first_step_done = True

        if state.is_saving:
            # when creating a saved model, TF does runs "signature related validation" checks
            # which traces model call function.

            return

        if not state.inside_smp_step:
            # we raise an error because we need to register state with the backend to be able to execute
            # the model, which is done at smp.step. This is not a fundamental restriction and can be relaxed later.
            raise GraphBuildError(
                "Cannot execute smp.DistributedModel.__call__ outside a smp.step-decorated function."
            )

        # Any time you need to re-trace the model, you have to re-trace twice, because the first
        # will construct the forward ops with correct attributes, and the second will construct the
        # backward ops with correct attributes. To force the second re-trace, we use the
        # step_compiling flag. In the second re-trace, we do not re-partition. The ops are
        # imported from SerializedGraph again.

        state.serialized_graph.has_partitioned = True

        # If the wrapped tf.function does another round of tracing (e.g., for evaluation),
        # the model graph can change, so we need to re-partition. During re-partitioning,
        # ownership of existing model variables will not change, since that would require transfer
        # of weights. The location of specific operations MIGHT change, since these are
        # stateless.
        _has_key = state.partition_cache.has_key(self.call_fn, args, kwargs)

        if not _has_key:
            if core.pp_rank() == 0:
                self._raise_if_layers_not_duplicated()
                self._raise_if_not_microbatch_zero()

                state.tracking_model = True
                spec_args, spec_kwargs = convert_tensor_to_spec(args, kwargs)
                state.spec_args = spec_args
                self.partition(spec_args, spec_kwargs)

                state.comm.broadcast(state.serialized_graph.get_data(), group=CommGroup.PP_GROUP)
                state.tracking_model = False
            else:
                sg_data = state.comm.recv_from(0, RankType.PP_RANK)
                state.serialized_graph.set_data(sg_data)
        else:
            state.partition_cache.load(self.call_fn, args, kwargs)

        step_compiling = state.compile_status == CompileStatus.STEP_COMPILE
        out = self.call_fn(state.microbatch, step_compiling, False, *args, **kwargs)
        if not _has_key:
            state.partition_cache.put(self.call_fn, args, kwargs, state.serialized_graph)

        return out

    def _raise_if_not_microbatch_zero(self):
        """ Only used when partitioning. If we enter the partitioning code for microbatch > 0 it means that the input tensors have different signatures for different microbatches,
        and the GraphDef import will fail due to shape mismatch with model inputs. The only known cause for this is using a batch size that is not divisible by num. microbatches. """
        if state.microbatch > 0:
            raise TypeError(
                "The input tensors to the model have different signatures for different microbatches. This is typically caused by using a microbatch count that is not divisible by the batch size."
            )

    def _raise_if_layers_not_duplicated(self):
        same_layers = []
        for layer, cpu_layer in zip(get_layers(self), get_layers(state._tracing_model)):
            if id(layer) == id(cpu_layer) and not hasattr(layer.build, "_is_default"):
                same_layers.append(layer)
        if len(same_layers) > 0:
            raise ValueError(
                f"Layers that have a build() method implemented must be created in the __init__ methods of an enclosing Layer or a smp.DistributedModel object. The following layers are found to be in violation of this: {[l.name for l in same_layers]}."
            )

    @tf.autograph.experimental.do_not_convert
    def partition(self, spec_args, spec_kwargs):
        # re-trace the model graph based on the new signature
        with SMPCpuContext.cpu_var_creator():
            model_graph = state._tracing_model.call_fn.get_concrete_function(
                0, True, True, *spec_args, **spec_kwargs
            ).graph
            # Check if there is variable that does not belong to smp.DistributedModel
            model_vars = {var.ref() for var in state._tracing_model.weights}
            all_vars = {var.ref() for var in model_graph.variables}
            extra = all_vars.difference(model_vars)
            if len(extra) > 0:
                raise GraphBuildError(
                    f"Any variable that is used in smp.DistributedModel.call function must be owned by the model or one of its sub-layers. Variables in violation: {[var.deref().name for var in extra]}"
                )

        # partition the new graph
        state.serialized_graph.finalize(model_graph)
        state.partitioner.partition()

    def _save_checkpoint(self, ckpt_path, prefix="smp_tf_ckpt"):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt_path_str = os.path.join(ckpt_path, "mp_rank_" + str(core.pp_rank()), prefix)
        ckpt.save(ckpt_path_str)

    def save_model(self, save_path="/opt/ml/model"):
        """
        Args: save_path: path to save the model.

        For creating saved model:
            step 1. save checkpoint on all ranks
            step 2. Load these checkpoints to complete model saved earlier in state._tracing_model on rank 0.
            step 3. create a saved model using tf.saved_model.save
        """

        if core.dp_rank() == 0:

            # step 1. : saving ckpt for all ranks to a temp location
            ckpt_path = "smp_latest_ckpt"
            self._save_checkpoint(ckpt_path)

        # Adding this barrier to make sure, the checkpoints are saved before they are restored.
        # In single instance case, observed cases where rank 0 will save and start restoring before
        # other ranks have finished writing checkpoints.
        # In multi node case, this generally wont be problem as their is send/receive of checkpoints
        # which will syncronize.
        core.barrier()

        if core.dp_rank() == 0:

            # sync checkpoinst across nodes if needed
            if state.sync_checkpoints:
                # For Handling multi-node use case.
                #  if current rank is outside node 0.
                #  Send checkpoints from current to rank 0.

                all_ranks = core.get_pp_group()

                # send from ranks that are not in rank 0.
                if not core.is_in_same_instance(0):
                    send_checkpoint_files(ckpt_path, 0, rank_type=RankType.WORLD_RANK)

            if core.pp_rank() == 0:
                model = state._tracing_model
                checkpoint = tf.train.Checkpoint(model=model)

                # receving checkpoints if needed.
                if state.sync_checkpoints:
                    all_ranks_instance = [core.is_in_same_instance(rank) for rank in all_ranks]

                    # if all the ranks_instance is not same, mp is across nodes.
                    if all_ranks_instance.count(all_ranks_instance[0]) != len(all_ranks_instance):
                        # Step 2: Fetch all checkpoints from ranks on other nodes.
                        for rank in all_ranks:
                            # receive from ranks not in the same instance.
                            if not core.is_in_same_instance(rank):
                                receive_checkpoint_files(
                                    ckpt_path, rank, rank_type=RankType.WORLD_RANK
                                )

                # Step 2: restoring checkpoints for all ranks to complete model
                for idx in range(core.pp_size()):
                    path = os.path.join(ckpt_path, "mp_rank_" + str(idx))
                    checkpoint.restore(tf.train.latest_checkpoint(path)).expect_partial()

                # Step 3: saving the model as saved model.
                func = tf.function(model.call)
                spec_args = state.spec_args

                state.is_saving = True
                tf.saved_model.save(
                    model, save_path, signatures=func.get_concrete_function(*(spec_args))
                )
                state.is_saving = False

                # Cleaning up
                if os.path.exists(ckpt_path):
                    shutil.rmtree(ckpt_path)

        # Adding a barrier to make sure we have consumed the checkpoints before deleting
        core.barrier()

        if core.dp_rank() == 0 and core.pp_rank() == 0:
            # Cleanup: removing temp ckpt
            if os.path.exists(ckpt_path):
                shutil.rmtree(ckpt_path)

    @step
    def get_grads(self, x, y, sample_weight):
        """The logic for SMP training step.
        This method is called by `DistributedModel.train_step()`
        """
        y_pred = self(x, training=True)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        gradients = self.optimizer.get_gradients(loss, self.trainable_variables)

        return gradients, loss, y_pred

    @tf.function
    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        This method is overriding the default keras.Model.train_step().

        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Arguments:
        data: A nested structure of `Tensor`s.

        Returns:
        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned. Example:
        `{'loss': 0.2, 'accuracy': 0.7}`.
        """

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        gradients, loss, y_pred = self.get_grads(x, y, sample_weight)

        if state.cfg.horovod:
            import horovod.tensorflow as hvd

            gradients = [hvd.allreduce(g.accumulate()) for g in gradients]
        else:
            gradients = [g.accumulate() for g in gradients]

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """The logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        This method is overriding the default keras.Model.test_step().

        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.

        Arguments:
        data: A nested structure of `Tensor`s.

        Returns:
        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned.
        """

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        loss, y_pred = self.get_eval(x, y, sample_weight)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    @step
    def get_eval(self, x, y, sample_weight):
        """The logic for SMP evaluation step.
        This method is called by `DistributedModel.test_step()`
        """

        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        return loss, y_pred

    def fit(self, *args, **kwargs):
        # these args require keras.utils.Sequence which is not supported by SMP
        unsupported_args = ["workers", "max_queue_size", "use_multiprocessing"]
        for arg in unsupported_args:
            if arg in kwargs:
                raise TypeError(
                    f"'%s' argument is not currently supported by SMP keras model fit" % arg
                )
        super(DistributedModel, self).fit(*args, **kwargs)

    def _is_forward_tensor(self, name):
        # smdebug forward tensor naming rules: *layername/inputs(outputs)
        return (
            (name.endswith("inputs") or name.endswith("outputs"))
            and "gradients" not in name
            and "weights" not in name
        )

    def profile(self, *args, path=None, aggregation="median", batch_size=None, force_cpu=True):
        """
        Profile the model and get tensor shapes using smdebugger. Smdebugger will record tensor shapes
        for inputs and outputs of each keras.layer. This profile function may run many steps and the
        results will be aggregrated using the aggregation method provided by the user.

        Args:
            path: str, path to save the profiling result.
            aggregation:str, method to aggregate result from multiple runs. Possible values: median, mean, max, p95
            batch_size: int, the input batch size
            force_cpu: bool, wthether to force to everything on CPU. Set to False to only put variables on CPU
        """
        if batch_size == None:
            for arg in tf.nest.flatten(args):
                if isinstance(arg, tf.Tensor) or isinstance(arg, np.ndarray):
                    batch_size = arg.shape[0]

        with state.serialized_graph.profiling_context(smd=smd) as should_profile:
            if should_profile:
                if path == None:
                    stamp = str(int(time.time()))
                    path = f"/tmp/smdebug_outputs_{stamp}"  # _{core.dp_rank()}"

                # Create the profiling hook and run model on CPU
                hook = smd.KerasHook(
                    path,
                    save_all=True,
                    save_config=smd.SaveConfig(save_steps=[0], save_interval=1),
                    reduction_config=smd.ReductionConfig(save_shape=True),
                )

                _cpu_context = (
                    SMPCpuContext.cpu_context if force_cpu else SMPCpuContext.cpu_var_creator
                )
                with _cpu_context():

                    hook.register_model(state._profile_model)

                    class DummyLoss(tf.keras.losses.Loss):
                        def call(self, y_true, y_pred):
                            return tf.reduce_mean(y_pred)

                    _dummy_loss = DummyLoss()

                    state._profile_model.compile(
                        optimizer="Adam", loss=_dummy_loss, run_eagerly=True
                    )
                    state._profile_model.fit(
                        *args,
                        tf.constant(1.0, shape=(batch_size, 1)),
                        epochs=1,
                        steps_per_epoch=1,
                        callbacks=[hook],
                    )

                    # Fetch the profiling result
                    trial = create_trial(path=path, name="training_run")
                    tensor_names = trial.tensor_names()
                    for name in tensor_names:
                        if not self._is_forward_tensor(name):
                            continue
                        shape = trial.tensor(name).shape(step_num=0)
                        # Convert shape to a list of tuple for post-processing
                        shape = (
                            list(shape)
                            if (len(shape) > 0 and isinstance(shape[0], tuple))
                            else [shape]
                        )

                        layer_name, tensor_type = smdebug_name_to_layer_name(name)
                        if layer_name not in state.serialized_graph.raw_profiling_results:
                            state.serialized_graph.raw_profiling_results[layer_name] = {
                                "inputs": [],
                                "outputs": [],
                            }
                        state.serialized_graph.raw_profiling_results[layer_name][
                            tensor_type
                        ].append(shape)
                        state.serialized_graph.aggregation_method = aggregation

        core.barrier()


class SMPDummy(Layer):
    """ A Layer that simply defines and returns a dummy trainable variable."""

    def __init__(self, sg):
        self.sg = sg
        super(SMPDummy, self).__init__()

    def build(self, input_shape):
        self.var = self.add_weight(
            name="SMPDummy_%s" % self.sg, shape=(), initializer="zeros", trainable=True
        )
        super(SMPDummy, self).build(input_shape)

    def call(self, x):
        return self.var


class Sequential(DistributedModel, KerasSequential):
    pass
    # 5. handle Sequential API
