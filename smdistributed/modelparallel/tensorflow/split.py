# Standard Library

# Third Party
import tensorflow as tf
from tensorflow.python.eager import context

# First Party
from smdistributed.modelparallel import tensorflow as smp
from smdistributed.modelparallel.backend.split import StepOutput as BaseStepOutput
from smdistributed.modelparallel.backend.split import TensorSplitter
from smdistributed.modelparallel.tensorflow.state_mod import state
from smdistributed.modelparallel.tensorflow.utils import is_tensor_or_var, make_tf_compatible_name

_LAST_SPLIT_OP_NAME_PREFIX = "SMPLastSplitOp_"
_LAST_SPLIT_OP_TYPE = "StridedSlice"


class TFTensorSplitter(TensorSplitter):
    def get_tensor_type(self):
        return tf.Tensor

    def map_structure(self, func, structure):
        return tf.nest.map_structure(func, structure)

    def slice(self, tensor, num_mb, mb, axis=0):
        """ Split a batch input tensor into num_mb tensors along the specified axis.
        Split tensors will be as uniform in size as possible.

        The last op in the split operation will be labelled so that `is_last_split_op(op)` returns True.

        Arguments:
            tensor: The tensor to split. The static shape need not be known.
            axis: The dimension on which to split the tensor.

        Returns:
            A list of Tensors which are split on the specified axis.
        """

        shape = tf.shape(tensor)

        # assert that the batch dimension is evenly divisible by the number of microbatches
        is_divisible = tf.math.equal(
            tf.cast(tf.math.floormod(shape[axis], num_mb), dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
        )

        strides = [num_mb if index == axis else 1 for index in range(tensor.shape.ndims)]
        begin_at_offset = [mb if index == axis else 0 for index in range(tensor.shape.ndims)]

        # this assertion sometimes does not work, but if the non-divisible tensor is being used in the model,
        # model will complain afterwards anyway. if it is not used in the model, it is not *really* a problem
        # that it is not divisible.
        tf.debugging.Assert(
            is_divisible,
            [
                "The batch dimension must be evenly divisible by the number of microbatches. Found batch size: ",
                shape[axis],
            ],
        )

        split_tensor = tf.strided_slice(
            tensor,
            begin=begin_at_offset,
            end=shape,
            strides=strides,
            name=get_last_split_op_name(tensor),
        )

        # If the unsplit size is known at graph construction time, calculate the size of the split tensors.
        batch_split_dim = tensor.shape.as_list()[axis]
        split_size = (
            (int(mb < batch_split_dim % num_mb) + batch_split_dim // num_mb)
            if batch_split_dim is not None
            else None
        )

        # Assign normative split shape so that consumers of the microbatch tensor have static shapes when possible.
        normative_split_shape = [
            dim if dim_index != axis else split_size for dim_index, dim in enumerate(tensor.shape)
        ]
        split_tensor.set_shape(normative_split_shape)

        return split_tensor


def postprocess_outputs(structure_list):
    """
    A function that postprocesses a list of Python structures, where the list is indexed
    by microbatch. In the structure, replaces every occurence of a tf.Tensor or tf.Variable
    with a StepOutput object which wraps the same tensor for all microbatches.
    tf.Operations across difference microbatches are grouped into a single operation.
    """
    num_mb = len(structure_list)
    if num_mb == 0:
        return

    if isinstance(structure_list[0], list):
        return [
            postprocess_outputs([structure_list[mb][i] for mb in range(num_mb)])
            for i in range(len(structure_list[0]))
        ]
    elif isinstance(structure_list[0], tuple):
        return tuple(
            postprocess_outputs([structure_list[mb][i] for mb in range(num_mb)])
            for i in range(len(structure_list[0]))
        )
    elif isinstance(structure_list[0], dict):
        return {
            k: postprocess_outputs([structure_list[mb][k] for mb in range(num_mb)])
            for k, v in structure_list[0].items()
        }
    # (TODO) Have a better handle of None gradients
    elif is_tensor_or_var(structure_list[0]):
        return StepOutput(structure_list)
    elif isinstance(structure_list[0], tf.Operation):
        return tf.group(*structure_list)
    else:
        return structure_list


def get_last_split_op_name(tensor):
    out = _LAST_SPLIT_OP_NAME_PREFIX + make_tf_compatible_name(tensor.name)
    return out


def is_last_split_op(op):
    return _LAST_SPLIT_OP_NAME_PREFIX in op.name and op.type == _LAST_SPLIT_OP_TYPE


class StepOutput(BaseStepOutput):
    """
    Represents a tensor output from smp.step. For a given output tensor from smp.step,
    StepOutput acts as a thin wrapper around the different versions of this tensor
    across microbatches. The actual list of tensor outputs, indexed by microbatch id,
    can be accessed through StepOutput.outputs.

    StepOutput exposes an API that covers commonly used post-processing operations
    on output tensors, such as averaging (reduce_mean()), merging (merge()), and
    accumulation (accumulate()).

    For typical TF1.x use case of returning grads_and_vars frorm smp.step, also see
    smp.accumulate, which is another wrapper on top of StepOutput.accumulate() intended
    for use with grads_and_vars.
    """

    def div(self, x):
        return tf.truediv(x, tf.cast(state.num_microbatches(), dtype=x.dtype))

    def reduce_mean(self):
        """
        Returns a tf.Tensor that is the average of the tensors in StepOutput.outputs.
        """

        def func(output):
            return tf.add_n([self.div(x) for x in output])

        return self.process_outputs(func)

    def reduce_sum(self):
        return self.process_outputs(tf.add_n)

    def concat(self):
        return self.process_outputs(lambda x: tf.concat(x, axis=0))

    def stack(self):
        return self.process_outputs(tf.stack)

    def merge(self, axis=0):
        """
        Concatenates the output tensors along the given axis. Does the inverse
        operation of `tf.strided_slice` used internally by smp.step for splitting the
        input tensors along the batch dimension. Should be used by default for output
        tensors that needs to be combined along the batch axis.
        """

        shapes = []
        for output in self._outputs:
            shape = output.shape.as_list()
            if not shape or axis >= len(shape):
                raise ValueError(
                    f"Cannot merge scalars or tensors whose specified axis does not exist. "
                )
            shapes.append(tf.shape(output)[axis])
        bs = tf.add_n(shapes)
        ind = [tf.range(mb, bs, smp.num_microbatches()) for mb in range(len(self._outputs))]
        return tf.dynamic_stitch(ind, self._outputs)

    def accumulate(self, method="variable", var=None):
        """
        Averages the input tensors, just like StepOutput.reduce_mean(). However, unlike
        StepOutput.reduce_mean(), does not wait for all tensors to be available before
        accumulating, thereby saving memory. Should be used as the default option when
        averaging large tensors such as gradients.

        For accumulating gradients across microbatches in TF1.x, also see smp.accumulate().

        Method is required to be one of 'variable', 'add_n', or 'accumulate_n'. Default: 'variable'.
        """

        # TODO(Fei): Currently there are three methods to accumulate gradients
        # tf.math.add_n, tf.math.accumulate_n and smp_accum. For Bert smp_accum
        # use least memory and have best performance. We need to investigate the reason keep one method that works for all cases.
        if self.outputs[0] is None:
            return None
        if method == "add_n":
            return tf.math.add_n([self.div(x) for x in self.outputs])
        elif method == "accumulate_n":
            return tf.math.accumulate_n([self.div(x) for x in self.outputs])
        elif method == "variable":
            return self._smp_accum(var)
        else:
            raise ValueError(
                f"accumulation method must be one of 'add_n', 'accumulate_n', or 'variable'."
            )

    def _smp_accum(self, var):
        if context.executing_eagerly():
            return self.reduce_mean()
        else:
            if var is not None and not isinstance(var, tf.Variable):
                raise ValueError(
                    f"A tf.Variable or None must be provided if variable is selected as the accumulate method, but {type(var)} is provided"
                )

            # Typical use case in TF1.x is to call
            # smp.accumulate(grads_and_vars), where grads_and_vars is the
            # output of optimizer.compute_gradients(). In this case `var`
            # is not None.

            # Typical use case in TF2.x is to return `gradients` from smp.step,
            # and call `[g.accumulate() for g in gradients]` to combine gradients
            # across microbatches. In this case `var` will be None, so we use the
            # attributes of the gradient itself to get or create the accumulation
            # variable.

            if state.num_microbatches() == 1:
                return self.outputs[0]

            tensor = self.outputs[0] if var is None else var
            if isinstance(tensor, (tf.Tensor, tf.Variable)):
                if None in tensor.shape.as_list():
                    # if the gradient shape has None, fall back to tf.accumulate_n
                    # because we cannot create a tf.Variable with unknown shape
                    return self.accumulate(method="accumulate_n")

                if state.tf_version == 1:
                    accum_vars = tf.get_variable(
                        name=tensor.name.split(":")[0] + "/accum",
                        shape=tensor.shape.as_list(),
                        dtype=tf.float32,
                        trainable=False,
                        initializer=tf.zeros_initializer(),
                    )
                else:
                    # TF2.x does not have tf.get_variable
                    accum_vars = state.accum_vars.get_variable(tensor)

                accum_grads = None
                for i, grad in enumerate(self.outputs):
                    if i == 0:
                        accum_grads = accum_vars.assign(self.div(grad))
                    else:
                        with tf.control_dependencies([accum_grads]):
                            accum_grads = accum_vars.assign_add(self.div(grad))
                return accum_grads
            elif isinstance(tensor, tf.IndexedSlices):
                indices = tf.concat([out.indices for out in self.outputs], axis=0)
                values = tf.concat([out.values for out in self.outputs], axis=0)
                dense_shape = tensor.dense_shape
                return tf.IndexedSlices(values, indices, dense_shape)
            else:
                raise TypeError(f"Unsupported type {type(tensor)}.")
