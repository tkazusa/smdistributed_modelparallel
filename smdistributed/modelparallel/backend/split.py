# Standard Library
import inspect
from abc import ABCMeta, abstractmethod
from collections import defaultdict

# First Party
from smdistributed.modelparallel.backend.logger import get_logger

NATIVE_TYPES = [str, int, float, bytes, bytearray, bool, type(None)]


class TensorSplitter(metaclass=ABCMeta):
    """ A class that implements framework-independent tensor splitting API. Frameworks
    implement three methods: `get_tensor_type`, `map_structure`, `slice`. The logic
    to handle nested structures, skip non-split inputs, and custom splitting axes is
    implemented here. """

    def __init__(
        self,
        func,
        non_split_inputs=None,
        input_split_axes=None,
        types_to_warn=None,
        types_to_suppress_warn=None,
    ):
        if non_split_inputs is not None and not isinstance(non_split_inputs, list):
            raise TypeError("Non-split inputs must be a list.")

        if input_split_axes is not None and not isinstance(input_split_axes, dict):
            raise TypeError("input_split_axes must be a dict.")

        self.func = func
        self.non_split_inputs = non_split_inputs if non_split_inputs is not None else []
        self.input_split_axes = input_split_axes if input_split_axes is not None else {}
        self.types_to_warn = types_to_warn or []
        self.types_to_suppress_warn = types_to_suppress_warn or []
        self.types_to_suppress_warn = self.types_to_suppress_warn + NATIVE_TYPES

        self.warned_non_splittable = defaultdict(lambda: False)

    @abstractmethod
    def get_tensor_type(self):
        """ Return the tensor type in the framework, e.g., tf.Tensor, torch.Tensor """
        return

    @abstractmethod
    def map_structure(self, func, structure):
        """ Return a function that applies a callable to all objects in a nested structure, e.g., tf.nest.map_structure """
        return

    @abstractmethod
    def slice(self, tensor, num_mb, mb, axis=0):
        """ Slice the given tensor into num_mb pieces along the given axis, and return the mb'th piece. """
        return

    def preprocess_args_all_mbs(self, args, kwargs, num_mb):
        """ Returns a list of tensor-sliced (arg, kwarg) pairs, indexed by microbatch."""
        all_args, all_kwargs = [], []
        for mb in range(num_mb):
            mb_args, mb_kwargs = self.preprocess_args(args, kwargs, num_mb, mb)
            all_args.append(mb_args)
            all_kwargs.append(mb_kwargs)
        return all_args, all_kwargs

    def preprocess_args(self, args, kwargs, num_mb, mb):
        """Slice the Tensor arguments, unless the argument is listed in non_split_inputs, for microbatch mb.

        Arguments:
            args: A list of args provided when func was called.
            kwargs: A dict of keyword args provided when func was called.
            num_mb: Number of microbatches
            mb: Current microbatch to do the slicing for

        Returns:
            A tuple `mb_args, mb_kwargs`. mb_args and mb_kwargs contain the sliced tensors.
        """

        arg_names = inspect.getfullargspec(self.func).args

        if len(set(self.non_split_inputs).difference(set(arg_names))) > 0:
            raise ValueError(
                "Non-split inputs list %s contains a non-argument of %s."
                % (self.non_split_inputs, self.func.__name__)
            )

        if len(set(self.input_split_axes.keys()).difference(set(arg_names))) > 0:
            raise ValueError(
                "Input split axes dict %s contains a non-argument key of %s."
                % (self.input_split_axes, self.func.__name__)
            )

        mb_args = self._split_tensors_in_list(
            args, arg_names, self.non_split_inputs, self.input_split_axes, num_mb, mb
        )
        mb_kwargs = self._split_tensors_in_dict(
            kwargs, self.non_split_inputs, self.input_split_axes, num_mb, mb
        )

        return mb_args, mb_kwargs

    def _split_tensors_in_list(self, args, arg_names, keys_to_skip, split_axis_for_key, num_mb, mb):
        """Split each tensor in the list and unpack StepOutputs. Any non-tensor, non-StepOutput value
        in args is unmodified.

        This relies on `_split_tensors_in_dict`.

        Arguments:
            args: A list of argument values.
            arg_names: A list of corresponding argument names.
            keys_to_skip: A list of argument names whose argument value should not be split.
            split_axis_for_key: A dict of argument names and split axis.

        Returns:
            A list of length `num_mb`. Each entry is a list similiar to the input `args`, but containing the split tensors
                or unpacked StepOutputs.
        """

        args_as_dict = dict(enumerate(args))
        indices_to_skip = [i for i, name in enumerate(arg_names) if name in keys_to_skip]
        split_axis_for_index = {
            arg_names.index(arg_name): axis for arg_name, axis in split_axis_for_key.items()
        }

        split_dict = self._split_tensors_in_dict(
            args_as_dict, indices_to_skip, split_axis_for_index, num_mb, mb
        )
        return tuple(v for _, v in sorted(split_dict.items()))

    def _split_tensors_in_dict(self, tensors, keys_to_skip, split_axes, num_mb, mb):
        """Split each tensor in the dict values and unpack StepOutputs. Any non-tensor, non-StepOutput value in
        tensors is unmodified.

        Arguments:
            tensors: A dict of (argument name: argument value).
            keys_to_skip: A list of keys to not split.
            split_axes: A dict of keys containing which axis to split the corresponding tensor on.

        Returns:
            A list of length `num_mb`. Each entry is a dict similiar to the input `tensors`, but containing the split tensors
                or unpacked StepOutputs.
        """

        return {
            k: self.map_structure(
                lambda x: self._split_util(
                    x, num_mb, num_mb, mb, k not in keys_to_skip, split_axes.get(k, 0)
                ),
                v,
            )
            for k, v in tensors.items()
        }

    def _split_util(self, value, num_splits, num_mb, mb, is_splittable_key, axis=0):
        """Splits the key only if it is splittable. Replaces the StepOutput with value.outputs irrespective of the fact
        that key is splittable or not"""
        if hasattr(value, "smp_slice"):
            return value.smp_slice(num_mb, mb, axis)
        if isinstance(value, StepOutput):
            return value.outputs[mb]
        if not is_splittable_key:
            return value
        if isinstance(value, self.get_tensor_type()):
            return self.slice(value, num_mb, mb, axis)
        if not self.warned_non_splittable[type(value).__name__]:
            if not callable(value) and not isinstance(value, tuple(self.types_to_suppress_warn)):
                get_logger().warn(
                    f"Non-splittable object of type {type(value)} passed to smp.step. If this object contains tensors that need to be split across microbatches, implement a 'smp_slice' method for this class. See SMP documentation for further information."
                )
            if isinstance(value, tuple(self.types_to_warn)):
                get_logger().warn(
                    f"Object of type {type(value)} passed to smp.step. In normal use of SMP, this object should be used outside of smp.step. This might *potentially* be a bug."
                )
            self.warned_non_splittable[type(value).__name__] = True
        return value


class StepOutput(metaclass=ABCMeta):
    def __init__(self, outputs):
        if not isinstance(outputs, list) and not isinstance(outputs, tuple):
            raise ValueError(f"StepOutput only accepts list or tuple, but get {type(outputs)}")
        self._outputs = outputs

    def __getitem__(self, index):
        return self.outputs[index]

    def clear(self):
        self._outputs = None

    @property
    def outputs(self):
        return self._outputs

    @abstractmethod
    def reduce_mean(self):
        pass

    @abstractmethod
    def reduce_sum(self):
        pass

    @abstractmethod
    def concat(self):
        pass

    @abstractmethod
    def stack(self):
        pass

    def add_output(self, out):
        self._outputs.append(out)

    def process_outputs(self, func):
        return func(self.outputs)

    def map(self, func):
        """
        Applies a callable func to each microbatch output individually
        """

        def _apply(outputs):
            return [func(out) for out in outputs]

        # can only be a list because of check in constructor
        self._outputs = _apply(self._outputs)
        return self
