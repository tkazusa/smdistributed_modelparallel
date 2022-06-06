# Standard Library
import types
from contextlib import contextmanager
from functools import wraps

# Third Party
import tensorflow as tf
from tensorflow.python.eager.context import eager_mode
from tensorflow.python.framework import func_graph

# First Party
from smdistributed.modelparallel.tensorflow import core, state
from smdistributed.modelparallel.tensorflow.compile import CompileStatus
from smdistributed.modelparallel.tensorflow.logger import get_logger
from smdistributed.modelparallel.tensorflow.ops import register_state
from smdistributed.modelparallel.tensorflow.pybind import inline_funclib
from smdistributed.modelparallel.tensorflow.split import TFTensorSplitter, postprocess_outputs
from smdistributed.modelparallel.tensorflow.utils import (
    GraphBuildError,
    GraphTraverser,
    _SpecArg,
    assert_compatible,
    cache_key,
    convert_tensor_to_spec,
    get_op,
    mainify_name,
    validate_input_signature,
    xla_scope,
)


def step(
    func=None,
    input_signature=None,
    autograph=True,
    *args,
    non_split_inputs=None,
    input_split_axes=None,
    **kwargs,
):
    """
    A decorator that specifies the pipelined execution context. Typically used to capture the forward/backward
    pass logic in training, but can contain forward pass only (e.g., for evaluation). Extend the functionality
    of @tf.function decorator.

    Splits the tensor inputs along the batch dimension.
    All tf.Tensor inputs will be split, unless the argument name is included
    in `non_split_inputs`. The batch dimension will be assumed to be 0, unless specified otherwise in
    `input_split_axes`. Each resulting microbatch will be executed sequentially, according to the selected
    pipeline type. All returned tensors will be converted to StepOutput objects, even when they are
    inside a nested structure.

    Internally creates a tf.function that represents the execution for a single microbatch, but by adding
    `microbatch` as an additional parameter to the wrapped Python function, creates a separate ConcreteFunction for
    each microbatch, and when called, individually calls each of these ConcreteFunctions on their corresponding
    microbatches.

    This supports all parameters that `tf.function` supports.
    """

    def decorated(inner_func):
        return StepFunction(
            inner_func,
            input_signature,
            autograph,
            non_split_inputs=non_split_inputs,
            input_split_axes=input_split_axes,
            *args,
            **kwargs,
        )

    if func is None:
        """
        If no func is provided, then return function which can be invoked to create the StepFunction.
        This handles cases such as:
        ```
        @smp.function(input_signature=[a, b, c])
        def graph():
            pass
        ```
        """
        return decorated
    else:
        """
        If a func is provided, then we can create the StepFunction immediately. This enables code such as:
        ```
        @smp.function
        def graph():
            pass
        ```
        """
        return decorated(func)


# Decorator to register post partition hooks
def register_post_partition_hook(func):
    def inner(*args, **kwargs):
        state._post_partition_hooks[func] = (args, kwargs)

    return inner


class StepFunction:
    """ A class representing a compilable python function. This class implements __call__ so it can be called like a normal function.
    When it is called, it leverages TensorFlow and the SMP compiler to compile the graph.
    For properties that are not specific to SMP, the class obtains values from an [actual TensorFlow Function](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/def_function.py).
    """

    """ Static field that stores compiled python function. Used to check that only one python function is ever compiled.
    """
    _smp_py_func_compilation_target = None
    function_cache = set()

    @staticmethod
    def raise_if_different_compilation_target(py_func):
        """ Raise exception if the specified python function is different than an existing python function.
        """
        if StepFunction._smp_py_func_compilation_target == None:
            StepFunction._smp_py_func_compilation_target = py_func
            return

        if StepFunction._smp_py_func_compilation_target == py_func:
            return

        original_name = getattr(
            StepFunction._smp_py_func_compilation_target, "__name__", "<no name>"
        )
        new_name = getattr(py_func, "__name__", "<no name>")

        raise GraphBuildError(
            f'Unable to compile python_function "{new_name}": a different python_function "{original_name}" has already been compiled.'
            " SMP does not support compilation of more than one unique python function."
            f" Compiled function: {StepFunction._smp_py_func_compilation_target}. New function: {py_func}"
        )

    def __init__(
        self,
        func,
        input_signature=None,
        autograph=True,
        non_split_inputs=None,
        input_split_axes=None,
        *args,
        **kwargs,
    ):
        """ Initialize the StepFunction. `args` and `kwargs` are passed directly to `tf.function`.
        """
        self.state = state
        self._python_function = func
        self._autograph = autograph
        self._init_args = args
        self._init_kwargs = kwargs
        self._descriptor_cache = {}
        self.non_split_inputs = non_split_inputs
        self.input_split_axes = input_split_axes
        self.input_signature = input_signature

        self._splitter = TFTensorSplitter(func, non_split_inputs, input_split_axes)

        if input_signature is not None:
            validate_input_signature(input_signature)

        # The tf.function used for tracing the smp.step function
        self._trace_tf_function = None

        # The inner tf.function used for training. We insert an additional Python argument
        # for microbatch, and re-use this tf.function for each microbatch. In other words,
        # each microbatch is a ConcreteFunction insideself._call_tf_function.
        self._call_tf_function = None

        # The outer tf.function exposed to the user. All external tf.function-style API calls
        # are forwarded to self._outer_tf_function, and it contains the num_mb copies of
        # self._call_tf_function.
        self._outer_tf_function = None

        self._init_tf_functions()

    def _init_tf_functions(self):
        """ Initializes the tf.functions """
        self._trace_tf_function = tf.function(
            self.__wrap_python_function(self.python_function, True),
            self._input_signature,
            self._autograph,
            *self._init_args,
            **self._init_kwargs,
        )
        self._call_tf_function = tf.function(
            self.__wrap_python_function(self.python_function, False)
        )
        self._outer_tf_function = tf.function(
            self.get_outer_function(),
            self._input_signature,
            self._autograph,
            *self._init_args,
            **self._init_kwargs,
        )

    @property
    def non_split_inputs(self):
        return self._non_split_inputs

    @non_split_inputs.setter
    def non_split_inputs(self, ns_inputs):
        self._non_split_inputs = ns_inputs

    @property
    def input_split_axes(self):
        return self._input_split_axes

    @input_split_axes.setter
    def input_split_axes(self, value):
        self._input_split_axes = value

    def __get__(self, obj, objtype):
        """ Implement the non-data descriptor. This enables StepFunction to implement automatic supplementation of "self" parameter
        where appropriate.
        See https://docs.python.org/3/howto/descriptor.html .
        """
        # If the obj is None, then the StepFunction is being accessed via class or it is a function defined outside of a class.
        # E.g., `ClassName.wrapped_function` or simply `wrapped_function`.
        # In these cases, simply return `self`, which implements `__call__`.
        if obj == None:
            return self

        # If obj is not None, then the function is a property of some instance. We return a new StepFunction in this case which binds the inner python_function
        # to the instance so that the `self` argument is correctly provided at calltime, and so that TF compiles the function without knowing about `self`.
        #
        # The created StepFunction is cached so that multiple accesses (instance.train_step) do not cause retracing.
        # This is a similiar approach to TF https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/eager/def_function.py#L792
        if obj not in self._descriptor_cache:
            bound_python_function = types.MethodType(self.python_function, obj)

            # Note that we do not need to use weakref here because SMP only supports compilation of a single python_function.
            self._descriptor_cache[obj] = StepFunction(
                bound_python_function,
                self.input_signature,
                self._autograph,
                *self._init_args,
                **self._init_kwargs,
            )

        return self._descriptor_cache[obj]

    def raise_if_incompatible_args(self, args, kwargs):
        if self.input_signature is not None:
            if len(kwargs) > 0:
                raise ValueError(
                    "When input_signature is specified, keyword arguments are not allowed."
                )

            for spec, arg in zip(self.input_signature, args):
                tf.nest.map_structure(assert_compatible, _SpecArg(spec, arg))

    def __call__(self, *args, **kwargs):
        """ Implement function interface.
        This enables code like the following to work:
        ```
        f = StepFunction(wrapped_function)
        f() # This invokes wrapped_function, plus additional logic.
        ```
        """
        self.state.inside_smp_step = True
        self.raise_if_incompatible_args(args, kwargs)
        # if cache key does not exist for microbatch 0, then all microbatches need to be traced / called.
        key = cache_key(self._trace_tf_function, args, kwargs, self.input_signature)
        if key not in StepFunction.function_cache:
            self._trace(args, kwargs)
            # recompute cache key after tracing
            key = cache_key(self._trace_tf_function, args, kwargs, self.input_signature)
            if key:
                StepFunction.function_cache.add(key)

        # Model has been traced and graph created. Call the post partition hooks here
        # The hook operations are one-time function calls that happen post-partion and before first step (for eg. loading checkpoints)
        # We call the hooks within eager_mode context to ensure that the operations do not get baked into the graph.
        if self.state._is_first_step:
            with eager_mode():
                for hook_fn, hargs in self.state._post_partition_hooks.items():
                    hook_args, hook_kwargs = hargs
                    hook_fn(*hook_args, **hook_kwargs)
                self.state._is_first_step = False

        ret = self._call(args, kwargs)
        self.state.inside_smp_step = False
        return ret

    def _call(self, args, kwargs):
        """
        Execute the function body in a pipelined manner, by splitting into microbatches.
        Can return StepOutput objects.
        """
        core.proxy_wake_up()

        spec_args, spec_kwargs = convert_tensor_to_spec(args, kwargs, self.input_signature)
        key = cache_key(self._trace_tf_function, spec_args, spec_kwargs, self.input_signature)
        self.state.compiler.load_state((key, id(self._trace_tf_function)))

        output_list = self._outer_tf_function(*args, **kwargs)
        return postprocess_outputs(output_list)

    def get_outer_function(self):
        def _outer_function(*args, **kwargs):
            func_outputs = []
            self.state.compile_status = CompileStatus.TRAIN
            register_op = register_state()
            for mb in range(self.state.num_microbatches()):
                with self.state.reset_allgather_index(
                    reset=(mb != self.state.num_microbatches() - 1)
                ):
                    self.state.microbatch = mb
                    self.state.backward = False
                    with self.strip_control_outputs():
                        with tf.control_dependencies([register_op]):
                            func_outputs.append(self._call_tf_function(mb, *args, **kwargs))

            return func_outputs

        return _outer_function

    @property
    def python_function(self):
        return self._python_function

    @property
    def input_signature(self):
        return self._input_signature

    @property
    def function_spec(self):
        return self._outer_tf_function.function_spec

    def get_initialization_function(self, *args, **kwargs):
        return self._outer_tf_function.get_initialization_function(*args, **kwargs)

    def get_concrete_function(self, *args, **kwargs):
        return self._outer_tf_function.get_concrete_function(*args, **kwargs)

    @input_signature.setter
    def input_signature(self, signature):
        self._input_signature = signature

    def __wrap_python_function(self, python_function, compiling):
        """
        Wrap the python_function in another function which does additional, SMP-specific work. Before the function
        computation, it splits the Tensor arguments along the batch dimension. It repeats the inner function execution
        for each microbatch.

        In addition, calls the register_state() op which registers the state with the backend.
        This is done dynamically during graph execution (through a custom TF op) to handle the (common) case where smp.step is
        inside a broader tf.function context, and there are multiple compiled smp.step concrete functions (e.g. for training and evaluation).
        In this case, the concrete smp.step function that will be used can change dynamically based on function arguments, without the control flow
        reaching self._trace function to update the backend state, resulting in possibly incompatible backend state.
        """
        if compiling:

            @wraps(self.python_function)
            def function_wrapper(*func_args, **func_kwargs):
                mb_args, mb_kwargs = self._splitter.preprocess_args(
                    func_args, func_kwargs, self.state.num_microbatches(), 0
                )
                return python_function(*mb_args, **mb_kwargs)

        else:

            @wraps(self.python_function)
            def function_wrapper(microbatch, *func_args, **func_kwargs):
                mb_args, mb_kwargs = self._splitter.preprocess_args(
                    func_args, func_kwargs, self.state.num_microbatches(), microbatch
                )
                with xla_scope():

                    return python_function(*mb_args, **mb_kwargs)

        function_wrapper._smp_step_function = True

        return function_wrapper

    def _warn_if_trainable_var_outside_model(self, func_graph):
        if self.state._tracing_model is not None:
            model_var_names = {
                mainify_name(var.name) for var in self.state._tracing_model.trainable_variables
            }

            for var in func_graph.variables:
                if (
                    var.trainable
                    and mainify_name(var.name) not in model_var_names
                    and not var.name.startswith("SMPDummy")
                ):
                    get_logger().warning(
                        f"Trainable variable {mainify_name(var.name)} is not part of smp.DistributedModel. This might create duplicate versions of the variable in different ranks, and lead to unintended consequences."
                    )

    def _trace(self, g_args, g_kwargs):
        """Traces and compiles the body of the smp.step for each microbatch."""

        spec_args, spec_kwargs = convert_tensor_to_spec(g_args, g_kwargs, self.input_signature)
        self.state.compile_status = CompileStatus.STEP_COMPILE

        self.state.microbatch = 0
        with self.strip_control_outputs():
            graph_with_functions = self._trace_tf_function.get_concrete_function(
                *spec_args, **spec_kwargs
            ).graph

        self._warn_if_trainable_var_outside_model(graph_with_functions)
        self.state.compile_graph = inline_funclib(graph_with_functions.as_graph_def())

        for mb in range(self.state.num_microbatches()):
            self.state.compiler.compile(
                self.state.compile_graph, self.state.core.pp_rank(), mb, self.state.op_id_to_device
            )

        if len(graph_with_functions.outputs) == 0:
            raise ValueError(
                "smp.step function must have a return value that depends on the smp.DistributedModel output."
            )

        traverser = GraphTraverser(graph_with_functions)
        output_nodes = traverser.get_nodes_by_names(
            [get_op(out).name for out in graph_with_functions.outputs]
        )

        key = cache_key(self._trace_tf_function, spec_args, spec_kwargs, self.input_signature)
        self.state.compiler.save_state((key, id(self._trace_tf_function)), output_nodes)

    @contextmanager
    def strip_control_outputs(self):
        """
        Removes the control outputs from the tf.functions. This is used to prevent
        control dependencies forming across tf.functions representing different
        microbatches, which causes hangs in pipelined execution, since all the
        control outputs of one tf.function must finish before the next tf.function
        can start.

        TF forms these control dependencies to prevent race conditions between ops
        in different tf.function accessing the same resource. In most cases, we
        will not have these race conditions because of strict ordering of microbatches
        across ticks. However, we are vulnerable to a race condition in the following
        scenario:

        A variable is being accessed by at least two ops, at least one of these is a write access,
        these two ops are assigned to the same device, and they lie in two different depth
        levels in the graph.

        In this case,
        it is possible that one such op is executed by one microbatch, and the other being
        executed by the other microbatch on the same device at the same time. There are two
        possible solutions for this:
            1. Prevent this from occurring while deciding partitions.
            2. Allow this scenario, but impose a strict ordering between ops lying
               in different depth levels that are to be executed on the same tick.
               This can be done by dividing a tick into subticks based on depth, and
               prioritizing higher depth levels in the backend.

        TODO(cakarak): implement one of these solutions.
        """

        # this function is called from two different places in tensorflow/python/eager/function.py
        org_func_graph_from_py_func = func_graph.func_graph_from_py_func

        @wraps(org_func_graph_from_py_func)
        def wrapped_func_graph_from_py_func(*args, **kwargs):
            ret = org_func_graph_from_py_func(*args, **kwargs)
            python_func_arg = kwargs["python_func"] if "python_func" in kwargs else args[1]

            # smp.step function is marked with the attribute `_smp_step_function`
            if hasattr(python_func_arg, "_smp_step_function"):
                ret.control_outputs = []
            return ret

        func_graph.func_graph_from_py_func = wrapped_func_graph_from_py_func
        yield
        func_graph.func_graph_from_py_func = org_func_graph_from_py_func
