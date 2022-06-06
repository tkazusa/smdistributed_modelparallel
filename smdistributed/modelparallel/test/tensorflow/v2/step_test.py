#!/usr/bin/env python3

# Standard Library
import types
import unittest

# Third Party
import tensorflow as tf
from mock import MagicMock, patch

# First Party
import smdistributed.modelparallel.tensorflow as smp
from smdistributed.modelparallel.backend.core import ModelParallelCore
from smdistributed.modelparallel.tensorflow.state_mod import TFModelParallelState
from smdistributed.modelparallel.tensorflow.v2 import step_mod

NUM_MICROBATCHES = 4


class TestDecoratorWrapping(unittest.TestCase):
    """ Test that various decorator usage patterns correctly wrap the target function.
    This does not test that the decorator correctly interacts with the compiler/state.
    """

    def setUp(self):
        # Patch test dependencies (this is roughly equivalent to @patch).
        core_patch = patch(
            "smdistributed.modelparallel.tensorflow.v2.step_mod.core", autospec=ModelParallelCore
        )
        self.addCleanup(core_patch.stop)
        mock_core = core_patch.start()

        state_patch = patch(
            "smdistributed.modelparallel.tensorflow.v2.step_mod.state",
            autospec=TFModelParallelState(),
        )
        self.addCleanup(state_patch.stop)
        mock_state = state_patch.start()

        # Configure test mocks.
        mock_state.num_microbatches = lambda: NUM_MICROBATCHES
        mock_state.compiler = MagicMock()
        mock_state.ctrl_mgr = MagicMock()

        # Trick StepFunction class into thinking its the first compilation at the start of every test.
        step_mod.StepFunction._smp_py_func_compilation_target = None

        step_mod.register_state = tf.no_op

    def assert_correct_arguments_with_self_reference(
        test_class_instance,
        expected_args,
        expected_kwargs,
        expected_self_reference,
        actual_args,
        actual_kwargs,
    ):
        test_class_instance.assertEqual(expected_self_reference, actual_args[0])
        test_class_instance.assertEqual(expected_args, actual_args[1:])
        test_class_instance.assertEqual(expected_kwargs, actual_kwargs)

    def assert_correct_arguments(
        test_class_instance, expected_args, expected_kwargs, actual_args, actual_kwargs
    ):
        test_class_instance.assertEqual(expected_args, actual_args)
        test_class_instance.assertEqual(expected_kwargs, actual_kwargs)

    def test_unit_test_helpers_do_not_swallow_assertion_failures(self):
        # Intentionally fail the test to verify that graph is called (and that unit test assertions are executed).
        expected_args = (0, 1, 2)
        actual_args = "this should throw an exception"

        expected_kwargs = {"kwarg1": True, "kwarg2": False}

        class Model:
            @smp.step
            def graph(*args, **kwargs):
                self.assert_correct_arguments_with_self_reference(
                    expected_args, expected_kwargs, model, args, kwargs
                )

        model = Model()

        with self.assertRaises(AssertionError):
            model.graph(*actual_args, **expected_kwargs)

    def test_static_method_wrapping(self):
        expected_return_value = "expected_return_value"
        expected_args = (0, 1, 2)
        expected_kwargs = {"kwarg1": True, "kwarg2": False}

        class Model:
            def graph(*args, **kwargs):
                self.assert_correct_arguments(expected_args, expected_kwargs, args, kwargs)
                return expected_return_value

        # Access the function through the class type, yielding a static method.
        decorated = smp.step(Model.graph)

        self.assertEqual(
            [expected_return_value for _ in range(NUM_MICROBATCHES)],
            decorated(*expected_args, **expected_kwargs).outputs,
        )

    def test_method_wrapping_via_manual_call(self):
        expected_return_value = "expected_return_value"
        expected_args = (0, 1, 2)
        expected_kwargs = {"kwarg1": True, "kwarg2": False}

        class Model:
            def graph(*args, **kwargs):
                self.assert_correct_arguments_with_self_reference(
                    expected_args, expected_kwargs, model, args, kwargs
                )
                return expected_return_value

        model = Model()
        decorated = smp.step(model.graph)

        self.assertEqual(
            [expected_return_value for _ in range(NUM_MICROBATCHES)],
            decorated(*expected_args, **expected_kwargs).outputs,
        )

    def test_method_wrapping_via_decorator(self):
        expected_return_value = "expected_return_value"
        expected_args = (0, 1, 2)
        expected_kwargs = {"kwarg1": True, "kwarg2": False}

        class Model:
            @smp.step
            def graph(*args, **kwargs):
                self.assert_correct_arguments_with_self_reference(
                    expected_args, expected_kwargs, model, args, kwargs
                )
                return expected_return_value

        model = Model()

        self.assertEqual(
            [expected_return_value for _ in range(NUM_MICROBATCHES)],
            model.graph(*expected_args, **expected_kwargs).outputs,
        )

    def test_function_wrapping_via_decorator(self):
        expected_return_value = "expected_return_value"
        expected_args = (0, 1, 2)
        expected_kwargs = {"kwarg1": True, "kwarg2": False}

        @smp.step
        def graph(*args, **kwargs):
            self.assert_correct_arguments(expected_args, expected_kwargs, args, kwargs)
            return expected_return_value

        self.assertEqual(
            [expected_return_value for _ in range(NUM_MICROBATCHES)],
            graph(*expected_args, **expected_kwargs).outputs,
        )

    def test_function_wrapping_via_decorator_with_parameters(self):
        expected_return_value = "expected_return_value"
        expected_args = (0, 1, 2)
        expected_kwargs = {"kwarg1": True, "kwarg2": False}

        @smp.step(input_signature=None)
        def graph(*args, **kwargs):
            self.assert_correct_arguments(expected_args, expected_kwargs, args, kwargs)
            return expected_return_value

        self.assertEqual(
            [expected_return_value for _ in range(NUM_MICROBATCHES)],
            graph(*expected_args, **expected_kwargs).outputs,
        )

    def test_function_wrapping_via_manual_call(self):
        expected_return_value = "expected_return_value"
        expected_args = (0, 1, 2)
        expected_kwargs = {"kwarg1": True, "kwarg2": False}

        def graph(*args, **kwargs):
            self.assert_correct_arguments(expected_args, expected_kwargs, args, kwargs)
            return expected_return_value

        decorated = smp.step(graph)

        self.assertEqual(decorated.python_function, graph)
        self.assertEqual(expected_return_value, graph(*expected_args, **expected_kwargs))

    def test_method_decorator_injects_self_arg(self):
        class Model:
            @smp.step(input_signature=[tf.TensorSpec(shape=(9, 3), dtype=tf.float32)])
            def graph(self_model, steps):
                pass

        model = Model()

        # Assert that the graph StepFunction python_function on the instance is equivalent to the
        # graph StepFunction python_function on the class, except bound to the instance.
        # In other words, this verifies that model.graph automatically injects `self` in the parameter list.
        self.assertEqual(
            model.graph.python_function, types.MethodType(Model.graph.python_function, model)
        )

    # def test_disallow_multiple_unique_compilation_targets(self):
    #    class Model:
    #        @smp.step
    #        def graph1(self_model):
    #            pass

    #        @smp.step
    #        def graph2(self_model):
    #            pass

    #    model = Model()
    #    model.graph1()

    #    with self.assertRaises(GraphBuildError):
    #        model.graph2()


if __name__ == "__main__":
    unittest.main()
