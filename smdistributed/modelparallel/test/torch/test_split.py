# Standard Library
import unittest

# Third Party
import torch

# First Party
from smdistributed.modelparallel.torch.step import PTTensorSplitter

NUM_MICROBATCHES = 4


class CustomType:
    def __init__(self, tensor, other):
        self.data = tensor
        self.other = other

    def __eq__(self, other):
        return torch.all(self.data.eq(other.data)) and self.other == other.other

    def smp_slice(self, num_mb, mb, axis):
        dim_size = list(self.data.size())[axis]

        split_size = dim_size // num_mb
        sliced_tensor = self.data.narrow(axis, mb * split_size, split_size)
        return CustomType(sliced_tensor, self.other)


class TestSplit(unittest.TestCase):
    def test_simple_args_split(self):
        args = (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        )
        kwargs = {}
        non_split_inputs = []
        input_split_axes = {}
        expected_args_list = [
            (torch.tensor([1]), torch.tensor([[2, 3]]), torch.tensor([1, 2])),
            (torch.tensor([2]), torch.tensor([[4, 5]]), torch.tensor([3, 4])),
            (torch.tensor([3]), torch.tensor([[3, 4]]), torch.tensor([5, 6])),
            (torch.tensor([4]), torch.tensor([[5, 6]]), torch.tensor([7, 8])),
        ]
        expected_kwargs_list = [{} for _ in range(NUM_MICROBATCHES)]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_simple_kwargs_split(self):
        args = ()
        kwargs = {
            "x": torch.tensor([1, 2, 3, 4]),
            "y": torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
            "z": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        }
        non_split_inputs = []
        input_split_axes = {}
        expected_args_list = [() for _ in range(NUM_MICROBATCHES)]
        expected_kwargs_list = [
            {"x": torch.tensor([1]), "y": torch.tensor([[2, 3]]), "z": torch.tensor([1, 2])},
            {"x": torch.tensor([2]), "y": torch.tensor([[4, 5]]), "z": torch.tensor([3, 4])},
            {"x": torch.tensor([3]), "y": torch.tensor([[3, 4]]), "z": torch.tensor([5, 6])},
            {"x": torch.tensor([4]), "y": torch.tensor([[5, 6]]), "z": torch.tensor([7, 8])},
        ]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_args_split_with_non_split_inputs(self):
        args = (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        )
        kwargs = {}
        non_split_inputs = ["x", "z"]
        input_split_axes = {}
        expected_args_list = [
            (
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([[2, 3]]),
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            ),
            (
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([[4, 5]]),
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            ),
            (
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([[3, 4]]),
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            ),
            (
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([[5, 6]]),
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            ),
        ]
        expected_kwargs_list = [{} for _ in range(NUM_MICROBATCHES)]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_kwargs_split_with_non_split_inputs(self):
        args = ()
        kwargs = {
            "x": torch.tensor([1, 2, 3, 4]),
            "y": torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
            "z": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        }
        non_split_inputs = ["y"]
        input_split_axes = {}
        expected_args_list = [() for _ in range(NUM_MICROBATCHES)]
        expected_kwargs_list = [
            {
                "x": torch.tensor([1]),
                "y": torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
                "z": torch.tensor([1, 2]),
            },
            {
                "x": torch.tensor([2]),
                "y": torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
                "z": torch.tensor([3, 4]),
            },
            {
                "x": torch.tensor([3]),
                "y": torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
                "z": torch.tensor([5, 6]),
            },
            {
                "x": torch.tensor([4]),
                "y": torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
                "z": torch.tensor([7, 8]),
            },
        ]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_args_split_with_input_split_axes(self):
        args = (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([[2, 3], [4, 5], [3, 4], [5, 6]]),
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        )
        kwargs = {}
        non_split_inputs = []
        input_split_axes = {}
        expected_args_list = [
            (torch.tensor([1]), torch.tensor([[2, 3]]), torch.tensor([1, 2])),
            (torch.tensor([2]), torch.tensor([[4, 5]]), torch.tensor([3, 4])),
            (torch.tensor([3]), torch.tensor([[3, 4]]), torch.tensor([5, 6])),
            (torch.tensor([4]), torch.tensor([[5, 6]]), torch.tensor([7, 8])),
        ]
        expected_kwargs_list = [{} for _ in range(NUM_MICROBATCHES)]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_kwargs_split_with_input_split_axes(self):
        args = ()
        kwargs = {
            "x": torch.tensor([1, 2, 3, 4]),
            "y": torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]]),
            "z": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        }
        non_split_inputs = []
        input_split_axes = {"y": 1}
        expected_args_list = [() for _ in range(NUM_MICROBATCHES)]
        expected_kwargs_list = [
            {"x": torch.tensor([1]), "y": torch.tensor([[2], [3]]), "z": torch.tensor([1, 2])},
            {"x": torch.tensor([2]), "y": torch.tensor([[3], [4]]), "z": torch.tensor([3, 4])},
            {"x": torch.tensor([3]), "y": torch.tensor([[4], [5]]), "z": torch.tensor([5, 6])},
            {"x": torch.tensor([4]), "y": torch.tensor([[5], [6]]), "z": torch.tensor([7, 8])},
        ]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_everything(self):
        args = (torch.tensor([1, 2, 3, 4]),)
        kwargs = {
            "y": torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]]),
            "z": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        }
        non_split_inputs = ["x"]
        input_split_axes = {"y": 1}
        expected_args_list = [
            (torch.tensor([1, 2, 3, 4]),),
            (torch.tensor([1, 2, 3, 4]),),
            (torch.tensor([1, 2, 3, 4]),),
            (torch.tensor([1, 2, 3, 4]),),
        ]
        expected_kwargs_list = [
            {"y": torch.tensor([[2], [3]]), "z": torch.tensor([1, 2])},
            {"y": torch.tensor([[3], [4]]), "z": torch.tensor([3, 4])},
            {"y": torch.tensor([[4], [5]]), "z": torch.tensor([5, 6])},
            {"y": torch.tensor([[5], [6]]), "z": torch.tensor([7, 8])},
        ]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_custom_type(self):
        args = (torch.tensor([1, 2, 3, 4]),)
        kwargs = {
            "y": CustomType(torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]]), "other_element"),
            "z": CustomType(torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]), "another_one"),
        }
        non_split_inputs = ["x"]
        input_split_axes = {"y": 1}
        expected_args_list = [
            (torch.tensor([1, 2, 3, 4]),),
            (torch.tensor([1, 2, 3, 4]),),
            (torch.tensor([1, 2, 3, 4]),),
            (torch.tensor([1, 2, 3, 4]),),
        ]
        expected_kwargs_list = [
            {
                "y": CustomType(torch.tensor([[2], [3]]), "other_element"),
                "z": CustomType(torch.tensor([1, 2]), "another_one"),
            },
            {
                "y": CustomType(torch.tensor([[3], [4]]), "other_element"),
                "z": CustomType(torch.tensor([3, 4]), "another_one"),
            },
            {
                "y": CustomType(torch.tensor([[4], [5]]), "other_element"),
                "z": CustomType(torch.tensor([5, 6]), "another_one"),
            },
            {
                "y": CustomType(torch.tensor([[5], [6]]), "other_element"),
                "z": CustomType(torch.tensor([7, 8]), "another_one"),
            },
        ]
        self._test_split(
            args,
            kwargs,
            non_split_inputs,
            input_split_axes,
            expected_args_list,
            expected_kwargs_list,
        )

    def test_uneven_split(self):
        args = (torch.tensor([1, 2, 3, 4]),)
        kwargs = {
            "y": torch.tensor([[2, 3, 4], [3, 4, 5]]),
            "z": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        }
        non_split_inputs = ["x"]
        input_split_axes = {"y": 1}
        expected_args_list = [
            (torch.tensor([1, 2, 3, 4])),
            (torch.tensor([1, 2, 3, 4])),
            (torch.tensor([1, 2, 3, 4])),
            (torch.tensor([1, 2, 3, 4])),
        ]
        expected_kwargs_list = [
            {"y": torch.tensor([[2], [3]]), "z": torch.tensor([1, 2])},
            {"y": torch.tensor([[3], [4]]), "z": torch.tensor([3, 4])},
            {"y": torch.tensor([[4], [5]]), "z": torch.tensor([5, 6])},
            {"y": torch.tensor([[5], [6]]), "z": torch.tensor([7, 8])},
        ]
        with self.assertRaises(ValueError):
            self._test_split(
                args,
                kwargs,
                non_split_inputs,
                input_split_axes,
                expected_args_list,
                expected_kwargs_list,
            )

    def _test_split(
        self,
        args,
        kwargs,
        non_split_inputs,
        input_split_axes,
        expected_args_list,
        expected_kwargs_list,
    ):
        splitter = PTTensorSplitter(lambda x, y, z: None, non_split_inputs, input_split_axes)
        split_args, split_kwargs = splitter.preprocess_args_all_mbs(args, kwargs, NUM_MICROBATCHES)
        for mb in range(NUM_MICROBATCHES):
            for split_arg, expected_arg in zip(split_args[mb], expected_args_list[mb]):
                if isinstance(split_arg, torch.Tensor):
                    self.assertTrue(torch.all(split_arg.eq(expected_arg)))
                else:
                    self.assertTrue(split_arg == expected_arg)

            for k, expected_kwarg in split_kwargs[mb].items():
                if isinstance(split_kwargs[mb][k], torch.Tensor):
                    self.assertTrue(torch.all(split_kwargs[mb][k].eq(expected_kwarg)))
                else:
                    self.assertTrue(split_kwargs[mb][k] == expected_kwarg)


if __name__ == "__main__":
    unittest.main()
