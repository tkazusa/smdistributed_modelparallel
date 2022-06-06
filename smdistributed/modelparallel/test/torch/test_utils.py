# Standard Library
import collections
import unittest

# Third Party
import numpy as np
import torch

# First Party
from smdistributed.modelparallel.torch.utils import (
    flatten_structure,
    map_structure,
    unflatten_structure,
)


class TestFlatten(unittest.TestCase):
    PointXY = collections.namedtuple("Point", ["x", "y"])

    def assert_correct(self, *args):
        flat_res = []
        ids = []
        for x in args:
            flat_structure, structure_id = flatten_structure(x)
            flat_res.append(flat_structure)
            ids.append(structure_id)

        for i, x in enumerate(args):
            structure = unflatten_structure(flat_res[i], ids[i])

            self.assertTrue(structure, x)
            self.assertTrue(isinstance(flat_res[i], list))
            self.assertTrue(
                all([not isinstance(item, (list, tuple, set, dict)) for item in flat_res[i]])
            )

    def test_flatten(self):
        d = {"a": [3, set()], "v": {"x", "y"}}
        self.assert_correct(d)

    def test_multiple_flatten(self):
        class A:
            pass

        a = A()
        x = (40, {"a": 3, "b": {3, 6, 1}}, a, a)
        y = {3, 7, torch.tensor([3, 4])}

        self.assert_correct(x, y)

    def test_flatten_unflatten(self):
        structure = ((3, 2), 1, (4, 5, (6, 7), 8))
        flat_structure, structure_id = flatten_structure(structure)
        self.assertEqual(flat_structure, [3, 2, 1, 4, 5, 6, 7, 8])
        structure2 = unflatten_structure(flat_structure, structure_id)
        self.assertEqual(structure2, structure)
        structure = (TestFlatten.PointXY(x=4, y=2), ((TestFlatten.PointXY(x=1, y=0),),))
        flat = [4, 2, 1, 0]
        flat_structure, structure_id = flatten_structure(structure)
        self.assertEqual(flat_structure, flat)
        structure2 = unflatten_structure(flat_structure, structure_id)
        self.assertEqual(structure2, structure)

        self.assertEqual(structure2[0].x, 4)
        self.assertEqual(structure2[0].y, 2)
        self.assertEqual(structure[1][0][0].x, 1)
        self.assertEqual(structure[1][0][0].y, 0)

    def test_flatten_dict(self):
        for mapping_type in [collections.OrderedDict]:
            ordered = mapping_type([("d", 3), ("b", 1), ("a", 0), ("c", 2)])
            plain = {"d": 3, "b": 1, "a": 0, "c": 2}
            ordered_flat, _ = flatten_structure(ordered)
            plain_flat, _ = flatten_structure(plain)
            self.assertEqual([3, 1, 0, 2], ordered_flat)
            self.assertEqual([3, 1, 0, 2], plain_flat)

    def test_unflatten_dict(self):
        for mapping_type in [collections.OrderedDict]:
            custom = mapping_type([("d", 0), ("b", 0), ("a", 0), ("c", 0)])
            plain = {"d": 0, "b": 0, "a": 0, "c": 0}
            seq = [0, 1, 2, 3]
            flat_structure_custom, flat_structure_custom_id = flatten_structure(custom)
            flat_structure_plain, flat_structure_plain_id = flatten_structure(plain)

            custom_reconstruction = unflatten_structure(
                flat_structure_custom, flat_structure_custom_id
            )
            plain_reconstruction = unflatten_structure(
                flat_structure_plain, flat_structure_plain_id
            )
            self.assertIsInstance(custom_reconstruction, mapping_type)
            self.assertIsInstance(plain_reconstruction, dict)
            self.assertEqual(
                mapping_type([("d", 0), ("b", 0), ("a", 0), ("c", 0)]), custom_reconstruction
            )
            self.assertEqual({"d": 0, "b": 0, "a": 0, "c": 0}, plain_reconstruction)

    @unittest.skip("Mapping views not supported")
    def test_flatten_unflatten_mapping_views(self):
        ordered = collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)])

        ordered_keys_flat, _ = flatten_structure(ordered.keys())
        ordered_values_flat, _ = flatten_structure(ordered.values())
        ordered_items_flat, oi_id = flatten_structure(ordered.items())
        self.assertEqual([3, 1, 0, 2], ordered_values_flat)
        self.assertEqual(["d", "b", "a", "c"], ordered_keys_flat)
        self.assertEqual(["d", 3, "b", 1, "a", 0, "c", 2], ordered_items_flat)

        self.assertEqual(
            [("d", 3), ("b", 1), ("a", 0), ("c", 2)], unflatten_structure(ordered_items_flat, oi_id)
        )

    Abc = collections.namedtuple("A", ("b", "c"))

    def test_flatten_unflatten_with_dicts(self):
        mess = [
            "z",
            TestFlatten.Abc(3, 4),
            {
                "d": collections.OrderedDict({41: 4}),
                "c": [1, collections.OrderedDict([("b", 3), ("a", 2)])],
                "b": 5,
            },
            17,
        ]
        flattened, structure_id = flatten_structure(mess)
        self.assertEqual(flattened, ["z", 3, 4, 4, 1, 3, 2, 5, 17])

        structure_of_mess = [
            14,
            TestFlatten.Abc("a", True),
            {
                "d": collections.OrderedDict({41: 42}),
                "c": [0, collections.OrderedDict([("b", 9), ("a", 8)])],
                "b": 3,
            },
            "hi everybody",
        ]

        unflattened = unflatten_structure(flattened, structure_id)
        self.assertEqual(unflattened, mess)

        unflattened_ordered_dict = unflattened[2]["c"][1]
        self.assertIsInstance(unflattened_ordered_dict, collections.OrderedDict)
        self.assertEqual(list(unflattened_ordered_dict.keys()), ["b", "a"])

        unflattened_d = unflattened[2]["d"]
        self.assertIsInstance(unflattened_d, collections.OrderedDict)
        self.assertEqual(list(unflattened_d.keys()), [41])

    def test_flatten_numpy_torch_tensor_not_flattened(self):
        for structure in [np.array([1, 2, 3]), torch.ones(3)]:
            structure = np.array([1, 2, 3])
            flattened, _ = flatten_structure(structure)
            assert len(flattened) == 1

    def test_flatten_string_not_flattened(self):
        structure = "lots of letters"
        flattened, struct_id = flatten_structure(structure)
        assert len(flattened) == 1
        unflattened = unflatten_structure(flattened, struct_id)
        self.assertEqual(unflattened, structure)

    def test_map_structure(self):
        structure1 = (((1, 2), 3), 4, (5, 6))
        structure2 = (((2, 3), 4), 5, (6, 7))
        structure1_plus1 = map_structure(lambda x: x + 1, structure1)
        self.assertEqual(structure2, structure1_plus1)

        structure3 = collections.defaultdict(list)
        structure3["a"] = [1, 2, 3, 4]
        structure3["b"] = [2, 3, 4, 5]

        expected_structure3 = collections.defaultdict(list)
        expected_structure3["a"] = [2, 3, 4, 5]
        expected_structure3["b"] = [3, 4, 5, 6]

        self.assertEqual(expected_structure3, map_structure(lambda x: x + 1, structure3))

        self.assertEqual((), map_structure(lambda x: x + 1, ()))
        self.assertEqual([], map_structure(lambda x: x + 1, []))
        self.assertEqual({}, map_structure(lambda x: x + 1, {}))

        with self.assertRaisesRegex(TypeError, "callable"):
            map_structure("bad", structure1_plus1)

        with self.assertRaisesRegex(TypeError, "missing.*required.*"):
            map_structure(lambda x: x)

    ABTuple = collections.namedtuple("ab_tuple", "a, b")

    def test_map_structure_with_strings(self):
        inp_a = TestFlatten.ABTuple(a="foo", b=("bar", "baz"))
        out = map_structure(lambda string: string * 2, inp_a)
        self.assertEqual("foofoo", out.a)
        self.assertEqual("barbar", out.b[0])
        self.assertEqual("bazbaz", out.b[1])


if __name__ == "__main__":
    unittest.main()
