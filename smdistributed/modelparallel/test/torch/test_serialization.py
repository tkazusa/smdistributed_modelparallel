# Standard Library
import unittest

# Third Party
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.serialization import SerializationManager, TensorStub


class TestSerialization(unittest.TestCase):
    def test_class_type_with_tensor(self):
        class A:
            def __init__(self, x):
                self.tensor = torch.tensor([x])

            def increment(self):
                self.tensor += 1

        a = A(3.2)
        s = SerializationManager()
        serialized, _ = s.serialize(a, False, [])

        self.assertTrue(isinstance(serialized.tensor, TensorStub))
        self.assertTrue(isinstance(a.tensor, torch.Tensor))

    def test_class_type_without_tensor(self):
        class A:
            def __init__(self, x):
                self.x = x

            def increment(self):
                self.x += 1

        a = A(3.2)
        s = SerializationManager()
        serialized, _ = s.serialize(a, False, [])
        self.assertTrue(a.x == serialized.x)


if __name__ == "__main__":
    smp.init({"partitions": 1})
    unittest.main()
