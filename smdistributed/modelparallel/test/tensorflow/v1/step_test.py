#!/usr/bin/env python3

# Standard Library
import unittest

# First Party
import smdistributed.modelparallel.tensorflow.v1 as smp


class TestStepDecorator(unittest.TestCase):
    def raises_on_invalid_non_split_inputs(self):

        with self.assertRaises(TypeError):

            @smp.step(non_split_inputs=23)
            def train_step():
                pass


if __name__ == "__main__":
    unittest.main()
