import unittest
import torch
import torch.nn as nn
import numpy as np
import smdistributed.modelparallel.torch as smp

class TestDelayParams(unittest.TestCase):
    def test_delay_params(self):
        class Net1(torch.nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                self.linear1 = nn.Linear(100000, 1000000)

            def forward(self, x):
                out = self.linear1(x)
                return out

        class NetSequential(torch.nn.Module):
            def __init__(self):
                super(NetSequential, self).__init__()
                self.net1 = Net1()
                self.linear2 = nn.Linear(100000, 1000000)
                self.linear3 = nn.Linear(100000, 1000000)
                self.sequential = torch.nn.Sequential(
                    self.net1, self.linear2, self.net1, self.linear3
                )

            def forward(self, x):
                out2 = self.sequential(x)
                return out2

        model = None
        # Removing the context manager will make the memory grow till OOM on p3.16x
        with smp.delay_param_initialization(enabled=True):
            model = NetSequential()

if __name__ == "__main__":
    unittest.main()
