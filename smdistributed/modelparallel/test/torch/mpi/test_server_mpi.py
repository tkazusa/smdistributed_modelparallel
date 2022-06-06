# Standard Library
import unittest

# Third Party
import torch
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp


@smp.step
def train_step(model, data):
    return model(data)


class TestModuleManager(unittest.TestCase):
    def test_fwdpass_microbatches(self):
        """
        Testing that forward pass split over microbatches produces same output
        """

        class SeqModel(nn.Module):
            def __init__(self, N, D):
                super(SeqModel, self).__init__()
                self.lin = nn.Linear(D, D)
                self.relu = nn.ReLU()
                self.lin2 = nn.Linear(D, D)
                self.lin3 = nn.Linear(D, D)

            def forward(self, x):
                x = self.lin(x)
                x = self.relu(x)
                x = self.lin2(x)
                x = self.lin3(x)
                return x

        N, D_in, H, D_out = 64, 100, 100, 100
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        smp.init({"microbatches": 4, "pipeline": "simple", "partitions": 2, "auto_partition": True})
        org_model = SeqModel(N, D_in)
        model = SeqModel(N, D_in)
        model.load_state_dict(org_model.state_dict())

        device = torch.device("cuda", smp.local_rank())
        model.to(device)
        model = smp.DistributedModel(model)
        x = x.to(device)
        outputs_mb = train_step(model, x)
        outputs = outputs_mb.concat()
        smp.reset()

        cpu = torch.device("cpu")
        x = x.to(cpu)
        model.to(cpu)
        outputs_nonsmp = org_model(x)
        torch.allclose(outputs.to(cpu), outputs_nonsmp)

    def test_fwd_pass_autopart(self):
        class SeqModel(nn.Module):
            def __init__(self, N, D):
                super(SeqModel, self).__init__()
                self.lin = nn.Linear(D, D)
                self.relu = nn.ReLU()
                self.lin2 = nn.Linear(D, D)
                self.lin3 = nn.Linear(D, D)

            def forward(self, x, dummy_kwarg=False):
                x = self.lin(x)
                x = self.relu(x)
                x = self.lin2(x)
                x = self.lin3(x)
                return x

        N, D_in, H, D_out = 64, 100, 100, 100
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        smp.init({"microbatches": 2, "pipeline": "simple", "partitions": 2, "auto_partition": True})
        model = SeqModel(N, D_in)
        model = smp.DistributedModel(model)
        train_step(model, x.to(torch.device("cuda", smp.local_rank())))

    def test_fwd_manual_partitioning(self):
        class SeqModel(nn.Module):
            def __init__(self, N, D):
                super(SeqModel, self).__init__()
                with smp.partition(0):
                    self.lin = nn.Linear(D, D)
                with smp.partition(1):
                    self.relu = nn.ReLU()
                    self.lin2 = nn.Linear(D, D)
                with smp.partition(0):
                    self.lin3 = nn.Linear(D, D)

            def forward(self, x, dummy_kwarg=False):
                x = self.lin(x)
                x = self.relu(x)
                x = self.lin2(x)
                x = self.lin3(x)
                return x

        N, D_in, H, D_out = 64, 100, 100, 100
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        smp.init(
            {
                "microbatches": 2,
                "pipeline": "simple",
                "partitions": 2,
                "auto_partition": False,
                "default_partition": 0,
            }
        )

        model = SeqModel(N, D_in)
        model = smp.DistributedModel(model)
        train_step(model, x.to(torch.device("cuda", smp.local_rank())))


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
