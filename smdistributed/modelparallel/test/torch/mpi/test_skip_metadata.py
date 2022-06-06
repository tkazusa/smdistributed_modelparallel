# Standard Library
import time
import unittest

# Third Party
import numpy as np
import torch

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch import smplib
from smdistributed.modelparallel.torch.ops import recv, send, synchronize, wait_and_clear

cfg = {"microbatches": 2, "partitions": 2}


class TestSkipMetadata(unittest.TestCase):
    def generate_tensor(self, size, source=0, link_id=0):
        device = torch.device("cuda")
        tensor = torch.ones(size)
        tensor = tensor.to(device)
        tensor_meta = smplib.TorchTensorMeta(
            dims=len(tensor.size()),
            dummy_tensor=tensor,
            shape_vec=list(tensor.size()),
            source=source,
            link_id=link_id,
        )
        return tensor, tensor_meta

    def test_simple_usage(self):
        smp.init(cfg)
        torch.cuda.set_device(smp.local_rank())
        device = torch.device("cuda")
        tensor, tensor_meta = self.generate_tensor((2, 2))

        if smp.rank() == 0:
            handle = send(
                tensor, 1, 0, needs_meta_transmission=False, server=False, release_after_send=False
            )
            wait_and_clear(handle)
        else:
            dummy = torch.empty((0,), device=device)
            handle = recv(dummy, 0, 0, metadata=tensor_meta, server=False)
            output_tensor = synchronize(handle)

        if smp.rank() == 1:
            np.testing.assert_allclose(
                tensor.detach().cpu(), output_tensor.detach().cpu(), atol=1e-3
            )
        smp.barrier()

    def test_listener_simple(self):
        smp.init(cfg)
        torch.cuda.set_device(smp.local_rank())
        tensor, tensor_meta = self.generate_tensor((2, 2))

        if smp.rank() == 0:
            handle = send(tensor, 1, 0, needs_meta_transmission=False, server=True)
            synchronize(handle)
        else:
            incoming_tensors_metadata = [[tensor_meta]]
            smplib.smp_torch_register_incoming_tensor_metadata(incoming_tensors_metadata, 0)
            smplib.smp_torch_register_minibatch_preemptive_receptions(0)

            while not smplib.smp_torch_check_tensor(0, 0):
                time.sleep(0.001)
            output_tensor = smplib.smp_torch_get_tensor(0, 0)
            smplib.smp_torch_clear_tensor_reception(0, 0)

        if smp.rank() == 1:
            np.testing.assert_allclose(
                tensor.detach().cpu(), output_tensor.detach().cpu(), atol=1e-3
            )
        smp.barrier()


if __name__ == "__main__":
    unittest.main()
    smp.barrier()
