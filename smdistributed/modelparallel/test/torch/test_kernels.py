# Standard Library
import unittest

# Third Party
import torch
import torch.nn.functional as F

# First Party
from smdistributed.modelparallel.torch.nn.softmax import (
    ScaledCausalMaskedSoftmax,
    ScaledMaskedSoftmax,
)

ATOL = 1e-3


class TestPartition(unittest.TestCase):
    def test_fused_masked_softmax(self):
        seq_length = 1024
        batch_size = 4
        num_heads = 16
        p = 0.2

        # prepare
        device = torch.device("cuda", 0)
        fused_softmax = ScaledMaskedSoftmax(1.0)

        # define masks
        mask_value = torch.tensor(-1e4, dtype=torch.float16, device=device)
        fused_mask_value = torch.tensor(1.0, dtype=torch.float16, device=device)
        mask = torch.zeros(
            batch_size, 1, seq_length, seq_length, dtype=torch.float16, device=device
        )
        indices = torch.rand(batch_size, 1, seq_length, seq_length) < p
        org_mask = torch.where(indices.to(device), mask, mask_value)
        fused_mask = torch.where(indices.to(device), mask, fused_mask_value).to(torch.uint8)

        input_tensor = torch.randn(
            batch_size, num_heads, seq_length, seq_length, dtype=torch.float16, device=device
        )

        # pass input through non-fused computation to find ground truth
        attention_scores = input_tensor + org_mask
        attention_probs = F.softmax(attention_scores, dim=-1)

        # pass input through kernel
        fused_attention_probs = fused_softmax(input_tensor, fused_mask)

        print(fused_attention_probs[0, 0, :, :])
        print(attention_probs[0, 0, :, :])
        self.assertTrue(
            torch.allclose(fused_attention_probs.cpu(), attention_probs.cpu(), atol=ATOL)
        )

    def test_fused_upper_triang_softmax(self):
        seq_length = 1024
        batch_size = 4
        num_heads = 16

        # prepare
        device = torch.device("cuda", 0)
        fused_softmax = ScaledCausalMaskedSoftmax(1.0)

        causal_mask = torch.tril(
            torch.ones((seq_length, seq_length), dtype=torch.uint8, device=device)
        )
        causal_mask = causal_mask.reshape(1, 1, seq_length, seq_length)
        mask_value = torch.tensor(-1e4, dtype=torch.float32, device=device)
        input_tensor = torch.randn(
            batch_size, num_heads, seq_length, seq_length, dtype=torch.float16, device=device
        )

        # pass input through non-fused computation to find ground truth
        seq_len1, seq_len2 = input_tensor.size(-2), input_tensor.size(-1)
        sliced_mask = causal_mask[:, :, (seq_len2 - seq_len1) : seq_len2, :seq_len2].bool()
        attention_scores = torch.where(sliced_mask, input_tensor, mask_value.to(input_tensor.dtype))
        attention_probs = F.softmax(attention_scores, dim=-1)

        # pass input through kernel
        fused_attention_probs = fused_softmax(input_tensor)

        print(fused_attention_probs[0, 0, :, :])
        print(attention_probs[0, 0, :, :])
        self.assertTrue(
            torch.allclose(fused_attention_probs.cpu(), attention_probs.cpu(), atol=ATOL)
        )


if __name__ == "__main__":
    unittest.main()
