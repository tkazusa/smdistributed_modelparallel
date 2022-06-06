# Standard Library
import os
import random
import unittest

# Third Party
import torch
import torch.nn as nn

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.test.torch.mpi_4ps.utils import TransformerConfig, TransformerLayer
from smdistributed.modelparallel.test.torch.utils import (
    ATOL,
    equalize_embedding_weights,
    equalize_linear_weights,
)
from smdistributed.modelparallel.torch.nn.embedding import DistributedEmbedding
from smdistributed.modelparallel.torch.nn.layer_norm import DistributedLayerNorm, FusedLayerNorm
from smdistributed.modelparallel.torch.nn.linear import DistributedLinear
from smdistributed.modelparallel.torch.nn.transformer import DistributedTransformerLayer
from smdistributed.modelparallel.torch.nn.utils import get_local_channels, get_start_pos_for_slicing
from smdistributed.modelparallel.torch.tp_registry import get_weight_slice

# TODO replace baseline from apex to native
# TODO multi-dimensional normalized_shape?


class TestDistributedOps(unittest.TestCase):
    def create_inputs_with_global_shape(self, shape):
        torch.manual_seed(432)
        torch.cuda.set_device(smp.local_rank())
        x = torch.randn(*shape).to(torch.device("cuda"))
        x.requires_grad = True
        x_slice = (
            torch.clone(x)
            .detach()
            .narrow(0, shape[0] // smp.tp_size() * smp.tp_rank(), shape[0] // smp.tp_size())
            .contiguous()
        )
        x_slice.requires_grad = True

        return x, x_slice

    def create_indices_with_global_shape(self, shape, low=0, high=4):
        torch.manual_seed(432)
        torch.cuda.set_device(smp.local_rank())
        x = torch.randint(low, high, shape, device=torch.device("cuda"), requires_grad=False)
        slice_size = shape[0] // smp.tp_size()

        x_slice = (
            torch.clone(x).detach().narrow(0, slice_size * smp.tp_rank(), slice_size).contiguous()
        )
        x_slice.requires_grad = False

        return x, x_slice

    def create_attention_mask(self, batch_size, seq_length):

        random.seed(1236)
        device = torch.device("cuda", smp.local_rank())
        mask = torch.zeros(*[batch_size, 1, 1, seq_length], dtype=torch.float32, device=device)

        batch_offset = 0
        masked_count = int(round(0.2 * seq_length))
        for i in range(batch_size):
            masked_indices = batch_offset + torch.tensor(
                [random.randint(0, seq_length) for _ in range(masked_count)],
                dtype=torch.int64,
                device=device,
            )
            value = -10000.0 * torch.ones(masked_count, dtype=torch.float32, device=device)
            mask.put_(masked_indices, value)
            batch_offset += seq_length

        local_bs = batch_size // smp.tp_size()
        mask_slice = mask.narrow(0, local_bs * smp.tp_rank(), local_bs)

        return mask, mask_slice

    def test_dist_transformer_opt_memory_unbalanced(self):
        self._dist_transformer(optimize="memory", unbalanced=True)

    def test_dist_transformer_opt_speed_unbalanced(self):
        self._dist_transformer(optimize="speed", unbalanced=True)

    def test_dist_transformer_opt_speed(self):
        self._dist_transformer(optimize="speed")

    def test_dist_transformer_opt_memory(self):
        self._dist_transformer(optimize="memory")

    def _dist_transformer(self, optimize="memory", unbalanced=False):
        """
        Run the full-batch input over a centralized transformer layer. Next, run the data-parallel version
        (batch split over 4 ranks) over the DistributedTransformerLayer, whose weights are set to the
        same values as the centralized version. Then assert that the layer output, input gradient, and
        parameter gradients are the same across the 2 versions.
        """

        smp.init(
            {
                "pipeline_parallel_degree": 1,
                "tensor_parallel_degree": 4,
                "ddp": True,
                "optimize": optimize,
            }
        )
        torch.cuda.set_device(smp.local_rank())
        os.environ["SMP_USE_HF_GELU"] = "1"

        batch_size = 40
        seq_length = 128
        if not unbalanced:
            config = TransformerConfig()
        else:
            config = TransformerConfig(
                hidden_size=845,
                num_attention_heads=13,
                attention_head_size=65,
                intermediate_size=3073,
                _precision_test=True,
            )

        torch.manual_seed(123)
        layer = TransformerLayer(config).to(torch.device("cuda"))
        dist_layer = DistributedTransformerLayer(**config.to_dict()).to(torch.device("cuda"))

        # (module, dist_module, which channels we are partitioning)
        layernorm_ptn = "output" if optimize == "memory" else None
        linear1_ptn = "input" if optimize == "memory" else "output"

        module_mapping = [
            (layer.attention.self.query, dist_layer.attention.query, linear1_ptn),
            (layer.attention.self.key, dist_layer.attention.key, linear1_ptn),
            (layer.attention.self.value, dist_layer.attention.value, linear1_ptn),
            (layer.attention.output.dense, dist_layer.attention.dense, "input"),
            (layer.intermediate.dense_act, dist_layer.output.dense1, linear1_ptn),
            (layer.output.dense, dist_layer.output.dense2, "input"),
            (layer.output.LayerNorm, dist_layer.output.layernorm, layernorm_ptn),
        ]

        if hasattr(layer.attention.output, "LayerNorm"):
            module_mapping.extend(
                [(layer.attention.output.LayerNorm, dist_layer.attention.layernorm, layernorm_ptn)]
            )

        # go through modules, equalize the weights
        for i, (mod, dist_mod, partition) in enumerate(module_mapping):
            # special treatment for query/key/value for speed and dense for all
            # since their splitted dimension has local_attention_size which is not generated by get_local_channels
            if (i < 3 and optimize == "speed") or i == 3:
                split_shapes = dist_layer.attention.all_local_attention_sizes
            else:
                split_shapes = None
            equalize_linear_weights(mod, dist_mod, partition=partition, split_shapes=split_shapes)

        x, x_slice = self.create_inputs_with_global_shape(
            [batch_size, seq_length, config.hidden_size]
        )
        attention_mask, attention_mask_slice = self.create_attention_mask(batch_size, seq_length)

        # test forward pass
        local_output, _ = layer((x, attention_mask))

        # reset the seed to make dropout realization is the same
        dist_output, _ = dist_layer((x_slice, attention_mask_slice))
        local_output_slice = local_output.narrow(
            0, batch_size // smp.tp_size() * smp.tp_rank(), batch_size // smp.tp_size()
        )
        self.assertTrue(torch.allclose(local_output_slice.cpu(), dist_output.cpu(), atol=ATOL))

        ## test backward pass
        loss_fn = torch.nn.CrossEntropyLoss().to(torch.device("cuda", smp.local_rank()))
        target = torch.randint(low=0, high=3, size=(batch_size, config.hidden_size)).to(
            torch.device("cuda", smp.local_rank())
        )
        local_bs = batch_size // smp.tp_size()
        target_slice = target[local_bs * smp.tp_rank() : local_bs * (smp.tp_rank() + 1)]

        y = loss_fn(local_output, target)
        y_dist = loss_fn(dist_output, target_slice) / float(smp.tp_size())

        y.backward()
        y_dist.backward()

        x_grad_slice = x.grad.narrow(
            0, batch_size // smp.tp_size() * smp.tp_rank(), batch_size // smp.tp_size()
        )
        self.assertTrue(torch.allclose(x_grad_slice.cpu(), x_slice.grad.cpu(), atol=ATOL))

        for i, (mod, dist_mod, ptn) in enumerate(module_mapping):
            if (i < 3 and optimize == "speed") or i == 3:
                split_shapes = dist_layer.attention.all_local_attention_sizes
            else:
                split_shapes = None

            if hasattr(mod, "weight"):
                weight_grad_slice = get_weight_slice(
                    mod.weight.grad, ptn, split_shapes=split_shapes
                )

                self.assertTrue(
                    torch.allclose(weight_grad_slice.cpu(), dist_mod.weight.grad.cpu(), atol=ATOL)
                )

            if hasattr(dist_mod, "bias") and dist_mod.bias is not None:
                if ptn == "input":
                    bias_grad_slice = mod.bias.grad
                elif ptn == "output":
                    bias_grad_slice = get_weight_slice(
                        mod.bias.grad, ptn, split_shapes=split_shapes
                    )
                else:
                    bias_grad_slice = mod.bias.grad

                self.assertTrue(
                    torch.allclose(
                        bias_grad_slice.cpu(), dist_mod.bias.grad.cpu(), atol=ATOL, rtol=1e-4
                    )
                )

    def test_dist_linear_unbalanced(self):
        self.test_dist_linear(in_features=2047)

    def test_dist_linear(self, in_features=2048):
        """
        Run the full-batch input over a centralized nn.Linear layer. Next, run the data-parallel version
        (batch split over 4 ranks) over the DistributedLinear, whose weights are set to the
        same values as the centralized version. Then assert that the layer output, input gradient, and
        parameter gradients are the same across the 2 versions.
        """

        smp.init({"pipeline_parallel_degree": 1, "tensor_parallel_degree": 4, "ddp": True})
        torch.cuda.set_device(smp.local_rank())

        batch_size = 40
        out_features = 4096

        torch.manual_seed(123)
        lin = nn.Linear(in_features, out_features).to(torch.device("cuda"))
        dist_lin = DistributedLinear(in_features, out_features).to(torch.device("cuda"))

        # equalize the weights
        equalize_linear_weights(lin, dist_lin, partition="input")

        x, x_slice = self.create_inputs_with_global_shape([batch_size, in_features])

        # test forward pass
        local_output = lin(x)
        dist_output = dist_lin(x_slice)
        local_output_slice = local_output.narrow(
            0, batch_size // smp.tp_size() * smp.tp_rank(), batch_size // smp.tp_size()
        )
        self.assertTrue(torch.allclose(local_output_slice.cpu(), dist_output.cpu(), atol=ATOL))

        # test backward pass
        weights = torch.ones(local_output.numel(), device=torch.device("cuda"))
        weights_dist = torch.ones(dist_output.numel(), device=torch.device("cuda"))

        y = torch.dot(torch.flatten(local_output), weights)
        y_dist = torch.dot(torch.flatten(dist_output), weights_dist)
        y.backward()
        y_dist.backward()

        x_grad_slice = x.grad.narrow(
            0, batch_size // smp.tp_size() * smp.tp_rank(), batch_size // smp.tp_size()
        )
        self.assertTrue(torch.allclose(x_grad_slice.cpu(), x_slice.grad.cpu(), atol=ATOL))

        local_in_features = get_local_channels(in_features)
        start = get_start_pos_for_slicing(in_features)
        weight_grad_slice = lin.weight.grad.narrow(1, start, local_in_features)

        self.assertTrue(
            torch.allclose(weight_grad_slice.cpu(), dist_lin.weight.grad.cpu(), atol=ATOL)
        )

        if smp.tp_rank() == 0:
            self.assertTrue(
                torch.allclose(lin.bias.grad.cpu(), dist_lin.bias.grad.cpu(), atol=ATOL)
            )

    def test_dist_embedding(self, embedding_dim=1024):
        def check_embedding(
            padding_idx=None,
            scale_grad_by_freq=False,
            max_norm=None,
            norm_type=2.0,
            sparse=False,
            _weight=None,
            num_embeddings=30528,
            embedding_dim=embedding_dim,
        ):
            smp.init({"pipeline_parallel_degree": 1, "tensor_parallel_degree": 4, "ddp": True})
            torch.cuda.set_device(smp.local_rank())

            num_embeddings = num_embeddings
            embedding_dim = embedding_dim
            batch_size = 32
            input_shape = (batch_size, 4)

            torch.manual_seed(123)
            embedding = nn.Embedding(
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                scale_grad_by_freq=scale_grad_by_freq,
                max_norm=max_norm,
                norm_type=norm_type,
                sparse=sparse,
                _weight=_weight,
            ).to(torch.device("cuda"))
            dist_embedding = DistributedEmbedding(
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                scale_grad_by_freq=scale_grad_by_freq,
                max_norm=max_norm,
                norm_type=norm_type,
                sparse=sparse,
                _weight=_weight,
            ).to(torch.device("cuda"))

            equalize_embedding_weights(embedding, dist_embedding)

            x, x_slice = self.create_indices_with_global_shape(
                input_shape, low=0, high=(num_embeddings - 1)
            )

            # test forward pass
            local_output = embedding(x)
            dist_output = dist_embedding(x_slice)
            local_output_slice = local_output.narrow(
                0, (batch_size // smp.tp_size()) * smp.tp_rank(), batch_size // smp.tp_size()
            ).contiguous()
            self.assertTrue(
                torch.allclose(local_output_slice.cpu(), dist_output.cpu(), atol=1e-5, rtol=1e-4)
            )
            weights = torch.ones(local_output.numel(), device=torch.device("cuda"))
            weights_dist = torch.ones(dist_output.numel(), device=torch.device("cuda"))
            y = torch.dot(torch.flatten(local_output), weights)
            y_dist = torch.dot(torch.flatten(dist_output), weights_dist)
            y.backward()
            y_dist.backward()
            slice_size = get_local_channels(embedding_dim)
            start = get_start_pos_for_slicing(embedding_dim)
            slice_weight = embedding.weight.narrow(-1, start, slice_size).contiguous()
            if not sparse:
                weight_grad_slice = embedding.weight.grad.narrow(-1, start, slice_size)
            else:
                weight_grad_slice = embedding.weight.grad.to_dense().narrow(-1, start, slice_size)

            if not sparse:
                self.assertTrue(
                    torch.allclose(
                        weight_grad_slice.cpu(),
                        dist_embedding.weight.grad.cpu(),
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )
            else:
                self.assertTrue(
                    torch.allclose(
                        weight_grad_slice.cpu(),
                        dist_embedding.weight.grad.cpu().to_dense(),
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )

                self.assertTrue(
                    torch.allclose(
                        slice_weight.cpu(), dist_embedding.weight.cpu(), atol=1e-5, rtol=1e-5
                    )
                )

        check_embedding()
        check_embedding(padding_idx=1)
        check_embedding(scale_grad_by_freq=True)
        check_embedding(sparse=True)
        # Not supported
        # check_embedding(num_embeddings=30528, embedding_dim=1024, _weight=torch.randn(30528, 1024))
        # check_embedding(max_norm=1.0)
        # check_embedding(max_norm=2.0)
        # check_embedding(max_norm=1.0, norm_type=2.0)

    def test_dist_embedding_unbalanced(self):
        self.test_dist_embedding(embedding_dim=1023)

    def test_dist_layer_norm_unbalanced(self):
        self.test_dist_layer_norm(num_channels=81)

    def test_dist_layer_norm(self, num_channels=80):
        smp.init({"tensor_parallel_degree": 4, "pipeline_parallel_degree": 1, "ddp": True})

        local_channels = get_local_channels(num_channels)
        batch_size = 32

        torch.manual_seed(42)

        x = torch.randn(batch_size, num_channels).to(torch.device("cuda", smp.local_rank()))
        start = get_start_pos_for_slicing(num_channels)
        x2 = x.narrow(1, start, local_channels).contiguous().detach()

        x.requires_grad = True
        x2.requires_grad = True

        l = FusedLayerNorm(num_channels)
        l_split = DistributedLayerNorm(num_channels)

        l.to(torch.device("cuda", smp.local_rank()))
        l_split.to(torch.device("cuda", smp.local_rank()))

        y = l(x)
        y2 = l_split(x2)

        def slice_tensor(tensor):
            if tensor.dim() > 1:
                return tensor.narrow(1, start, local_channels)
            else:
                return tensor.narrow(0, start, local_channels)

        torch.allclose(slice_tensor(y).cpu(), y2.cpu(), atol=ATOL)

        torch.autograd.backward(
            y, torch.ones(batch_size, num_channels, device=torch.device("cuda", smp.local_rank()))
        )
        torch.autograd.backward(
            y2,
            torch.ones(batch_size, local_channels, device=torch.device("cuda", smp.local_rank())),
        )

        torch.allclose(slice_tensor(x.grad).cpu(), x2.grad.cpu(), atol=ATOL)

        torch.allclose(slice_tensor(l.weight.grad).cpu(), l_split.weight.grad.cpu(), atol=ATOL)
        torch.allclose(slice_tensor(l.bias.grad).cpu(), l_split.bias.grad.cpu(), atol=ATOL)


if __name__ == "__main__":
    unittest.main()
