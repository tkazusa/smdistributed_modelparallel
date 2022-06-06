# Third Party
from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import Average, allreduce_async_, synchronize

# First Party
from smdistributed.modelparallel.torch.allreduce.reducer import GradReducer
from smdistributed.modelparallel.torch.utils import rmsg


class HorovodAllreducer(GradReducer):
    def __init__(
        self,
        named_parameters,
        grad_counter,
        overlapping_allreduce,
        average_grads_across_microbatches,
        num_microbatches,
        scaled_batch,
        tp_size,
        horovod_config=None,
    ):
        super(HorovodAllreducer, self).__init__(
            named_parameters,
            grad_counter,
            overlapping_allreduce,
            average_grads_across_microbatches,
            num_microbatches,
            scaled_batch,
            tp_size,
        )
        if horovod_config is None:
            horovod_config = {}
        self._validate_hvd_config(horovod_config)
        self._compression = horovod_config.get("compression", Compression.none)
        self.op = horovod_config.get("op", Average)
        self.gradient_predivide_factor = horovod_config.get("gradient_predivide_factor", 1.0)
        self._handles = {}

    def _validate_hvd_config(self, horovod_config):
        supported_configs = ["compression", "op", "gradient_predivide_factor"]
        if any([k not in supported_configs for k in horovod_config.keys()]):
            raise ValueError(
                f"Only the following horovod configs are supported {supported_configs}"
            )

    def _allreduce_grad_async(self, name, p):
        if p.grad is None:
            raise AssertionError(
                rmsg(f"Param {name} has None grad although it required grad, can not allreduce it")
            )

        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)

        if self.op == Average:
            # Split average operation across pre/postscale factors
            # C++ backend will apply additional 1 / size() factor to postscale_factor for op == Average.
            prescale_factor = 1.0 / self.gradient_predivide_factor
            postscale_factor = self.gradient_predivide_factor
        else:
            prescale_factor = 1.0
            postscale_factor = 1.0

        handle = allreduce_async_(
            tensor_compressed,
            name=name,
            op=self.op,
            prescale_factor=prescale_factor,
            postscale_factor=postscale_factor,
        )
        return handle, ctx

    def _synchronize_internal(self):
        for n, p in self.named_parameters.items():
            if p.requires_grad:
                if not self.overlapping_allreduce or p not in self._handles:
                    if not self.overlapping_allreduce:
                        if self.average_grads_across_microbatches:
                            if self.scaled_batch:
                                p.grad /= float(self.num_microbatches * self.tp_size)
                            else:
                                p.grad /= float(self.num_microbatches)
                    else:
                        pass
                        # this grad should only be 0 if grad hook wasn't called, as long as optimizer.zero_grad was called
                        # so it doesn't matter whether or not we divide by num_microbatches for params whose hook wasn't called
                        # ignoring divide by num microbatches

                    handle, ctx = self._allreduce_grad_async(n, p)
                    self._handles[p] = (handle, ctx)

        for p, (handle, ctx) in self._handles.items():
            assert (
                handle is not None
            ), "Handle should not be none, it is expected that handles only has params whose allreduce was called from the hook"
            output = synchronize(handle)
            p.grad.set_(self._compression.decompress(output, ctx))
        self._handles.clear()

    def _hook_internal(self, n, p, grads):
        handle, ctx = self._allreduce_grad_async(n, p)
        self._handles[p] = (handle, ctx)

    def _make_grad_hook(self, name, param):
        def hook(*grads):
            if self.enable_grad_counting:
                self.grad_counter.mark_grad_computed(name)
            if self.overlapping_allreduce and self._to_sync_grads:
                assert not param.grad.requires_grad, "grad should not require grad"
                if self.grad_counter.is_grad_ready(name):
                    # scale grad to account for different microbatches
                    # do it here just before allreduce
                    if self.average_grads_across_microbatches:
                        param.grad /= float(self.num_microbatches)
                    return self._hook_internal(name, param, grads)

            return None

        return hook
