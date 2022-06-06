# First Party
from smdistributed.modelparallel.torch.allreduce.reducer import GradReducer


class GradScaler(GradReducer):
    def __init__(
        self,
        named_parameters,
        grad_counter,
        overlapping_allreduce,
        average_grads_across_microbatches,
        num_microbatches,
        scaled_batch,
        tp_size,
    ):
        assert overlapping_allreduce is False
        super(GradScaler, self).__init__(
            named_parameters=named_parameters,
            grad_counter=grad_counter,
            overlapping_allreduce=False,
            average_grads_across_microbatches=average_grads_across_microbatches,
            num_microbatches=num_microbatches,
            scaled_batch=scaled_batch,
            tp_size=tp_size,
        )

    def _hook_internal(self, n, p, grads):
        pass

    def _synchronize_internal(self):
        for name, param in self.named_parameters.items():
            if (
                param.requires_grad
                and param.grad is not None
                and self.average_grads_across_microbatches
            ):
                if self.scaled_batch:
                    param.grad /= float(self.num_microbatches * self.tp_size)
                else:
                    param.grad /= float(self.num_microbatches)
