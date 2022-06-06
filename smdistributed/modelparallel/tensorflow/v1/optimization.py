# Third Party
from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops, gen_control_flow_ops, gen_math_ops, math_ops
from tensorflow.python.training import optimizer

# First Party
from smdistributed.modelparallel import tensorflow as smp

# from tensorflow.contrib.mixed_precision import LossScaleOptimizer


class LossScaleOptimizer(optimizer.Optimizer):
    """
    A SMP version of tf.contrib.mixed_precision.LossScaleOptimizer. Its main function is to allgather the is_finite parameter
    across model partitions.
    For details check https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/mixed_precision/python/loss_scale_optimizer.py#L28
    """

    def __init__(self, opt, loss_scale_manager, name=None, use_locking=False):
        """
        Construct a loss scaling optimizer.
        Args:
          opt: The actual optimizer that will be used to compute and apply the
            gradients. Must be an implementation of the
            `tf.compat.v1.train.Optimizer` interface.
          loss_scale_manager: A LossScaleManager object.
        """
        self._opt = opt
        self._loss_scale_manager = loss_scale_manager
        if name is None:
            name = "LossScaleOptimizer{}".format(type(opt).__name__)
        super(LossScaleOptimizer, self).__init__(name=name, use_locking=use_locking)

    def compute_gradients(
        self,
        loss,
        var_list=None,
        gate_gradients=optimizer.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None,
    ):
        """Compute gradients. See base class `tf.compat.v1.train.Optimizer`."""
        loss_scale = self._loss_scale_manager.get_loss_scale()
        if context.executing_eagerly():

            def scaled_loss():
                loss_val = loss()
                return loss_val * math_ops.cast(loss_scale, loss_val.dtype.base_dtype)

        else:
            if callable(loss):
                loss_val = loss()
            else:
                loss_val = loss
            scaled_loss = loss_val * math_ops.cast(loss_scale, loss_val.dtype.base_dtype)
        grads_and_vars = self._opt.compute_gradients(
            scaled_loss,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss,
        )
        return self._down_scale(grads_and_vars, loss_scale)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients. See base class `tf.compat.v1.train.Optimizer`."""
        if self._loss_scale_manager != None:
            grads = [g for (g, _) in grads_and_vars]

            is_finite_grad = []
            for g in grads:
                is_finite_grad.append(math_ops.reduce_all(gen_math_ops.is_finite(g)))
            is_overall_finite = math_ops.reduce_all(is_finite_grad)

            if smp.pp_size() > 1:
                is_overall_finite = math_ops.reduce_all(
                    smp.allgather_tensor_ppgroup(is_overall_finite)
                )

            # Only update gradients when all grads are finite.
            def true_apply_gradients_fn():
                return self._opt.apply_gradients(grads_and_vars, global_step, name)

            update_vars = control_flow_ops.cond(
                is_overall_finite, true_apply_gradients_fn, gen_control_flow_ops.no_op
            )
            # Potentially adjust gradient scale in case of finite gradients.
            return control_flow_ops.group(
                update_vars, self._loss_scale_manager.update_loss_scale(is_overall_finite)
            )
        else:
            return self._opt.apply_gradients(grads_and_vars, global_step, name)

    def _down_scale(self, grads_vars, loss_scale):
        # Down scale grads by the loss_scale.
        gv = []
        inv_loss_scale = gen_math_ops.reciprocal(loss_scale)
        for g, v in grads_vars:
            if g is not None:
                gv.append((g * math_ops.cast(inv_loss_scale, g.dtype.base_dtype), v))
            else:
                gv.append((g, v))
        return gv
