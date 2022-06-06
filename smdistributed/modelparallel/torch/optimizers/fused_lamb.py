# Third Party
import torch
from apex.multi_tensor_apply import multi_tensor_applier
from apex.optimizers import FusedLAMB as ApexFusedLAMB

# First Party
from smdistributed.modelparallel.backend.collectives import CommGroup
from smdistributed.modelparallel.torch.optimizers.utils import get_device
from smdistributed.modelparallel.torch.state_mod import state


class FusedLAMB(ApexFusedLAMB):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad is False or p.grad is None:
                    continue
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError("FusedLAMB only support fp16 and fp32.")

        device = get_device(self.param_groups)
        g_norm_32, g_norm_16 = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = multi_tensor_applier(
                self.multi_tensor_l2norm, self._dummy_overflow_buf, [g_all_32], False
            )[0]
        if len(g_all_16) > 0:
            g_norm_16 = multi_tensor_applier(
                self.multi_tensor_l2norm, self._dummy_overflow_buf, [g_all_16], False
            )[0]
        # blend two grad norms to get global grad norm
        local_grad_norm = multi_tensor_applier(
            self.multi_tensor_l2norm, self._dummy_overflow_buf, [[g_norm_32, g_norm_16]], False
        )[0]

        # allgather the grad norms across pp_ranks
        local_grad_norms = state.comm.allgather(
            local_grad_norm.to(torch.device("cpu")), group=CommGroup.PP_GROUP
        )
        global_grad_norm = torch.norm(torch.tensor(local_grad_norms)).to(torch.device("cuda"))

        max_grad_norm = self.defaults["max_grad_norm"]

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]
            grad_averaging = 1 if group["grad_averaging"] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group["params"]:
                if p.requires_grad is False or p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedLAMB does not support sparse gradients, please consider SparseAdam instead"
                    )

                param_state = self.state[p]
                # State initialization
                if len(param_state) == 0:
                    # Exponential moving average of gradient values
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(param_state["exp_avg"])
                    v_16.append(param_state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(param_state["exp_avg"])
                    v_32.append(param_state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedLAMB only support fp16 and fp32.")

            if len(g_16) > 0:
                multi_tensor_applier(
                    self.multi_tensor_lamb,
                    self._dummy_overflow_buf,
                    [g_16, p_16, m_16, v_16],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    bias_correction,
                    group["weight_decay"],
                    grad_averaging,
                    self.adam_w_mode,
                    global_grad_norm,
                    max_grad_norm,
                    self.use_nvlamb,
                )
            if len(g_32) > 0:
                multi_tensor_applier(
                    self.multi_tensor_lamb,
                    self._dummy_overflow_buf,
                    [g_32, p_32, m_32, v_32],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    bias_correction,
                    group["weight_decay"],
                    grad_averaging,
                    self.adam_w_mode,
                    global_grad_norm,
                    max_grad_norm,
                    self.use_nvlamb,
                )

        return loss
