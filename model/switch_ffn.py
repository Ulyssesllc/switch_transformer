"""
switch_ffn.py

Switch Feed-Forward Network (SwitchFFN) implementation in PyTorch.
This is the Mixture-of-Experts (MoE) layer used inside Transformer blocks,
following the original Switch Transformer paper.

Features implemented:
- Top-1 expert routing
- Router with noise injection for exploration
- Capacity factor and token dropping for overflow
- Auxiliary load balancing loss (fi * Pi)
- Selective precision for router logits (float32 softmax)
- Truncated normal initialization with reduced scale (0.1x)
- Expert dropout


"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utility: truncated normal init similar to TensorFlow
# ---------------------------
def truncated_normal_(tensor, mean=0.0, std=1.0):
    with torch.no_grad():
        tmp = tensor.normal_(mean=mean, std=std)
        tmp = torch.clamp(tmp, min=mean - 2 * std, max=mean + 2 * std)
        tensor.copy_(tmp)
    return tensor


class SwitchFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        capacity_factor: float = 1.0,
        eval_capacity_factor: Optional[float] = None,
        expert_dropout: float = 0.0,
        router_noise_eps: float = 1e-2,
        alpha: float = 1e-2,
        init_scale: float = 0.1,
        switch_dropout: float = 0.1,
        z_loss_coef: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = (
            eval_capacity_factor
            if eval_capacity_factor is not None
            else capacity_factor
        )
        self.expert_dropout = expert_dropout
        self.router_noise_eps = router_noise_eps
        self.alpha = alpha
        self.init_scale = init_scale
        self.switch_dropout = nn.Dropout(switch_dropout)
        self.z_loss_coef = z_loss_coef

        # Router: token -> logits over experts
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert FFNs
        experts = []
        for _ in range(num_experts):
            experts.append(
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(expert_dropout),
                    nn.Linear(d_ff, d_model),
                )
            )
        self.experts = nn.ModuleList(experts)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                std = math.sqrt(1.0 / p.size(1)) * self.init_scale
                truncated_normal_(p, mean=0.0, std=std)

    def forward(self, x: torch.Tensor, is_training: bool = True):
        # Flatten to [T, d_model]
        orig_shape = x.shape
        tokens = x.view(-1, self.d_model)
        T = tokens.size(0)
        E = self.num_experts

        # Router logits with optional noise
        router_input = self.switch_dropout(tokens)
        router_logits = self.router(router_input)
        if is_training and self.router_noise_eps > 0:
            noise = torch.randn_like(router_logits) * self.router_noise_eps
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits.to(torch.float32), dim=-1)

        # Top-1 selection
        top1_prob, top1_idx = torch.max(router_probs, dim=-1)

        # Load-balancing loss
        with torch.no_grad():
            counts = torch.bincount(top1_idx, minlength=E).to(router_probs.dtype)
            fi = counts / T
        Pi = router_probs.mean(dim=0)
        aux_loss = self.alpha * E * torch.dot(fi.to(Pi.dtype), Pi)

        # z-loss to stabilize router logits
        if self.z_loss_coef > 0:
            z_loss = torch.square(torch.logsumexp(router_logits, dim=-1)).mean()
            aux_loss = aux_loss + self.z_loss_coef * z_loss

        # Capacity per expert
        capacity_factor = (
            self.capacity_factor if is_training else self.eval_capacity_factor
        )
        expert_capacity = int(math.ceil((T / float(E)) * capacity_factor))
        expert_capacity = max(expert_capacity, 1)

        # Dispatch tokens
        positions_in_expert = torch.zeros(E, dtype=torch.long, device=tokens.device)
        token_fits = torch.zeros(T, dtype=torch.bool, device=tokens.device)
        pos_in_bucket = torch.zeros(T, dtype=torch.long, device=tokens.device)

        for t in range(T):
            e = int(top1_idx[t])
            pos = positions_in_expert[e].item()
            if pos < expert_capacity:
                token_fits[t] = True
                pos_in_bucket[t] = pos
                positions_in_expert[e] += 1

        d = self.d_model
        expert_inputs = torch.zeros(
            (E, expert_capacity, d), device=tokens.device, dtype=tokens.dtype
        )
        idx_map = [[] for _ in range(E)]

        for t in range(T):
            if not token_fits[t]:
                continue
            e = int(top1_idx[t])
            pos = int(pos_in_bucket[t])
            expert_inputs[e, pos] = tokens[t]
            idx_map[e].append(t)

        expert_outputs = torch.zeros_like(expert_inputs)
        for e in range(E):
            used = positions_in_expert[e].item()
            if used == 0:
                continue
            out = self.experts[e](expert_inputs[e, :used, :])
            expert_outputs[e, :used, :] = out

        outputs_tokens = tokens.clone()
        for e in range(E):
            used = positions_in_expert[e].item()
            if used == 0:
                continue
            out = expert_outputs[e, :used, :]
            idxs = idx_map[e]
            probs = top1_prob[torch.tensor(idxs, device=tokens.device)].to(out.dtype)
            outputs_tokens[torch.tensor(idxs, device=tokens.device)] = (
                out * probs.unsqueeze(-1)
            )

        outputs = outputs_tokens.view(orig_shape)
        return outputs, aux_loss


if __name__ == "__main__":
    # Quick test
    B, T, D = 2, 8, 16
    ff = SwitchFFN(d_model=D, d_ff=32, num_experts=2)
    x = torch.randn(B, T, D)
    out, aux = ff(x)
    print(out.shape, aux.item())
