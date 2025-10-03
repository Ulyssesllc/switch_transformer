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
import torch.distributed as dist


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
        distributed_strategy: str = "none",
        process_group: Optional[dist.ProcessGroup] = None,
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
        self.distributed_strategy = distributed_strategy.lower()
        self.process_group = process_group

        # Router: token -> logits over experts
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Distribution state
        self._setup_distribution()
        self._build_experts()
        self._init_weights()
        self.last_routing_stats = {}

    def _setup_distribution(self) -> None:
        self.world_size = 1
        self.rank = 0
        self.experts_per_rank = self.num_experts
        self.global_start = 0

        if (
            self.distributed_strategy == "all_to_all"
            and dist.is_available()
            and dist.is_initialized()
        ):
            group = self.process_group or dist.group.WORLD
            world_size = dist.get_world_size(group)
            rank = dist.get_rank(group)
            if self.num_experts % world_size != 0:
                raise ValueError(
                    "num_experts must be divisible by world_size when using all_to_all"
                )
            self.world_size = world_size
            self.rank = rank
            self.experts_per_rank = self.num_experts // world_size
            self.global_start = self.rank * self.experts_per_rank

    def _build_experts(self) -> None:
        expert_indices = range(
            self.global_start, self.global_start + self.experts_per_rank
        )
        experts = []
        for _ in expert_indices:
            experts.append(
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_ff),
                    nn.ReLU(),
                    nn.Dropout(self.expert_dropout),
                    nn.Linear(self.d_ff, self.d_model),
                )
            )
        self.experts = nn.ModuleList(experts)

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

        use_distributed = (
            self.distributed_strategy == "all_to_all"
            and self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        group = self.process_group or dist.group.WORLD if use_distributed else None

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

        # Routing bookkeeping
        positions_in_expert = torch.zeros(E, dtype=torch.long, device=tokens.device)
        token_fits = torch.zeros(T, dtype=torch.bool, device=tokens.device)
        pos_in_bucket = torch.zeros(T, dtype=torch.long, device=tokens.device)
        owner_ranks = (
            torch.zeros(T, dtype=torch.long, device=tokens.device)
            if use_distributed
            else None
        )

        for t in range(T):
            e = int(top1_idx[t])
            pos = positions_in_expert[e].item()
            if pos < expert_capacity:
                token_fits[t] = True
                pos_in_bucket[t] = pos
                positions_in_expert[e] += 1
                if use_distributed:
                    owner = e // self.experts_per_rank
                    owner_ranks[t] = owner
            elif use_distributed:
                owner_ranks[t] = -1

        d = self.d_model
        local_inputs = torch.zeros(
            (self.experts_per_rank, expert_capacity, d),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        position_info = [dict() for _ in range(self.experts_per_rank)]

        remote_tokens_per_rank = (
            [[] for _ in range(self.world_size)] if use_distributed else []
        )

        token_indices = torch.arange(T, device=tokens.device, dtype=torch.long)

        for t in range(T):
            if not token_fits[t]:
                continue
            global_expert = int(top1_idx[t])
            pos = int(pos_in_bucket[t].item())
            prob = top1_prob[t]
            if use_distributed:
                owner = int(owner_ranks[t].item())
            else:
                owner = self.rank

            if owner == self.rank:
                local_idx = global_expert - self.global_start
                local_inputs[local_idx, pos] = tokens[t]
                position_info[local_idx][pos] = (self.rank, token_indices[t], prob)
            elif use_distributed and owner >= 0:
                remote_tokens_per_rank[owner].append(
                    (tokens[t], prob, global_expert, pos, token_indices[t])
                )

        if use_distributed:
            send_counts = torch.zeros(
                self.world_size, dtype=torch.long, device=tokens.device
            )
            send_tokens_list = []
            send_probs_list = []
            send_meta_list = []

            for target in range(self.world_size):
                if target == self.rank:
                    continue
                entries = remote_tokens_per_rank[target]
                send_counts[target] = len(entries)
                for token_tensor, prob_tensor, g_expert, pos, orig_idx in entries:
                    send_tokens_list.append(token_tensor)
                    send_probs_list.append(prob_tensor)
                    send_meta_list.append([g_expert, pos, orig_idx.item(), self.rank])

            if send_tokens_list:
                send_tokens_buffer = torch.stack(send_tokens_list, dim=0)
                send_probs_buffer = torch.stack(send_probs_list, dim=0)
                send_meta_buffer = torch.tensor(
                    send_meta_list, dtype=torch.long, device=tokens.device
                )
            else:
                send_tokens_buffer = torch.zeros(
                    (0, d), device=tokens.device, dtype=tokens.dtype
                )
                send_probs_buffer = torch.zeros(
                    (0,), device=tokens.device, dtype=top1_prob.dtype
                )
                send_meta_buffer = torch.zeros(
                    (0, 4), dtype=torch.long, device=tokens.device
                )

            recv_counts = torch.zeros_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=group)

            total_send = int(send_counts.sum().item())
            total_recv = int(recv_counts.sum().item())

            recv_tokens_buffer = torch.zeros(
                (total_recv, d), device=tokens.device, dtype=tokens.dtype
            )
            recv_probs_buffer = torch.zeros(
                (total_recv,), device=tokens.device, dtype=top1_prob.dtype
            )
            recv_meta_buffer = torch.zeros(
                (total_recv, 4), dtype=torch.long, device=tokens.device
            )

            if total_send > 0 or total_recv > 0:
                dist.all_to_all_single(
                    recv_tokens_buffer,
                    send_tokens_buffer,
                    recv_counts.tolist(),
                    send_counts.tolist(),
                    group=group,
                )
                dist.all_to_all_single(
                    recv_probs_buffer,
                    send_probs_buffer,
                    recv_counts.tolist(),
                    send_counts.tolist(),
                    group=group,
                )
                dist.all_to_all_single(
                    recv_meta_buffer,
                    send_meta_buffer,
                    recv_counts.tolist(),
                    send_counts.tolist(),
                    group=group,
                )

            for i in range(total_recv):
                global_expert = int(recv_meta_buffer[i, 0].item())
                pos = int(recv_meta_buffer[i, 1].item())
                origin_index = int(recv_meta_buffer[i, 2].item())
                origin_rank = int(recv_meta_buffer[i, 3].item())
                local_idx = global_expert - self.global_start
                if 0 <= local_idx < self.experts_per_rank:
                    local_inputs[local_idx, pos] = recv_tokens_buffer[i]
                    position_info[local_idx][pos] = (
                        origin_rank,
                        torch.tensor(
                            origin_index, device=tokens.device, dtype=torch.long
                        ),
                        recv_probs_buffer[i],
                    )

        expert_outputs = torch.zeros_like(local_inputs)
        for local_idx in range(self.experts_per_rank):
            global_idx = self.global_start + local_idx
            used = min(
                expert_capacity,
                int(positions_in_expert[global_idx].item()),
            )
            if used == 0:
                continue
            out = self.experts[local_idx](local_inputs[local_idx, :used, :])
            expert_outputs[local_idx, :used, :] = out

        outputs_tokens = tokens.clone()
        remote_returns = (
            [[] for _ in range(self.world_size)] if use_distributed else None
        )

        for local_idx in range(self.experts_per_rank):
            global_idx = self.global_start + local_idx
            used = min(
                expert_capacity,
                int(positions_in_expert[global_idx].item()),
            )
            if used == 0:
                continue
            out = expert_outputs[local_idx, :used, :]
            info = position_info[local_idx]
            for pos, (origin_rank, token_index, prob_tensor) in info.items():
                if pos >= used:
                    continue
                scaled = out[pos] * prob_tensor.unsqueeze(-1)
                if origin_rank == self.rank or not use_distributed:
                    outputs_tokens[token_index] = scaled
                else:
                    remote_returns[origin_rank].append((token_index, scaled))

        if use_distributed:
            return_counts = torch.zeros(
                self.world_size, dtype=torch.long, device=tokens.device
            )
            return_tokens_list = []
            return_index_list = []
            for target in range(self.world_size):
                if target == self.rank:
                    continue
                entries = remote_returns[target]
                return_counts[target] = len(entries)
                for idx_tensor, token_tensor in entries:
                    return_index_list.append(idx_tensor)
                    return_tokens_list.append(token_tensor)

            return_recv_counts = torch.zeros_like(return_counts)
            dist.all_to_all_single(return_recv_counts, return_counts, group=group)

            total_return_send = int(return_counts.sum().item())
            total_return_recv = int(return_recv_counts.sum().item())

            if return_tokens_list:
                send_return_tokens = torch.stack(return_tokens_list, dim=0)
                send_return_indices = torch.stack(return_index_list, dim=0)
            else:
                send_return_tokens = torch.zeros(
                    (0, d), device=tokens.device, dtype=tokens.dtype
                )
                send_return_indices = torch.zeros(
                    (0,), device=tokens.device, dtype=torch.long
                )

            recv_return_tokens = torch.zeros(
                (total_return_recv, d), device=tokens.device, dtype=tokens.dtype
            )
            recv_return_indices = torch.zeros(
                (total_return_recv,), device=tokens.device, dtype=torch.long
            )

            if total_return_send > 0 or total_return_recv > 0:
                dist.all_to_all_single(
                    recv_return_tokens,
                    send_return_tokens,
                    return_recv_counts.tolist(),
                    return_counts.tolist(),
                    group=group,
                )
                dist.all_to_all_single(
                    recv_return_indices,
                    send_return_indices,
                    return_recv_counts.tolist(),
                    return_counts.tolist(),
                    group=group,
                )

            for i in range(total_return_recv):
                idx = recv_return_indices[i]
                outputs_tokens[idx] = recv_return_tokens[i]

        outputs = outputs_tokens.view(orig_shape)

        routed = token_fits.sum()
        dropped = T - routed
        capacity_util = positions_in_expert.to(torch.float32) / float(expert_capacity)
        expert_load = counts / T
        self.last_routing_stats = {
            "tokens_total": int(T),
            "tokens_routed": int(routed.item()),
            "tokens_dropped": int(dropped.item()),
            "tokens_dropped_frac": float(dropped.float().item() / max(T, 1)),
            "expert_load": expert_load.detach().cpu(),
            "capacity_util": capacity_util.detach().cpu(),
            "world_size": self.world_size,
        }

        return outputs, aux_loss


if __name__ == "__main__":
    # Quick test
    B, T, D = 2, 8, 16
    ff = SwitchFFN(d_model=D, d_ff=32, num_experts=2)
    x = torch.randn(B, T, D)
    out, aux = ff(x)
    print(out.shape, aux.item())
