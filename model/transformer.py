"""
transformer.py

PyTorch implementation of a Transformer model that integrates a Switch-style MoE
feed-forward layer (SwitchFFN). This file is designed to follow the original
Switch Transformer architecture principles while remaining runnable on a single
GPU/CPU for experimentation.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .switch_ffn import SwitchFFN


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, T, D = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int = 4,
        dropout: float = 0.1,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SwitchFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            expert_dropout=expert_dropout,
            router_noise_eps=router_noise_eps,
            alpha=aux_loss_coef,
            init_scale=init_scale,
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None, is_training: bool = True):
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, mask=mask)
        x = x + self.dropout(attn_out)

        x_norm = self.ln2(x)
        ffn_out, aux_loss = self.ffn(x_norm, is_training=is_training)
        x = x + self.dropout(ffn_out)

        return x, aux_loss


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int = 4,
        dropout: float = 0.1,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_experts=num_experts,
                    dropout=dropout,
                    expert_dropout=expert_dropout,
                    capacity_factor=capacity_factor,
                    router_noise_eps=router_noise_eps,
                    aux_loss_coef=aux_loss_coef,
                    init_scale=init_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x, mask: Optional[torch.Tensor] = None, is_training: bool = True):
        total_aux = 0.0
        for layer in self.layers:
            x, aux = layer(x, mask=mask, is_training=is_training)
            total_aux = total_aux + aux
        x = self.final_ln(x)
        return x, total_aux


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_experts: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        tie_word_embeddings: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_experts=num_experts,
            dropout=dropout,
            expert_dropout=expert_dropout,
            capacity_factor=capacity_factor,
            router_noise_eps=router_noise_eps,
            aux_loss_coef=aux_loss_coef,
            init_scale=init_scale,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.max_len = max_len

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ):
        B, T = input_ids.size()
        pos = (
            torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, T)
        )
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        mask = (
            attention_mask.unsqueeze(1).unsqueeze(2)
            if attention_mask is not None
            else None
        )

        enc_out, aux_loss = self.encoder(x, mask=mask, is_training=is_training)
        logits = self.lm_head(enc_out)
        return logits, aux_loss


if __name__ == "__main__":
    vocab = 1000
    model = TransformerModel(
        vocab_size=vocab,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        num_experts=2,
        max_len=64,
    )
    input_ids = torch.randint(0, vocab, (2, 16))
    logits, aux = model(input_ids)
    print("logits", logits.shape, "aux_loss", aux.item())
