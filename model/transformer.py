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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(
        self, relative_position: torch.Tensor
    ) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        ret = 0
        n = -relative_position
        if self.bidirectional:
            half = num_buckets // 2
            ret = (n < 0).long() * half
            n = n.abs()
            num_buckets = half
        else:
            n = torch.clamp(n, min=0)

        max_exact = num_buckets // 2
        is_small = n < max_exact
        large_val = max_exact + (
            (
                torch.log(n.float() / max_exact + 1e-6)
                / math.log(max_distance / max_exact)
            )
            * (num_buckets - max_exact)
        ).to(torch.long)
        large_val = torch.clamp(large_val, max=num_buckets - 1)
        ret = ret + torch.where(is_small, n.long(), large_val)
        return ret

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)
        return values.permute(2, 0, 1)  # [num_heads, q_len, k_len]


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

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ):
        B, T, D = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias.unsqueeze(0)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ):
        B, T_q, D = query.size()
        T_k = key_value.size(1)

        q = self.q_proj(query).view(B, T_q, self.num_heads, self.d_head)
        kv = self.kv_proj(key_value).view(B, T_k, 2, self.num_heads, self.d_head)
        k, v = kv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias.unsqueeze(0)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T_q, D)
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
        eval_capacity_factor: Optional[float] = None,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        init_scale: float = 0.1,
        num_relative_buckets: int = 32,
        max_relative_distance: int = 128,
        switch_dropout: float = 0.1,
        z_loss_coef: float = 1e-3,
        distributed_strategy: str = "none",
        moe_process_group=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.ln2 = RMSNorm(d_model)
        self.rel_pos_bias = RelativePositionBias(
            num_heads=num_heads,
            num_buckets=num_relative_buckets,
            max_distance=max_relative_distance,
            bidirectional=True,
        )
        self.ffn = SwitchFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            expert_dropout=expert_dropout,
            router_noise_eps=router_noise_eps,
            alpha=aux_loss_coef,
            init_scale=init_scale,
            switch_dropout=switch_dropout,
            z_loss_coef=z_loss_coef,
            distributed_strategy=distributed_strategy,
            process_group=moe_process_group,
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None, is_training: bool = True):
        x_norm = self.ln1(x)
        rel_bias = self.rel_pos_bias(x.size(1), x.size(1), x.device)
        attn_out = self.attn(x_norm, mask=mask, rel_pos_bias=rel_bias)
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
        eval_capacity_factor: Optional[float] = None,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        init_scale: float = 0.1,
        num_relative_buckets: int = 32,
        max_relative_distance: int = 128,
        switch_dropout: float = 0.1,
        z_loss_coef: float = 1e-3,
        distributed_strategy: str = "none",
        moe_process_group=None,
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
                    eval_capacity_factor=eval_capacity_factor,
                    router_noise_eps=router_noise_eps,
                    aux_loss_coef=aux_loss_coef,
                    init_scale=init_scale,
                    num_relative_buckets=num_relative_buckets,
                    max_relative_distance=max_relative_distance,
                    switch_dropout=switch_dropout,
                    z_loss_coef=z_loss_coef,
                    distributed_strategy=distributed_strategy,
                    moe_process_group=moe_process_group,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = RMSNorm(d_model)

    def forward(self, x, mask: Optional[torch.Tensor] = None, is_training: bool = True):
        total_aux = 0.0
        for layer in self.layers:
            x, aux = layer(x, mask=mask, is_training=is_training)
            total_aux = total_aux + aux
        x = self.final_ln(x)
        return x, total_aux


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int = 4,
        dropout: float = 0.1,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        eval_capacity_factor: Optional[float] = None,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        init_scale: float = 0.1,
        num_relative_buckets: int = 32,
        max_relative_distance: int = 128,
        switch_dropout: float = 0.1,
        z_loss_coef: float = 1e-3,
        distributed_strategy: str = "none",
        moe_process_group=None,
    ):
        super().__init__()
        self.self_ln = RMSNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.cross_ln = RMSNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout=dropout)

        self.ffn_ln = RMSNorm(d_model)
        self.self_rel_bias = RelativePositionBias(
            num_heads=num_heads,
            num_buckets=num_relative_buckets,
            max_distance=max_relative_distance,
            bidirectional=False,
        )
        self.cross_rel_bias = RelativePositionBias(
            num_heads=num_heads,
            num_buckets=num_relative_buckets,
            max_distance=max_relative_distance,
            bidirectional=True,
        )
        self.ffn = SwitchFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            expert_dropout=expert_dropout,
            router_noise_eps=router_noise_eps,
            alpha=aux_loss_coef,
            init_scale=init_scale,
            switch_dropout=switch_dropout,
            z_loss_coef=z_loss_coef,
            distributed_strategy=distributed_strategy,
            process_group=moe_process_group,
        )

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones((size, size), device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ):
        T = x.size(1)
        causal = self._causal_mask(T, x.device)
        if self_mask is not None:
            mask = self_mask.unsqueeze(1).unsqueeze(2).to(torch.bool) & causal
        else:
            mask = causal

        x_norm = self.self_ln(x)
        rel_bias = self.self_rel_bias(T, T, x.device)
        self_attn = self.self_attn(x_norm, mask=mask, rel_pos_bias=rel_bias)
        x = x + self.dropout(self_attn)

        x_norm = self.cross_ln(x)
        if encoder_mask is not None:
            enc_mask = encoder_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
        else:
            enc_mask = None
        cross_bias = self.cross_rel_bias(x.size(1), encoder_out.size(1), x.device)
        cross_attn = self.cross_attn(
            x_norm,
            encoder_out,
            mask=enc_mask,
            rel_pos_bias=cross_bias,
        )
        x = x + self.dropout(cross_attn)

        x_norm = self.ffn_ln(x)
        ffn_out, aux_loss = self.ffn(x_norm, is_training=is_training)
        x = x + self.dropout(ffn_out)

        return x, aux_loss


class TransformerDecoder(nn.Module):
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
        eval_capacity_factor: Optional[float] = None,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        init_scale: float = 0.1,
        num_relative_buckets: int = 32,
        max_relative_distance: int = 128,
        switch_dropout: float = 0.1,
        z_loss_coef: float = 1e-3,
        distributed_strategy: str = "none",
        moe_process_group=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_experts=num_experts,
                    dropout=dropout,
                    expert_dropout=expert_dropout,
                    capacity_factor=capacity_factor,
                    eval_capacity_factor=eval_capacity_factor,
                    router_noise_eps=router_noise_eps,
                    aux_loss_coef=aux_loss_coef,
                    init_scale=init_scale,
                    num_relative_buckets=num_relative_buckets,
                    max_relative_distance=max_relative_distance,
                    switch_dropout=switch_dropout,
                    z_loss_coef=z_loss_coef,
                    distributed_strategy=distributed_strategy,
                    moe_process_group=moe_process_group,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ):
        total_aux = 0.0
        for layer in self.layers:
            x, aux = layer(
                x,
                encoder_out,
                self_mask=self_mask,
                encoder_mask=encoder_mask,
                is_training=is_training,
            )
            total_aux = total_aux + aux
        x = self.final_ln(x)
        return x, total_aux


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_decoder_layers: Optional[int] = None,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_experts: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        eval_capacity_factor: Optional[float] = None,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
        tie_word_embeddings: bool = True,
        init_scale: float = 0.1,
        switch_dropout: float = 0.1,
        z_loss_coef: float = 1e-3,
        distributed_strategy: str = "none",
        moe_process_group=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else num_layers
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_experts=num_experts,
            dropout=dropout,
            expert_dropout=expert_dropout,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            router_noise_eps=router_noise_eps,
            aux_loss_coef=aux_loss_coef,
            init_scale=init_scale,
            max_relative_distance=max_len,
            switch_dropout=switch_dropout,
            z_loss_coef=z_loss_coef,
            distributed_strategy=distributed_strategy,
            moe_process_group=moe_process_group,
        )

        self.decoder = TransformerDecoder(
            num_layers=decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_experts=num_experts,
            dropout=dropout,
            expert_dropout=expert_dropout,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            router_noise_eps=router_noise_eps,
            aux_loss_coef=aux_loss_coef,
            init_scale=init_scale,
            max_relative_distance=max_len,
            switch_dropout=switch_dropout,
            z_loss_coef=z_loss_coef,
            distributed_strategy=distributed_strategy,
            moe_process_group=moe_process_group,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.max_len = max_len

    def forward(
        self,
        input_ids,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ):
        B, T = input_ids.size()
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        mask = (
            attention_mask.unsqueeze(1).unsqueeze(2)
            if attention_mask is not None
            else None
        )

        enc_out, aux_enc_loss = self.encoder(x, mask=mask, is_training=is_training)

        if decoder_input_ids is None:
            logits = self.lm_head(enc_out)
            return logits, aux_enc_loss

        dec_emb = self.token_emb(decoder_input_ids)
        dec_emb = self.dropout(dec_emb)

        dec_out, aux_dec_loss = self.decoder(
            dec_emb,
            enc_out,
            self_mask=decoder_attention_mask,
            encoder_mask=attention_mask,
            is_training=is_training,
        )
        logits = self.lm_head(dec_out)
        total_aux = aux_enc_loss + aux_dec_loss
        return logits, total_aux


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
