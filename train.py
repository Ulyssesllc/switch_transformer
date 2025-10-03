import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

from model.transformer import TransformerEncoder
from dataset import get_toy_dataset, get_hf_dataset

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from transformers.optimization import Adafactor, AdafactorSchedule
except ImportError:
    Adafactor = None
    AdafactorSchedule = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow_param = self.shadow.get(name)
            if shadow_param is None:
                self.shadow[name] = param.detach().clone()
                continue
            shadow_param.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return {name: tensor.clone() for name, tensor in self.shadow.items()}


class SwitchClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 256,
        num_experts: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        capacity_factor_eval: float = 2.0,
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
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

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_experts=num_experts,
            dropout=dropout,
            expert_dropout=expert_dropout,
            capacity_factor=capacity_factor,
            eval_capacity_factor=capacity_factor_eval,
            router_noise_eps=router_noise_eps,
            aux_loss_coef=aux_loss_coef,
            init_scale=init_scale,
            max_relative_distance=max_len,
            switch_dropout=switch_dropout,
            z_loss_coef=z_loss_coef,
            distributed_strategy=distributed_strategy,
            moe_process_group=moe_process_group,
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None, is_training=True):
        B, T = input_ids.size()
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        mask = (
            attention_mask.unsqueeze(1).unsqueeze(2)
            if attention_mask is not None
            else None
        )
        enc_out, aux_loss = self.encoder(x, mask=mask, is_training=is_training)

        if attention_mask is not None:
            mask_float = attention_mask.unsqueeze(-1).to(enc_out.dtype)
            denom = mask_float.sum(dim=1).clamp_min(1.0)
            pooled = (enc_out * mask_float).sum(dim=1) / denom
        else:
            pooled = enc_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, aux_loss

    def latest_routing_metrics(self):
        stats = []
        for layer in self.encoder.layers:
            ffn_stats = getattr(layer.ffn, "last_routing_stats", None)
            if ffn_stats:
                stats.append(ffn_stats)
        return stats


def train_classifier(args):
    # dataset
    if args.dataset == "toy":
        dataset = get_toy_dataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes,
        )
        num_classes = args.num_classes
        vocab_size = args.vocab_size
        dataset.lengths = torch.full((len(dataset),), args.seq_len, dtype=torch.long)

        def collate_fn(batch):
            X, y = zip(*batch)
            input_ids = torch.stack(X)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            labels = torch.tensor(y)
            return input_ids, attention_mask, labels

    else:
        if AutoTokenizer is None:
            raise ImportError("Please install transformers: pip install transformers")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        dataset, num_classes = get_hf_dataset(
            args.dataset,
            tokenizer=tokenizer,
            max_len=args.seq_len,
            split="train",
            limit=args.hf_limit,
        )
        vocab_size = tokenizer.vocab_size

        max_length = getattr(tokenizer, "model_max_length", None)
        if max_length and max_length < 1e9 and args.seq_len > max_length:
            print(
                f"Warning: seq_len {args.seq_len} exceeds tokenizer.model_max_length {max_length}."
            )

        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            if "attention_mask" in batch[0]:
                attention_mask = torch.stack([item["attention_mask"] for item in batch])
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            labels = torch.tensor([item["labels"] for item in batch])
            return input_ids, attention_mask, labels

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    if hasattr(dataset, "lengths") and dataset.lengths is not None:
        lengths = dataset.lengths.to(torch.float32)
        avg_len = float(lengths.mean().item())
        max_len = float(lengths.max().item())
        pct_full = float((lengths == args.seq_len).float().mean().item()) * 100.0
        print(
            f"Dataset stats -> avg_len={avg_len:.1f}, max_len={max_len:.1f}, pct_at_max={pct_full:.1f}%"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwitchClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        max_len=args.seq_len,
        dropout=args.dropout,
        expert_dropout=args.expert_dropout,
        capacity_factor=args.capacity_factor,
        capacity_factor_eval=args.capacity_factor_eval,
        router_noise_eps=args.router_noise_eps,
        aux_loss_coef=args.aux_loss_coef,
        init_scale=args.init_scale,
        switch_dropout=args.switch_dropout,
        z_loss_coef=args.z_loss_coef,
        distributed_strategy=args.distributed_strategy,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    use_amp = args.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    if Adafactor is None or AdafactorSchedule is None:
        raise ImportError(
            "Please install transformers>=4.0.0 to use the Adafactor optimizer: pip install transformers"
        )

    use_relative_step = args.lr is None
    optimizer = Adafactor(
        model.parameters(),
        lr=None if use_relative_step else args.lr,
        clip_threshold=args.adafactor_clip_threshold,
        relative_step=use_relative_step,
        scale_parameter=use_relative_step,
        warmup_init=use_relative_step,
        weight_decay=args.weight_decay,
    )
    if use_relative_step:
        scheduler = AdafactorSchedule(optimizer)
    elif args.warmup_steps > 0:
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / float(args.warmup_steps)),
        )
    else:
        scheduler = None

    ema = (
        ExponentialMovingAverage(model, args.ema_decay) if args.ema_decay > 0 else None
    )

    history_loss, history_acc = [], []
    routing_drop_sum = 0.0
    routing_load_std_sum = 0.0
    routing_capacity_sum = 0.0
    routing_records = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_acc = 0, 0
        batch_iter = (
            dataloader
            if tqdm is None
            else tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        )
        for batch in batch_iter:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits, aux_loss = model(input_ids, attention_mask=attention_mask)
                cls_loss = criterion(logits, labels)
                loss = cls_loss + aux_loss

            loss_value = loss.detach().item()

            if use_amp:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if ema is not None and global_step >= args.ema_start_step:
                ema.update(model)
            global_step += 1

            total_loss += loss_value * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            total_acc += (preds == labels).sum().item()

            layer_stats = model.latest_routing_metrics()
            for stat in layer_stats:
                if not stat:
                    continue
                routing_drop_sum += stat.get("tokens_dropped_frac", 0.0)
                expert_load = stat.get("expert_load")
                if expert_load is not None and expert_load.numel() > 0:
                    routing_load_std_sum += float(expert_load.float().std().item())
                capacity_util = stat.get("capacity_util")
                if capacity_util is not None and capacity_util.numel() > 0:
                    routing_capacity_sum += float(capacity_util.float().mean().item())
                routing_records += 1

            if tqdm is not None:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataset)
        avg_acc = total_acc / len(dataset)
        history_loss.append(avg_loss)
        history_acc.append(avg_acc)

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        if routing_records > 0:
            avg_drop = routing_drop_sum / routing_records
            avg_load_std = routing_load_std_sum / routing_records
            avg_capacity = routing_capacity_sum / routing_records
            print(
                f"Routing stats -> drop_frac={avg_drop:.4f}, load_std={avg_load_std:.4f}, capacity_util={avg_capacity:.4f}"
            )
        routing_drop_sum = 0.0
        routing_load_std_sum = 0.0
        routing_capacity_sum = 0.0
        routing_records = 0

    if ema is not None:
        model.ema_state_dict = ema.state_dict()

    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history_loss, marker="o")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(history_acc, marker="o")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.savefig("training_curve.png")
        print("Training curve saved as training_curve.png")
    else:
        print("Matplotlib not available; skipping training curve plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Switch Transformer Classifier with visualization"
    )

    # dataset
    parser.add_argument(
        "--dataset", type=str, default="toy", choices=["toy", "ag_news", "sst2"]
    )
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--hf_limit", type=int, default=None)

    # model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--expert_dropout", type=float, default=0.0)
    parser.add_argument("--capacity_factor", type=float, default=1.0)
    parser.add_argument("--capacity_factor_eval", type=float, default=2.0)
    parser.add_argument("--router_noise_eps", type=float, default=1e-2)
    parser.add_argument("--aux_loss_coef", type=float, default=1e-2)
    parser.add_argument("--init_scale", type=float, default=0.1)
    parser.add_argument("--switch_dropout", type=float, default=0.1)
    parser.add_argument("--z_loss_coef", type=float, default=1e-3)
    parser.add_argument(
        "--distributed_strategy",
        type=str,
        default="none",
        choices=["none", "all_to_all"],
        help="MoE routing strategy across devices",
    )

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Absolute learning rate (set to override Adafactor relative step)",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adafactor_clip_threshold", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument(
        "--use_amp", action="store_true", help="Enable torch.cuda.amp mixed precision"
    )
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--ema_start_step", type=int, default=100)

    args = parser.parse_args()
    train_classifier(args)
