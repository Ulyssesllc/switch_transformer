import argparse
import torch
from torch.utils.data import DataLoader

from train import SwitchClassifier
from dataset import get_toy_dataset, get_hf_dataset

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            texts = [""] * len(y)
            return input_ids, attention_mask, labels, texts

    else:
        if AutoTokenizer is None:
            raise ImportError("Please install transformers: pip install transformers")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        dataset, num_classes = get_hf_dataset(
            args.dataset,
            tokenizer=tokenizer,
            max_len=args.seq_len,
            split="test",
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
            texts = [item.get("text", "") for item in batch]
            return input_ids, attention_mask, labels, texts

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    if hasattr(dataset, "lengths") and dataset.lengths is not None:
        lengths = dataset.lengths.to(torch.float32)
        avg_len = float(lengths.mean().item())
        max_len = float(lengths.max().item())
        pct_full = float((lengths == args.seq_len).float().mean().item()) * 100.0
        print(
            f"Dataset stats -> avg_len={avg_len:.1f}, max_len={max_len:.1f}, pct_at_max={pct_full:.1f}%"
        )

    # model
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

    if args.ckpt is not None:
        print(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            if args.use_ema and "ema_state_dict" in checkpoint:
                state_dict = checkpoint["ema_state_dict"]
                print("Loaded EMA weights from checkpoint")
            else:
                state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    for batch in dataloader:
        input_ids, attention_mask, labels, texts = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits, _ = model(input_ids, attention_mask=attention_mask, is_training=False)
        preds = logits.argmax(dim=-1)

        for i in range(input_ids.size(0)):
            label = labels[i].item()
            pred = preds[i].item()
            if args.dataset == "toy":
                print(f"Sample {i}: label={label}, pred={pred}")
            else:
                text = texts[i]
                print(f"Text: {text[:60]}... | True={label}, Pred={pred}")
        break  # chỉ in batch đầu tiên


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with Switch Transformer Classifier"
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

    # inference
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="When loading trainer checkpoints, prefer EMA weights if available",
    )

    args = parser.parse_args()
    run_inference(args)
