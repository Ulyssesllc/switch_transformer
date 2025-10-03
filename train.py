import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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
        router_noise_eps: float = 1e-2,
        aux_loss_coef: float = 1e-2,
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

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None, is_training=True):
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

        if attention_mask is not None:
            mask_float = attention_mask.unsqueeze(-1).to(enc_out.dtype)
            denom = mask_float.sum(dim=1).clamp_min(1.0)
            pooled = (enc_out * mask_float).sum(dim=1) / denom
        else:
            pooled = enc_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, aux_loss


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

        def collate_fn(batch):
            X, y = zip(*batch)
            input_ids = torch.stack(X)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            labels = torch.tensor(y)
            return input_ids, attention_mask, labels

    else:
        if AutoTokenizer is None:
            raise ImportError("Please install transformers: pip install transformers")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset, num_classes = get_hf_dataset(
            args.dataset, tokenizer=tokenizer, max_len=args.seq_len, split="train"
        )
        vocab_size = tokenizer.vocab_size

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
        router_noise_eps=args.router_noise_eps,
        aux_loss_coef=args.aux_loss_coef,
        init_scale=args.init_scale,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history_loss, history_acc = [], []

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
            logits, aux_loss = model(input_ids, attention_mask=attention_mask)

            cls_loss = criterion(logits, labels)
            loss = cls_loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            total_acc += (preds == labels).sum().item()

            if tqdm is not None:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataset)
        avg_acc = total_acc / len(dataset)
        history_loss.append(avg_loss)
        history_acc.append(avg_acc)

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

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

    # model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--expert_dropout", type=float, default=0.0)
    parser.add_argument("--capacity_factor", type=float, default=1.0)
    parser.add_argument("--router_noise_eps", type=float, default=1e-2)
    parser.add_argument("--aux_loss_coef", type=float, default=1e-2)
    parser.add_argument("--init_scale", type=float, default=0.1)

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train_classifier(args)
