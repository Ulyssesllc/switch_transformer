# Switch Transformer Text Classifier

Bring the routing magic of Switch Transformers to lightweight text classification experiments. This repo implements a compact Mixture-of-Experts transformer encoder that you can train on synthetic data or popular HuggingFace corpora, then run fast inference to inspect predictions.

---

## 🌟 Highlights
- **Switch-style MoE** encoder with configurable expert count, routing noise, and auxiliary load-balancing loss.
- **T5-inspired attention stack** using RMSNorm and relative position bias (no absolute positional embeddings).
- **Toy dataset generator** for instant experimentation without external downloads.
- **Plug-and-play HuggingFace integration** for `ag_news` and `sst2` (GLUE) with attention masks and raw text preservation.
- **Matplotlib visualizations**—automatic loss/accuracy curves saved as `training_curve.png` when the library is available.
- **CPU/GPU ready** using vanilla PyTorch APIs.

---

## 🚀 Quick start
1. **Clone & enter the project**
   ```bash
   git clone <your-fork-url>
   cd switch_transformer
   ```

2. **Create a Python environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
   pip install transformers datasets matplotlib
   ```
   > Skip `matplotlib` if you do not care about saved training plots.

4. **Run a smoke test on the synthetic dataset**
   ```bash
   python train.py --dataset toy --epochs 1 --batch_size 4 --num_samples 32
   python infer.py --dataset toy --num_samples 8 --batch_size 4
   ```
   Expect a `training_curve.png` plus printed predictions for the first inference batch.

---

## 🗂️ Project layout
```
switch_transformer/
├── dataset.py          # dataset builders for toy + HuggingFace corpora
├── train.py            # SwitchClassifier definition and training CLI
├── infer.py            # batch inference CLI (prints human-readable samples)
├── model/
│   ├── transformer.py  # transformer encoder with SwitchFFN blocks
│   └── switch_ffn.py   # Mixture-of-Experts feed-forward implementation
└── .gitignore
```

---

## 📥 Data sources
| Dataset flag | Source                                      | Notes |
|--------------|---------------------------------------------|-------|
| `toy`        | Random integer tokens + labels              | Fully in-memory, great for debugging. |
| `ag_news`    | HuggingFace [`ag_news`](https://huggingface.co/datasets/ag_news) | 4-class news topic classification. |
| `sst2`       | HuggingFace [`glue`, subset `sst2`](https://huggingface.co/datasets/glue) | Validation split used when `--split test` because GLUE test labels are hidden. |

- Ensure you have internet access when the loader downloads a dataset the first time.
- All HuggingFace samples retain their raw text via `item["text"]`, so inference prints meaningful snippets.

---

## 🧠 Model overview
- Multi-head self-attention encoder layers (`model/transformer.py`) with relative position bias and RMSNorm.
- Switch Feed-Forward Network (`model/switch_ffn.py`) with top-1 expert routing, capacity factor, expert dropout, SwitchDrop, and auxiliary losses (load-balance + z-loss).
- Token embeddings only (no absolute positions); temporal information flows through the relative bias module.
- Classification head that pools token representations with or without attention masks.

---

## 🏋️‍♂️ Training
Train on the toy dataset:
```bash
python train.py --dataset toy --epochs 5 --batch_size 32 --num_samples 2000
```

Train on AG News using a pretrained BERT tokenizer:
```bash
python train.py --dataset ag_news --seq_len 128 --batch_size 32 --epochs 3
```
> The script automatically downloads `bert-base-uncased` the first time; set `TRANSFORMERS_CACHE` if you want a custom cache directory.

Artifacts:
- `training_curve.png` (if matplotlib is installed).
- Standard output logs with loss/accuracy per epoch.
> Optimization uses Adafactor with a relative-step schedule (requires `transformers>=4.0.0`).

### Key CLI switches (training)
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `toy` | `toy`, `ag_news`, or `sst2`. |
| `--seq_len` | `16` | Maximum sequence length (used for padding/truncation). |
| `--d_model` | `128` | Transformer embedding dimension. |
| `--num_layers` | `2` | Number of Transformer blocks. |
| `--num_heads` | `4` | Attention heads per block. |
| `--num_experts` | `2` | Experts per SwitchFFN layer. |
| `--capacity_factor` | `1.0` | Capacity multiplier for token routing during training. |
| `--capacity_factor_eval` | `2.0` | Higher routing capacity used for evaluation/inference. |
| `--router_noise_eps` | `1e-2` | Additive Gaussian noise injected into router logits during training. |
| `--aux_loss_coef` | `1e-2` | Load-balancing loss coefficient. |
| `--switch_dropout` | `0.1` | Dropout applied to router inputs ("SwitchDrop"). |
| `--z_loss_coef` | `1e-3` | Router z-loss stabilizer. |
| `--label_smoothing` | `0.1` | Cross-entropy label smoothing factor. |
| `--grad_clip` | `1.0` | Global gradient clipping threshold (L2 norm). |
| `--weight_decay` | `0.0` | Weight decay applied by Adafactor. |
| `--adafactor_clip_threshold` | `1.0` | Internal Adafactor gradient clipping threshold. |
| `--lr` | `None` | Provide a value to disable relative-step Adafactor and use a fixed LR. |

---

## 🔎 Inference
Run inference on the toy dataset (random weights):
```bash
python infer.py --dataset toy --num_samples 16 --batch_size 4
```

Load a trained classifier checkpoint:
```bash
python infer.py --dataset ag_news --seq_len 128 --batch_size 8 --ckpt checkpoints/ag_news_best.pt
```

Outputs show true vs. predicted labels and the first 60 characters of the original text for HuggingFace samples. The script currently prints only the first batch for readability.

### CLI switches (inference)
| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | `None` | Path to a `.pt` state dict to restore before inference. |
| `--batch_size` | `4` | Number of samples per batch when iterating through the dataset. |
| `--num_samples` | `1000` | Toy dataset size if `--dataset toy`. |
| `--num_classes` | `3` | Toy dataset class count. |
| `--vocab_size` | `500` | Toy dataset vocabulary size. |
| `--capacity_factor_eval` | `2.0` | Router capacity multiplier during inference. |
| `--switch_dropout` | `0.1` | Router dropout kept consistent with training. |
| `--z_loss_coef` | `1e-3` | Router z-loss coefficient (match training). |

---

## 🧪 Validation commands
The repository was last smoke-tested with:
```bash
python train.py --dataset toy --epochs 1 --batch_size 2 --num_samples 8
python infer.py --dataset toy --num_samples 4 --batch_size 2
```
Both commands succeed on CPU, creating `training_curve.png` and printing prediction samples.

---

## 🧰 Troubleshooting
- **`ImportError: Please install transformers`** – install `transformers` and `datasets` via pip and re-run.
- **`ImportError: Please install datasets`** – the HuggingFace loader is optional; stick to `--dataset toy` or install `datasets`.
- **Matplotlib warnings** – the script switches to the `Agg` backend automatically; install `matplotlib` for curve exports or ignore the message.
- **Out-of-memory on GPU** – reduce `--batch_size`, `--seq_len`, or `--d_model`, or run on CPU.

---

## 🛣️ Roadmap ideas
- Save best-performing checkpoints automatically.
- Add evaluation metrics beyond accuracy (precision/recall/F1).
- Extend dataset loaders to additional GLUE tasks or custom CSV inputs.
- Provide notebook demos and hyperparameter sweeps.

---

## 🙌 Acknowledgements
- Inspired by the Google Research paper [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*](https://arxiv.org/abs/2101.03961).
- Built on the shoulders of [PyTorch](https://pytorch.org/) and the [HuggingFace Datasets & Transformers](https://huggingface.co/docs) ecosystem.

Happy experimenting! 🧪🔀
