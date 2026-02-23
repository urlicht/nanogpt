# NanoGPT
Replicating a small, readable GPT-style language model training code in PyTorch.

This project includes:

- A decoder-only transformer (`NanoGPT`) with causal self-attention.
- Data preparation utilities that write token IDs to memmap-friendly `train.bin` / `val.bin`.
- A simple training loop with periodic eval, checkpointing, CSV logs, and TensorBoard metrics.
- KV caching

## Requirements

- Python `>=3.13`
- `uv` (recommended) or `pip`

Project dependencies are defined in `pyproject.toml`.

## Install

Using `uv`:

```bash
uv sync --dev
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## Run Tests

```bash
uv run pytest
```

## Project Layout

```text
src/nanogpt/model.py            # transformer model + generation
src/nanogpt/train.py            # train config, optimizer builder, training loop
src/nanogpt/data/processing.py  # tokenizers (char, tiktoken), vocab helpers
src/nanogpt/data/datasets.py    # dataset standardization to train.bin/val.bin
src/nanogpt/data/batch.py       # memmap batch sampling + iterable dataloader
tests/                          # unit tests
```

## Data Preparation

`prepare_dataset(...)` writes:

- `train.bin`
- `val.bin` (can be empty if no validation split)
- `metadata.json` (or yaml if configured)

### Example: Hugging Face dataset + GPT-2 tokenizer

```python
from nanogpt.data.processing import build_tiktoken_tokenizer
from nanogpt.data.datasets import prepare_dataset

tokenizer = build_tiktoken_tokenizer("gpt2")

prepare_dataset(
    input_source="roneneldan/TinyStories",
    tokenizer=tokenizer,
    output_dir="data/tinystories_gpt2",
    is_huggingface=True,
    text_field="text",
    streaming=True,
    val_fraction=0.01
)
```

### Example: local text file + char tokenizer

```python
from pathlib import Path
from nanogpt.data.processing import build_char_tokenizer
from nanogpt.data.datasets import prepare_dataset

text = Path("input.txt").read_text(encoding="utf-8")
tokenizer = build_char_tokenizer(text)

prepare_dataset(
    input_source="input.txt",
    tokenizer=tokenizer,
    output_dir="data/char_data",
    val_fraction=0.1,
)
```

Note: default output dtype is `uint16`. If your tokenizer vocab exceeds `65535`, use a larger dtype (for example `np.uint32`).

## Training

There is no CLI entrypoint yet; training is driven from Python:

```python
import torch
from nanogpt.train import TrainConfig, build_model, build_optimizer, train_loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = TrainConfig(
    out_dir="runs",
    name="tinystories_small",
    data_dir="data/tinystories_gpt2",
    device=device,
    n_vocab=50257,   # match tokenizer.n_vocab
    n_layer=4,
    n_head=8,
    d_emb=256,
    n_block=256,
    n_batch=32,
    max_iter=2000,
    eval_every=100,
    save_every=500,
)

model = build_model(cfg).to(cfg.device)
optimizer = build_optimizer(cfg, model)
train_loop(cfg, model, optimizer)
```

Training artifacts are written to:

```text
<out_dir>/<name>/<name>-YYYY-MM-DD-HHMM/
  checkpoints/ckpt_XXXXXX.pt
  losses.csv
  tb/
```

Start TensorBoard:

```bash
uv run tensorboard --logdir runs
```

## Inference / Generation

```python
import torch
from nanogpt.train import build_model
from nanogpt.data.processing import build_tiktoken_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("runs/tinystories_small/tinystories_small-YYYY-MM-DD-HHMM/checkpoints/ckpt_000500.pt", map_location=device)

cfg = ckpt["cfg"]
model = build_model(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = build_tiktoken_tokenizer("gpt2")
prompt = "Once upon a time"
x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

out = model.generate(x, max_n_token=50)
print(tokenizer.decode(out[0].tolist()))
```

Current behavior: generation keeps a rolling context window of size `n_block`. To keep full prompt+completion in one returned tensor, ensure `prompt_tokens + max_n_token <= n_block`.

## API Highlights

- `nanogpt.model.ModelConfig`
- `nanogpt.model.NanoGPT`
- `nanogpt.train.TrainConfig`
- `nanogpt.train.build_model`
- `nanogpt.train.build_optimizer`
- `nanogpt.train.train_loop`
- `nanogpt.data.datasets.prepare_dataset`
- `nanogpt.data.processing.build_char_tokenizer`
- `nanogpt.data.processing.build_tiktoken_tokenizer`
