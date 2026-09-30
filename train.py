"""Train a tiny character-level GPT on Tiny Shakespeare with one learning rate.

    python train.py --lr 0.003

Everything except the learning rate is fixed: same data, same model, same seed,
same batches, same number of steps. The loss curve is saved to runs/lr_<lr>.json.
"""
import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

# ---------------------------------------------------------------- settings (only --lr changes between runs)
STEPS = 600
BATCH = 32
CONTEXT = 64
D_MODEL = 128
HEADS = 4
LAYERS = 3
EVAL_EVERY = 25
EVAL_BATCHES = 10
SEED = 1337


def load_text():
    path = ROOT / "data" / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print("downloading Tiny Shakespeare (1.1 MB)...")
        urllib.request.urlretrieve(DATA_URL, path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise SystemExit(f"{path} does not match the expected file; delete it and run again")
    return raw.decode("utf-8")


# ---------------------------------------------------------------- model: a small GPT
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab)
        self.register_buffer("mask", torch.triu(torch.ones(CONTEXT, CONTEXT, dtype=torch.bool), 1))

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t))
        for b in self.blocks:
            x = b(x, self.mask[:t, :t])
        return self.head(self.ln(x))


# ---------------------------------------------------------------- training
def batches(data, n, gen):
    """n random (input, target) batches, drawn from a fixed generator so every run sees the same ones."""
    for _ in range(n):
        i = torch.randint(len(data) - CONTEXT - 1, (BATCH,), generator=gen)
        x = torch.stack([data[j:j + CONTEXT] for j in i])
        y = torch.stack([data[j + 1:j + CONTEXT + 1] for j in i])
        yield x, y


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in batches(data, EVAL_BATCHES, gen)]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, stoi, itos, prompt="ROMEO:", n=120):
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]])
    gen = torch.Generator().manual_seed(0)
    for _ in range(n):
        logits = model(idx[:, -CONTEXT:])[0, -1]
        if not torch.isfinite(logits).all():
            return prompt + " <model weights are NaN>"
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
        idx = torch.cat([idx, nxt[None]], 1)
    return "".join(itos[i] for i in idx[0].tolist())


def train(lr, quiet=False):
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text])
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = TinyGPT(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    gen = torch.Generator().manual_seed(SEED)
    log = {"lr": lr, "steps": STEPS, "train": [], "val": []}
    log["val"].append([0, evaluate(model, val_data)])
    t0 = time.time()
    for step, (x, y) in enumerate(batches(train_data, STEPS, gen), 1):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log["train"].append(loss.item() if math.isfinite(loss.item()) else None)
        if step % EVAL_EVERY == 0:
            v = evaluate(model, val_data)
            log["val"].append([step, v if math.isfinite(v) else None])
            if not quiet:
                print(f"lr {lr:<8g} step {step:4d}  train {loss.item():.3f}  val {v:.3f}")
    log["seconds"] = round(time.time() - t0, 1)
    log["sample"] = sample(model, stoi, itos)
    out = ROOT / "runs" / f"lr_{lr:g}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(log))
    if not quiet:
        print(f"\nfinal val loss {log['val'][-1][1]}  ({log['seconds']} s)\n\n{log['sample']}\n\nsaved {out}")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True, help="learning rate for AdamW")
    train(ap.parse_args().lr)
