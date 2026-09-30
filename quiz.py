"""Train the same tiny GPT three times with three learning rates and plot the curves.

    python quiz.py            # trains the 3 runs (~2-3 min each on a laptop CPU), writes quiz.png + answer.png
    python quiz.py --plot     # only re-plot from runs/*.json

quiz.png hides which learning rate made which curve (A/B/C). answer.png shows it.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import ROOT, train

LRS = [0.1, 0.003, 0.00003]      # too high, about right, too low
LABELS = {0.00003: "A", 0.1: "B", 0.003: "C"}   # the quiz order
COLORS = {"A": "#2f5d8a", "B": "#b5532a", "C": "#3d7a4a"}


def smooth(xs, k=4):
    """Centered moving average over up to 2k+1 steps (keeps spikes where they happened; the first point stays exact)."""
    out = []
    for i in range(len(xs)):
        r = min(k, i, len(xs) - 1 - i)
        good = [x for x in xs[i - r:i + r + 1] if x is not None]
        out.append(sum(good) / len(good) if good else None)
    return out


def plot(runs, reveal, path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for lr, log in sorted(runs.items(), key=lambda r: LABELS[r[0]]):
        name = LABELS[lr]
        label = f"{name}: lr = {lr:g}" if reveal else f"{name}"
        ax.plot(smooth(log["train"]), color=COLORS[name], lw=2.2, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (lower is better)")
    ax.set_ylim(1.4, 5.0)
    for lr, log in runs.items():  # mark any spike that leaves the chart
        peak = max(x for x in log["train"] if x is not None)
        if peak > 5.0:
            i = log["train"].index(peak)
            ax.annotate(f"{LABELS[lr]} spikes to {peak:.1f}", (i, 4.95), (i + 40, 4.7), color=COLORS[LABELS[lr]],
                        arrowprops=dict(arrowstyle="->", color=COLORS[LABELS[lr]]))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_title("Same model, same data, same seed. Only the learning rate differs." if not reveal
                 else "Too high: fast start, then it bounces back up. Too low: barely moves.")
    fig.tight_layout()
    fig.savefig(path)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="skip training, plot existing runs")
    args = ap.parse_args()
    runs = {}
    for lr in LRS:
        f = ROOT / "runs" / f"lr_{lr:g}.json"
        runs[lr] = json.loads(f.read_text()) if args.plot else train(lr)
    plot(runs, False, ROOT / "quiz.png")
    plot(runs, True, ROOT / "answer.png")
    print("\nfinal validation loss")
    for lr in LRS:
        print(f"  {LABELS[lr]}  lr {lr:<8g} {runs[lr]['val'][-1][1]:.3f}")


if __name__ == "__main__":
    main()
