"""
Cross-model summary figures (the paper's headline visuals), built from the
per-model graded records in analysis/<model>/.

Produces:
  1. combined_reliability.png -- 4 reliability curves (baseline 'none' condition,
     pooled across seeds) overlaid on one panel: older models hug the right edge
     (overconfident); 2026 frontier models track the diagonal.
  2. ece_by_model.png         -- ECE per model (baseline), split old vs frontier.

Run after analyze.py has produced analysis/<model>/*_graded.json for each model.
    ./.venv-spike/bin/python experiments/phase2/combined_figure.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_ROOT = os.path.join(HERE, "analysis")
OUT_DIR = os.path.join(ANALYSIS_ROOT, "_combined")

# (model dir, display label, era) in plotting order
MODELS = [
    ("aiml-gpt-4o", "GPT-4o (2024)", "older"),
    ("aiml-gemini-2.5-flash", "Gemini-2.5-flash", "older"),
    ("aiml-gpt-5.1-chat-latest", "GPT-5.1 (2026)", "frontier"),
    ("aiml-claude-sonnet-4-6", "Claude-Sonnet-4.6 (2026)", "frontier"),
]
COND = "none"  # baseline condition for the cross-model comparison


def reliability(model_dir, cond=COND, n_bins=10):
    recs = []
    for f in glob.glob(os.path.join(ANALYSIS_ROOT, model_dir, f"{cond}_seed*_graded.json")):
        recs.extend(json.load(open(f)))
    confs, corr = [], []
    for r in recs:
        if r.get("abstained") or r.get("confidence") is None:
            continue
        confs.append(r["confidence"] / 100.0)
        corr.append(1.0 if r["grade"] == "CORRECT" else 0.0)
    if not confs:
        return None
    confs, corr = np.array(confs), np.array(corr)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ece = [], [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.sum() == 0:
            continue
        bc, ba = confs[m].mean(), corr[m].mean()
        xs.append(bc); ys.append(ba)
        ece += (m.sum() / len(confs)) * abs(bc - ba)
    return {"xs": xs, "ys": ys, "ece": ece, "mean_conf": confs.mean(), "acc": corr.mean()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    colors = {"older": None, "frontier": None}
    styles = {"older": "--s", "frontier": "-o"}

    # ---- Fig 1: combined reliability ----
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k:", alpha=0.6, label="perfect calibration")
    eces = []
    for mdir, label, era in MODELS:
        rr = reliability(mdir)
        if not rr:
            print(f"  (no data for {mdir})"); continue
        eces.append((label, era, rr["ece"]))
        plt.plot(rr["xs"], rr["ys"], styles[era], linewidth=2, markersize=6,
                 label=f"{label} — ECE={rr['ece']:.2f}, conf={rr['mean_conf']*100:.0f}%")
    plt.xlabel("Stated confidence", fontsize=12)
    plt.ylabel("Empirical accuracy", fontsize=12)
    plt.title("Calibration across model generations (SimpleQA, baseline prompt)", fontsize=12)
    plt.legend(fontsize=9, loc="upper left")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    f1 = os.path.join(OUT_DIR, "combined_reliability.png")
    plt.savefig(f1, dpi=150); plt.close()

    # ---- Fig 2: ECE by model ----
    plt.figure(figsize=(7, 5))
    labels = [e[0] for e in eces]
    vals = [e[2] for e in eces]
    bar_colors = ["#d62728" if e[1] == "older" else "#2ca02c" for e in eces]
    plt.bar(range(len(labels)), vals, color=bar_colors)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=9)
    plt.ylabel("Expected Calibration Error (lower = better)", fontsize=11)
    plt.title("Older/cheaper models are severely overconfident;\n2026 frontier models are well-calibrated", fontsize=11)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color="#d62728", label="older/cheaper"),
                        Patch(color="#2ca02c", label="2026 frontier")], fontsize=9)
    plt.tight_layout()
    f2 = os.path.join(OUT_DIR, "ece_by_model.png")
    plt.savefig(f2, dpi=150); plt.close()

    print("saved", f1)
    print("saved", f2)
    print("\nECE (baseline 'none'):")
    for label, era, e in eces:
        print(f"  {label:<28} {e:.3f}  ({era})")


if __name__ == "__main__":
    main()
