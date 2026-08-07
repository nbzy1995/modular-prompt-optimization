"""
Phase-2 figures, built from the judge-graded records in analysis/.

Produces:
  1. reliability.png       -- calibration / reliability diagram (stated confidence vs
                              empirical accuracy) for selected conditions, with ECE.
  2. coverage_accuracy.png -- accuracy-vs-coverage frontier from the abstain@<T> sweep
                              (and any 'none'/'abstention' anchors present).

Run analyze.py first (it writes analysis/*_graded.json). Then:
    ./.venv-spike/bin/python experiments/phase2/plots.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_ROOT = os.path.join(HERE, "analysis")
FIG_DIR = None  # set per-model in main()
FNAME_RE = re.compile(r"(.+?)_seed(\d+)_graded\.json$")

# conditions to overlay on the reliability diagram (whichever exist)
RELIABILITY_CONDS = ["none", "abstention", "cot", "expert_persona"]


def load_pooled(analysis_dir):
    """cond -> list of records pooled across seeds."""
    pooled = defaultdict(list)
    for f in glob.glob(os.path.join(analysis_dir, "*_graded.json")):
        m = FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        cond = m.group(1)
        pooled[cond].extend(json.load(open(f)))
    return pooled


def reliability_rows(records, n_bins=10):
    confs, corr = [], []
    for r in records:
        if r.get("abstained") or r.get("confidence") is None:
            continue
        confs.append(r["confidence"] / 100.0)
        corr.append(1.0 if r["grade"] == "CORRECT" else 0.0)
    confs, corr = np.array(confs), np.array(corr)
    if len(confs) == 0:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ece = [], [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bc, ba = confs[mask].mean(), corr[mask].mean()
        xs.append(bc); ys.append(ba)
        ece += (mask.sum() / len(confs)) * abs(bc - ba)
    return {"xs": xs, "ys": ys, "ece": ece, "n": len(confs),
            "mean_conf": float(confs.mean()), "acc": float(corr.mean())}


def plot_reliability(pooled):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    for cond in RELIABILITY_CONDS:
        if cond not in pooled:
            continue
        rr = reliability_rows(pooled[cond])
        if not rr:
            continue
        plt.plot(rr["xs"], rr["ys"], "o-", label=f"{cond} (ECE={rr['ece']:.2f}, conf={rr['mean_conf']*100:.0f}%, acc={rr['acc']*100:.0f}%)")
    plt.xlabel("Stated confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram — SimpleQA")
    plt.legend(fontsize=8, loc="upper left")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "reliability.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def plot_coverage_accuracy(pooled):
    """Frontier from abstain@<T> conditions (+ none/abstention anchors)."""
    pts = []
    for cond, recs in pooled.items():
        thr = None
        if cond == "none":
            thr = 0
        elif cond.startswith("abstain@"):
            thr = int(cond.split("@")[1])
        elif cond == "abstention":
            thr = 999  # unconditional, plotted separately
        else:
            continue
        n = len(recs)
        n_abs = sum(1 for r in recs if r.get("abstained") or r["grade"] == "NOT_ATTEMPTED")
        attempted = [r for r in recs if not (r.get("abstained") or r["grade"] == "NOT_ATTEMPTED")]
        att_acc = np.mean([1.0 if r["grade"] == "CORRECT" else 0.0 for r in attempted]) if attempted else 0
        coverage = len(attempted) / n if n else 0
        pts.append((thr, cond, coverage, att_acc))
    if not pts:
        print("  (no sweep/anchor conditions found for coverage curve)")
        return None
    pts.sort(key=lambda x: x[0])
    sweep = [p for p in pts if p[0] < 999]
    plt.figure(figsize=(6, 5))
    if sweep:
        cov = [p[2] for p in sweep]; acc = [p[3] for p in sweep]
        plt.plot(cov, acc, "o-", label="abstain@threshold sweep")
        for thr, cond, c, a in sweep:
            plt.annotate(f"{'base' if thr==0 else f'@{thr}'}", (c, a),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    for thr, cond, c, a in pts:
        if thr == 999:
            plt.scatter([c], [a], marker="s", s=60, label="abstention (unconditional)", zorder=5)
    plt.xlabel("Coverage (fraction of questions answered)")
    plt.ylabel("Accuracy on answered questions")
    plt.title("Accuracy–coverage frontier — SimpleQA")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "coverage_accuracy.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aiml-gpt-4o",
                    help="which model's analysis/<model>/ dir to plot")
    args = ap.parse_args()
    analysis_dir = os.path.join(ANALYSIS_ROOT, args.model)
    global FIG_DIR
    FIG_DIR = os.path.join(analysis_dir, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)
    pooled = load_pooled(analysis_dir)
    if not pooled:
        raise SystemExit(f"No *_graded.json in {analysis_dir}. Run analyze.py --model {args.model} first.")
    print(f"[{args.model}] conditions found:", ", ".join(sorted(pooled)))
    r = plot_reliability(pooled)
    print("saved", r)
    c = plot_coverage_accuracy(pooled)
    if c:
        print("saved", c)


if __name__ == "__main__":
    main()
