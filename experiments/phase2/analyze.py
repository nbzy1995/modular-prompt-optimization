"""
Phase-2 analysis: judge-grade generations + compute calibration metrics.

For each generation file in gen/:
  - Grade each non-abstained answer with the SimpleQA LLM judge
    (CORRECT / INCORRECT / NOT_ATTEMPTED). Abstentions count as NOT_ATTEMPTED.
  - Aggregate accuracy, hallucination rate, abstention rate, attempted accuracy.
  - Calibration over answers that carry a confidence score: ECE (10 bins),
    Brier score, mean confidence vs accuracy gap, and reliability-curve data.
  - Average across seeds and report mean +/- std per condition.

Usage:
    export AIML_API_KEY=...   (or in .env)
    ./.venv-spike/bin/python experiments/phase2/analyze.py
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
GEN_DIR = os.path.join(HERE, "gen")
ANALYSIS_ROOT = os.path.join(HERE, "analysis")

# reuse the judge from the spike
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "spike"))
from llm_judge_grade import get_client, grade_one, load_dotenv  # noqa: E402

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
# capture (model, condition, seed) from gen filenames: <model>_simpleqa_<cond>_seed<n>.json
FNAME_RE = re.compile(r"^(.+?)_simpleqa_(.+?)_seed(\d+)\.json$")


def ece_and_brier(confs, corrects, n_bins=10):
    """confs in [0,1], corrects in {0,1}. Returns (ECE, Brier, reliability rows)."""
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    if len(confs) == 0:
        return None, None, []
    brier = float(np.mean((confs - corrects) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = corrects[mask].mean()
        w = mask.sum() / len(confs)
        ece += w * abs(bin_conf - bin_acc)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                     "mean_conf": round(bin_conf, 3), "accuracy": round(bin_acc, 3)})
    return float(ece), brier, rows


def grade_records(records, client):
    """Grade each record via judge (abstentions = NOT_ATTEMPTED). Returns graded list."""
    graded = []
    for r in records:
        if r.get("abstained"):
            grade = "NOT_ATTEMPTED"
        else:
            grade = grade_one(client, JUDGE_MODEL, r["Question"], r["target"],
                              r.get("Final Answer Section") or r.get("Optimized Answer", ""))
        graded.append({**r, "grade": grade})
    return graded


def metrics_from_graded(graded):
    confs, corrects = [], []
    n_cor = n_inc = n_abs = 0
    for r in graded:
        grade = r["grade"]
        if grade == "CORRECT":
            n_cor += 1
        elif grade == "INCORRECT":
            n_inc += 1
        else:
            n_abs += 1
        # calibration uses answers that carry a confidence (incl. confident abstentions excluded:
        # an abstention has no factual answer to be right/wrong about)
        if r.get("confidence") is not None and not r.get("abstained"):
            confs.append(r["confidence"] / 100.0)
            corrects.append(1.0 if grade == "CORRECT" else 0.0)

    n = len(graded)
    attempted = n_cor + n_inc
    ece, brier, rel = ece_and_brier(confs, corrects)
    return {
        "n": n,
        "accuracy": n_cor / n if n else 0,
        "attempted_accuracy": n_cor / attempted if attempted else 0,
        "hallucination_rate": n_inc / attempted if attempted else 0,
        "abstention_rate": n_abs / n if n else 0,
        "ece": ece,
        "brier": brier,
        "mean_conf": float(np.mean(confs)) if confs else None,
        "n_calibration": len(confs),
        "reliability": rel,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aiml-gpt-4o",
                    help="only analyze gen files for this model id (keeps models separate)")
    args = ap.parse_args()

    load_dotenv()
    client = get_client()
    OUT_DIR = os.path.join(ANALYSIS_ROOT, args.model)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(GEN_DIR, f"{args.model}_simpleqa_*.json")))
    if not files:
        sys.exit(f"No generation files for model '{args.model}' in {GEN_DIR}. Run run_generation.py first.")

    by_cond = defaultdict(list)
    for f in files:
        m = FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        fmodel, cond, seed = m.group(1), m.group(2), m.group(3)
        graded_path = os.path.join(OUT_DIR, f"{cond}_seed{seed}_graded.json")
        if os.path.exists(graded_path):
            # resume: reuse previously graded records, no API calls
            print(f"reuse  {os.path.basename(graded_path)}", flush=True)
            graded = json.load(open(graded_path))
        else:
            print(f"grading {os.path.basename(f)} ...", flush=True)
            graded = grade_records(json.load(open(f)), client)
            json.dump(graded, open(graded_path, "w"), indent=2)
        by_cond[cond].append(metrics_from_graded(graded))

    # aggregate across seeds
    def agg(cond, key):
        vals = [m[key] for m in by_cond[cond] if m[key] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    order = ["none", "abstention", "expert_persona", "expert_persona+abstention",
             "cot", "cot+abstention", "cove", "cove+abstention"]
    conds = [c for c in order if c in by_cond] + [c for c in by_cond if c not in order]

    print("\n" + "=" * 100)
    print(f"{'condition':<28}{'acc':>8}{'halluc':>9}{'abstain':>9}{'ECE':>8}{'Brier':>8}{'mean_conf':>11}{'seeds':>7}")
    print("-" * 100)
    summary = {}
    for c in conds:
        acc, acc_s = agg(c, "accuracy")
        hal, _ = agg(c, "hallucination_rate")
        abs, _ = agg(c, "abstention_rate")
        ece, _ = agg(c, "ece")
        brier, _ = agg(c, "brier")
        mc, _ = agg(c, "mean_conf")
        ns = len(by_cond[c])
        def f(x, pct=True):
            if x is None:
                return "  n/a"
            return f"{x*100:.0f}%" if pct else f"{x:.3f}"
        print(f"{c:<28}{f(acc):>8}{f(hal):>9}{f(abs):>9}{f(ece,0):>8}{f(brier,0):>8}{f(mc):>11}{ns:>7}")
        summary[c] = {"accuracy": acc, "hallucination_rate": hal, "abstention_rate": abs,
                      "ece": ece, "brier": brier, "mean_conf": mc, "seeds": ns}
    print("=" * 100)
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    print(f"Saved per-file graded records and summary.json to {OUT_DIR}")
    print("\nReading: lower ECE = better calibrated. 'mean_conf' >> 'attempted_acc' = overconfident.")


if __name__ == "__main__":
    main()
