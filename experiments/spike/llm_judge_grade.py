"""
LLM-as-judge re-grading of existing SimpleQA result files.

Replaces the repo's lexical/exact-match grader (evaluate.py) with a proper
SimpleQA-style judge that classifies each answer as CORRECT / INCORRECT /
NOT_ATTEMPTED, following OpenAI's SimpleQA grading protocol.

Runs on the ALREADY-GENERATED answers in experiments/result/ -- no regeneration.
Uses the AI/ML API (OpenAI-compatible) so it works with an aimlapi.com key.

Usage:
    export AIML_API_KEY=...        # or put AIML_API_KEY=... in .env
    python experiments/spike/llm_judge_grade.py
    python experiments/spike/llm_judge_grade.py --judge-model gpt-4o   # higher fidelity
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter

# --- locate repo root so we can import the dataset helpers and read files ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(REPO_ROOT, "experiments", "result")
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "simpleqa_small.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graded")

# --- load .env if present (no dependency on python-dotenv) ---
def load_dotenv():
    path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# OpenAI's published SimpleQA grader template (condensed).
GRADER_TEMPLATE = """\
Grade the predicted answer to a question as one of: CORRECT, INCORRECT, or NOT_ATTEMPTED.

Question: {question}
Gold target (the single correct answer): {target}
Predicted answer: {predicted}

Rules:
- CORRECT: the predicted answer fully contains the gold target with no contradicting
  information. Minor differences in phrasing, capitalization, or extra hedging that do
  not change the factual content are fine. Semantically equivalent forms count
  (e.g. an abbreviation vs. full name, "650k" vs "651,000").
- NOT_ATTEMPTED: the predicted answer does not give the gold target and does not
  contradict it (e.g. "I don't know", "I'm not sure", refusal, or an unrelated non-answer).
- INCORRECT: the predicted answer states something that contradicts the gold target,
  i.e. a confident wrong factual claim (a hallucination).

Reply with EXACTLY one word: CORRECT, INCORRECT, or NOT_ATTEMPTED.
"""

CONDITION_ORDER = [
    "none", "cot", "cove", "uncertainty",
    "cot_uncertainty", "cove_uncertainty",
    "expert_persona_cot", "expert_persona_uncertainty",
]


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai SDK not installed. Run: pip install openai")
    api_key = os.environ.get("AIML_API_KEY") or os.environ.get("AIMLAPI_KEY")
    if not api_key:
        sys.exit("Set AIML_API_KEY in your environment or .env file.")
    return OpenAI(api_key=api_key, base_url="https://api.aimlapi.com/v1",
                  timeout=60.0, max_retries=0)


def grade_one(client, model, question, target, predicted, retries=6):
    prompt = GRADER_TEMPLATE.format(question=question, target=target, predicted=predicted)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            text = resp.choices[0].message.content.strip().upper()
            if "NOT_ATTEMPTED" in text or "NOT ATTEMPTED" in text:
                return "NOT_ATTEMPTED"
            if "INCORRECT" in text:
                return "INCORRECT"
            if "CORRECT" in text:
                return "CORRECT"
            return "NOT_ATTEMPTED"  # unparseable -> treat as no-attempt
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def condition_of(filename):
    # Match the most specific (longest) condition name first, since e.g.
    # "uncertainty" is a substring of "cot_uncertainty" and "cot" of
    # "expert_persona_cot".
    for c in sorted(CONDITION_ORDER, key=len, reverse=True):
        if f"_{c}_results" in filename:
            return c
    return os.path.basename(filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="gpt-4o-mini",
                    help="AI/ML API model id for the judge (default gpt-4o-mini; use gpt-4o for higher fidelity)")
    ap.add_argument("--limit", type=int, default=None, help="grade only first N per file (smoke test)")
    args = ap.parse_args()

    load_dotenv()
    client = get_client()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build question -> gold target map (robust to row ordering).
    dataset = json.load(open(DATASET_PATH))
    gold = {d["problem"].strip(): d["answer"].strip() for d in dataset}

    files = sorted(glob.glob(os.path.join(RESULT_DIR, "scaledown*simpleqa_small*results.json")),
                   key=lambda f: CONDITION_ORDER.index(condition_of(f)) if condition_of(f) in CONDITION_ORDER else 99)

    summary = []
    for f in files:
        cond = condition_of(f)
        records = json.load(open(f))
        if args.limit:
            records = records[: args.limit]
        counts = Counter()
        graded_records = []
        for i, r in enumerate(records):
            q = r["Question"].strip()
            target = gold.get(q)
            if target is None:
                counts["NO_GOLD"] += 1
                continue
            pred = r.get("Final Answer Section", "") or r.get("Optimized Answer", "")
            label = grade_one(client, args.judge_model, q, target, pred)
            counts[label] += 1
            graded_records.append({"Question": q, "target": target, "predicted": pred, "grade": label})
            print(f"\r{cond:<28} {i + 1}/{len(records)}  {dict(counts)}", end="", flush=True)
        print()

        json.dump(graded_records, open(os.path.join(OUT_DIR, f"{cond}_graded.json"), "w"), indent=2)

        n = sum(counts[k] for k in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED"))
        cor, inc, na = counts["CORRECT"], counts["INCORRECT"], counts["NOT_ATTEMPTED"]
        attempted = cor + inc
        summary.append({
            "condition": cond,
            "n": n,
            "accuracy": cor / n if n else 0,
            "attempted_accuracy": cor / attempted if attempted else 0,
            "hallucination_rate": inc / attempted if attempted else 0,
            "not_attempted_rate": na / n if n else 0,
        })

    # --- print + save summary table ---
    print("\n" + "=" * 88)
    print(f"{'condition':<28}{'n':>4}{'acc':>8}{'att_acc':>9}{'halluc':>9}{'abstain':>9}")
    print("-" * 88)
    for s in summary:
        print(f"{s['condition']:<28}{s['n']:>4}{s['accuracy'] * 100:>7.0f}%"
              f"{s['attempted_accuracy'] * 100:>8.0f}%{s['hallucination_rate'] * 100:>8.0f}%"
              f"{s['not_attempted_rate'] * 100:>8.0f}%")
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    print("=" * 88)
    print(f"Saved per-condition graded files and summary.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
