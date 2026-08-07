"""
Phase-2 generation for the calibration / abstention study.

Fixes the two design flaws the spike exposed in the committed data:
  1. Always elicits a PARSEABLE confidence score (so calibration/ECE is computable
     for every condition, not just persona_uncertainty).
  2. The 'abstention' module gives an explicit, emphatic instruction to answer
     "I don't know" when unsure -- the old 'uncertainty' prompt only asked for a
     confidence number and never actually induced abstention (~1% in the data).

Reuses the repo's optimizer module text and the AI/ML provider; uses a dedicated
runner (not experiment.py) so we get full-response capture, confidence parsing,
seed control, and clean output paths without touching the canonical code path.

Usage:
    ./.venv-spike/bin/python experiments/phase2/run_generation.py \
        --model aiml-gpt-4o --n 100 --seeds 3 \
        --conditions none abstention expert_persona expert_persona+abstention cot cot+abstention
"""
import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from dotenv import dotenv_values
from src.llms import LLMProviderFactory
from src.prompts import (
    EXPERT_PERSONA_PROMPT, COT_PROMPT, COVE_PROMPT,
    SIMPLEQA_TASK_PROMPT, SIMPLEQA_EXAMPLES_PROMPT, SIMPLEQA_QUESTION_PROMPT,
)
from src.utils import extract_final_answer_section

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen")
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "simpleqa.json")
DATASET_SMALL = os.path.join(REPO_ROOT, "dataset", "simpleqa_small.json")

# --- redesigned prompt components -------------------------------------------

# Explicit abstention manipulation (replaces the old confidence-only "uncertainty").
ABSTENTION_PROMPT = """ANSWERING POLICY: Only answer if you are confident the answer is correct. \
It is better to abstain than to guess. If you are not sure, respond with exactly "I don't know" \
as the final answer rather than guessing."""

# Threshold-parametrized abstention for the accuracy-vs-coverage sweep.
# Used via condition tokens like "abstain@70".
THRESHOLD_ABSTENTION_TEMPLATE = """ANSWERING POLICY: Only answer if you are at least {t}% \
confident that the answer is correct. If your confidence is below {t}%, respond with exactly \
"I don't know" as the final answer rather than guessing."""

# Always-on confidence measurement, appended AFTER the answer block so the
# existing extractor still returns just the answer.
FINAL_ANSWER_FORMAT_WITH_CONF = """
FINAL ANSWER FORMAT: After any reasoning, you MUST end your response with a clearly delimited \
final answer section, followed by a confidence line:

=== FINAL ANSWER ===
[Your single, precise answer here - no additional details. Write "I don't know" only if instructed by the answering policy and you are unsure.]
=== END FINAL ANSWER ===
CONFIDENCE: [integer from 0 to 100 = your probability that the final answer is correct]"""

MODULES = {
    "expert_persona": EXPERT_PERSONA_PROMPT,
    "cot": COT_PROMPT,
    "cove": COVE_PROMPT,
    "abstention": ABSTENTION_PROMPT,
}


def build_prompt(question, modules):
    """ROLE -> TASK -> [cot/cove/abstention] -> FORMAT(+confidence) -> EXAMPLES -> QUESTION."""
    parts = []
    if "expert_persona" in modules:
        parts.append(EXPERT_PERSONA_PROMPT)
    parts.append(SIMPLEQA_TASK_PROMPT)
    for m in modules:
        if m in ("expert_persona",):
            continue
        if m.startswith("abstain@"):
            parts.append(THRESHOLD_ABSTENTION_TEMPLATE.format(t=int(m.split("@")[1])))
        else:
            parts.append(MODULES[m])
    parts.append(FINAL_ANSWER_FORMAT_WITH_CONF)
    parts.append(SIMPLEQA_EXAMPLES_PROMPT)
    parts.append(SIMPLEQA_QUESTION_PROMPT.format(question=question))
    return "\n\n".join(parts)


CONF_RE = re.compile(r"CONFIDENCE(?:\s*ASSESSMENT)?\s*[:=]?\s*(\d{1,3})\s*%?", re.IGNORECASE)


def parse_confidence(response):
    """Return the last confidence integer (0-100) found, or None."""
    matches = CONF_RE.findall(response or "")
    for v in reversed(matches):
        iv = int(v)
        if 0 <= iv <= 100:
            return iv
    return None


def parse_condition(name):
    """'expert_persona+abstention' -> ['expert_persona','abstention']; 'none' -> []."""
    if name == "none":
        return []
    return name.split("+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aiml-gpt-4o")
    ap.add_argument("--n", type=int, default=100, help="number of questions")
    ap.add_argument("--seeds", type=int, default=1, help="number of repeat runs")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temp; >0 so seeds differ (calibration needs spread)")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--full", action="store_true", help="use full simpleqa.json instead of the 100-q subset")
    ap.add_argument("--conditions", nargs="+",
                    default=["none", "abstention", "expert_persona",
                             "expert_persona+abstention", "cot", "cot+abstention"])
    args = ap.parse_args()

    cfg = dotenv_values(os.path.join(REPO_ROOT, ".env"))
    os.makedirs(OUT_DIR, exist_ok=True)

    ds_path = DATASET_PATH if args.full else DATASET_SMALL
    dataset = json.load(open(ds_path))[: args.n]
    questions = [d["problem"] for d in dataset]
    golds = [d["answer"] for d in dataset]

    total_calls = len(args.conditions) * args.seeds * len(questions)
    print(f"Model={args.model} temp={args.temperature} | {len(args.conditions)} conditions "
          f"x {args.seeds} seeds x {len(questions)} q = {total_calls} generations\n")

    for seed in range(args.seeds):
        # New provider per seed (temperature>0 gives run-to-run variation).
        llm = LLMProviderFactory.create_provider(args.model, temperature=args.temperature, configuration=cfg)
        for cond in args.conditions:
            modules = parse_condition(cond)
            out_file = os.path.join(OUT_DIR, f"{args.model}_simpleqa_{cond}_seed{seed}.json")
            if os.path.exists(out_file):
                print(f"  skip (exists): {os.path.basename(out_file)}")
                continue
            records = []
            n_abstain = n_conf = 0
            for i, (q, gold) in enumerate(zip(questions, golds)):
                prompt = build_prompt(q, modules)
                resp = llm.call_llm(prompt, args.max_tokens)
                ans = extract_final_answer_section(resp)
                conf = parse_confidence(resp)
                is_abstain = "i don't know" in (ans or resp).lower() or "i do not know" in (ans or resp).lower()
                n_abstain += int(is_abstain)
                n_conf += int(conf is not None)
                records.append({
                    "Question": q, "target": gold,
                    "Optimizers Used": cond,
                    "Optimized Answer": resp,
                    "Final Answer Section": ans,
                    "confidence": conf,
                    "abstained": is_abstain,
                })
                print(f"\r  seed{seed} {cond:<26} {i+1}/{len(questions)} "
                      f"| conf_parsed={n_conf} abstain={n_abstain}", end="", flush=True)
            print()
            json.dump(records, open(out_file, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved generations to {OUT_DIR}")


if __name__ == "__main__":
    main()
