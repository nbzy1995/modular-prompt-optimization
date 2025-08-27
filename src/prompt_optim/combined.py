#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week-3: Expert Persona + Uncertainty + CoT+Uncertainty + Persona+CoT — OpenAI pipeline

- Robust column mapping (question/context/ground_truth).
- Prompt modes: persona, uncertainty, combined, cot_uncertainty, persona_cot.
- Strict "Final Answer:" extraction (with safe fallback).
- Scoring: exact/fuzzy/contains + abstention + suspicious.
- Aggregation + plotting:
    (1) grouped bars per mode,
    (2) 100%-stacked outcome distribution per mode.

Requirements:
  pip install pandas matplotlib python-dotenv openai
Env:
  setx OPENAI_API_KEY "sk-..."   # or export OPENAI_API_KEY=...
"""

import os, re
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

# ============= Config =============
MODEL = "gpt-4o-mini"          # or "gpt-4o", "gpt-4.1"
DATA_PATH = "clean_simple_qa_100.csv"
TEMPERATURE = 0.0
RESULTS_CSV = f"cot+personav2_week3_results(temp{TEMPERATURE}).csv"
CHART_PATH  = f"cot+personav2_week3_chart(temp{TEMPERATURE}).png"
CHART_PATH2 = f"cot+personav2_week3_chart_stacked(temp{TEMPERATURE}).png"

# ============= OpenAI Client (new + legacy fallback) =============
_client_mode = None
def _init_openai():
    global _client_mode
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")
    try:
        from openai import OpenAI  # new SDK
        client = OpenAI(api_key=api_key)
        _client_mode = "new"
        return client
    except Exception:
        import openai  # legacy SDK
        openai.api_key = api_key
        _client_mode = "legacy"
        return openai

_openai_client = _init_openai()

def chat_complete(messages, model=MODEL, temperature=TEMPERATURE) -> str:
    """Unified call supporting both SDK styles. Returns string content."""
    if _client_mode == "new":
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    else:
        resp = _openai_client.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"].strip()

# ============= Helpers =============
def is_fuzzy_match(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, (a or "").lower().strip(), (b or "").lower().strip()).ratio() >= threshold

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s.]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def contains_match(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    return bool(g) and (g in p or p in g)

def _clean_ctx(ctx: str) -> str:
    s = str(ctx or "").strip()
    return s if s and s.lower() != "nan" else "None"

# ============= Prompts =============
def persona_prompt(context: str, question: str):
    return [
        {"role": "system",
         "content": (
            "You are a highly knowledgeable and careful scientific expert. "
            "Answer the following question based only on the provided context.\n\n"
            f"Context: {_clean_ctx(context)}"
         )},
        {"role": "user", "content": f"Question: {question}"},
    ]

def uncertainty_prompt(context: str, question: str):
    return [
        {"role": "system",
         "content": (
            "You are a cautious and factual assistant. "
            "Only answer the question if you are certain based on the context. "
            "If not, respond with: 'I don't know.'\n\n"
            f"Context: {_clean_ctx(context)}"
         )},
        {"role": "user", "content": f"Question: {question}"},
    ]

def combined_prompt(context: str, question: str):
    return [
        {"role": "system",
         "content": (
            "You are a cautious and knowledgeable scientific expert. "
            "Only answer the question if the answer is supported by the context and you are confident. "
            "If unsure or unsupported, respond with: 'I don't know.'\n\n"
            f"Context: {_clean_ctx(context)}"
         )},
        {"role": "user", "content": f"Question: {question}"},
    ]

# CoT + Uncertainty (strict Final Answer line)
def cot_uncertainty_prompt(context: str, question: str):
    return [
        {"role": "system",
         "content": (
            "You are a careful senior scientific expert.\n"
            "Rules:\n"
            "1) Think step by step briefly (2–4 concise steps) using ONLY the provided context.\n"
            "2) If you are uncertain or context is insufficient, abstain.\n"
            "3) Your LAST line MUST be exactly: Final Answer: <answer>\n"
            "   - If unsure, write: Final Answer: I don't know\n"
            "4) Do NOT add any text after the Final Answer line.\n\n"
            f"Context: {_clean_ctx(context)}"
         )},
        {"role": "user", "content": f"Question: {question}"},
    ]

# Persona + CoT (no explicit uncertainty rule)
def persona_cot_prompt(context: str, question: str):
    return [
        {"role": "system",
         "content": (
            "You are a highly knowledgeable and precise scientific expert.\n"
            "Explain your reasoning step by step in 2–4 concise steps using ONLY the provided context.\n"
            "Then on the LAST line output exactly: Final Answer: <answer>\n"
            "Do NOT add any text after the Final Answer line.\n\n"
            f"Context: {_clean_ctx(context)}"
         )},
        {"role": "user", "content": f"Question: {question}"},
    ]

PROMPT_BUILDERS = {
    "persona": persona_prompt,
    "uncertainty": uncertainty_prompt,
    "combined": combined_prompt,
    "cot_uncertainty": cot_uncertainty_prompt,
    "persona_cot": persona_cot_prompt,
}

SUSPICIOUS_KWS = [
    "some believe", "may help", "debated", "conspiracy",
    "healing energy", "controversial", "theory suggests",
]

FINAL_RE = re.compile(r"(?i)^\s*final\s*answer\s*:\s*(.+?)\s*$")
ABSTAIN_RE = re.compile(
    r"\bi\s*do\s*not\s*know\b|\bi\s*don[’']?t\s*know\b|\binsufficient\s+information\b|\bnot\s+enough\s+(?:info|information|context)\b",
    re.IGNORECASE
)

def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = FINAL_RE.match(ln)
        if m:
            return m.group(1).strip()
    return lines[-1] if lines else ""

def is_abstain(ans: str) -> bool:
    return bool(ABSTAIN_RE.search((ans or "").strip()))

# ============= Load Data & Robust Column Mapping =============
df = pd.read_csv(DATA_PATH)
lower_map = {c.lower().strip(): c for c in df.columns}

Q_COL = next((lower_map[k] for k in
              ["question", "query", "prompt", "problem"] if k in lower_map),
             list(df.columns)[0])

C_COL = next((lower_map[k] for k in
              ["retrieved_context", "context", "metadata", "passage", "evidence", "support", "doc", "text"]
              if k in lower_map), None)

A_COL = next((lower_map[k] for k in
              ["ground_truth", "answer", "expected_answer", "label", "gold", "target"]
              if k in lower_map), None)

if C_COL: df[C_COL] = df[C_COL].fillna("")
if A_COL: df[A_COL] = df[A_COL].fillna("")
df[Q_COL] = df[Q_COL].fillna("")

print(f"[config] MODEL={MODEL}  TEMPERATURE={TEMPERATURE}")
print(f"[mapping] question={Q_COL}  context={C_COL or '(none)'}  ground_truth={A_COL or '(none)'}")

# ============= Evaluation Log =============
log = {
    "prompt_type": [],
    "question": [],
    "ground_truth": [],
    "retrieved_context": [],
    "model_output_raw": [],
    "final_answer": [],
    "accurate": [],
    "hallucinated": [],
    "abstained": [],
    "suspicious": [],
}

# ============= Run Experiments =============
for prompt_type, builder in PROMPT_BUILDERS.items():
    print(f"\n==== Running prompt type: {prompt_type.upper()} ====\n")
    for _, row in df.iterrows():
        question = str(row[Q_COL])
        context  = str(row[C_COL]) if C_COL else ""
        ground_truth = str(row[A_COL]) if A_COL else ""

        messages = builder(context, question)

        try:
            output_raw = chat_complete(messages, model=MODEL, temperature=TEMPERATURE)
        except Exception as e:
            print(f"[ERROR] {e}")
            output_raw = "error"

        final_ans = extract_final_answer(output_raw).lower()
        gt = (ground_truth or "").lower()

        is_binary = gt in ["yes", "no"]
        abstained = is_abstain(final_ans) or final_ans == "error"
        is_suspicious = any(kw in (output_raw.lower()) for kw in SUSPICIOUS_KWS)

        if is_binary:
            accurate = (final_ans.startswith(gt) and not abstained and not is_suspicious and final_ans != "error")
        else:
            if gt:
                exact = normalize(final_ans) == normalize(gt)
                fuzzy = is_fuzzy_match(final_ans, gt)
                contains = contains_match(final_ans, gt)
                accurate = ((exact or fuzzy or contains) and not abstained and not is_suspicious and final_ans != "error")
            else:
                accurate = False

        hallucinated = (not abstained) and (final_ans != "error") and (not accurate)

        # Log
        log["prompt_type"].append(prompt_type)
        log["question"].append(question)
        log["ground_truth"].append(ground_truth)
        log["retrieved_context"].append(context)
        log["model_output_raw"].append(output_raw)
        log["final_answer"].append(final_ans)
        log["accurate"].append(accurate)
        log["hallucinated"].append(hallucinated)
        log["abstained"].append(abstained)
        log["suspicious"].append(is_suspicious)

        print(f"Q: {question}")
        print(f"GT: {ground_truth}")
        print(f"RAW: {output_raw}")
        print(f"FA : {final_ans}")
        print(f"accurate={accurate}  hallucinated={hallucinated}  abstained={abstained}  suspicious={is_suspicious}\n")

# ============= Save & Aggregate =============
results = pd.DataFrame(log)
results.to_csv(RESULTS_CSV, index=False)

agg = (
    results.groupby("prompt_type")
    .agg(
        accuracy_rate=("accurate", "mean"),
        hallucination_rate=("hallucinated", "mean"),
        abstention_rate=("abstained", "mean"),
        suspicious_rate=("suspicious", "mean"),
        total=("question", "count"),
    )
    .reset_index()
)

print("\n==== Aggregated Metrics ====\n")
print(agg)

# ============= Plot 1: Grouped bars (like before) =============
# Coerce to numeric & fill NaN
metric_cols = ["accuracy_rate", "hallucination_rate", "abstention_rate", "suspicious_rate"]
for c in metric_cols:
    agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0.0)

# Order on x-axis
order = ["combined", "cot_uncertainty", "persona", "persona_cot", "uncertainty"]
agg_plot = (
    agg.set_index("prompt_type")
       .reindex([m for m in order if m in agg["prompt_type"].tolist()])
       .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(agg_plot))
bar_w = 0.2

ax.bar([p - 1.5*bar_w for p in x], agg_plot["accuracy_rate"] * 100.0,      width=bar_w, label="Accuracy")
ax.bar([p - 0.5*bar_w for p in x], agg_plot["hallucination_rate"] * 100.0,  width=bar_w, label="Hallucination")
ax.bar([p + 0.5*bar_w for p in x], agg_plot["abstention_rate"] * 100.0,     width=bar_w, label="Abstention")
ax.bar([p + 1.5*bar_w for p in x], agg_plot["suspicious_rate"] * 100.0,     width=bar_w, label="Suspicious")

name_map = {
    "persona": "Persona",
    "uncertainty": "Uncertainty",
    "combined": "Combined (Persona + Uncertainty)",
    "cot_uncertainty": "CoT + Uncertainty",
    "persona_cot": "Persona + CoT",
}
labels = [name_map.get(m, m.title().replace("_", " ")) for m in agg_plot["prompt_type"]]
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=0)
ax.set_ylim(0, 100)
ax.set_ylabel("Percentage")
ax.set_title(f"Persona vs Uncertainty vs Combined vs CoT+Uncertainty vs Persona+CoT ({MODEL}, T={TEMPERATURE})")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(CHART_PATH, bbox_inches="tight", dpi=150)
plt.show()

# ============= Plot 2: 100%-stacked outcome distribution per mode =============
# Build outcome proportions from results
dist = (
    results.assign(
        correct=lambda d: d["accurate"].astype(int),
        wrong=lambda d: d["hallucinated"].astype(int),
        abstained=lambda d: d["abstained"].astype(int),
        suspicious=lambda d: d["suspicious"].astype(int),
    )
    .groupby("prompt_type")[["correct", "wrong", "abstained", "suspicious"]]
    .mean()  # proportions in [0,1]
    .reindex([m for m in order if m in results["prompt_type"].unique()])
)

fig, ax = plt.subplots(figsize=(10, 6))
bottom = pd.Series(0, index=dist.index)
for col, label in [("correct","Correct"), ("wrong","Hallucination"), ("abstained","Abstention"), ("suspicious","Suspicious")]:
    vals = dist[col] * 100.0
    ax.bar(dist.index, vals, bottom=bottom*100.0, label=label)
    bottom += dist[col]

ax.set_ylim(0, 100)
ax.set_ylabel("Percentage")
# Pretty x labels
labels2 = [name_map.get(m, m.title().replace("_", " ")) for m in dist.index]
ax.set_xticklabels(labels2)
ax.set_title(f"Outcome Distribution per Mode (100% stacked) ({MODEL}, T={TEMPERATURE})")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(CHART_PATH2, bbox_inches="tight", dpi=150)
plt.show()

print(f"\nSaved CSV: {RESULTS_CSV}")
print(f"Saved Charts: {CHART_PATH} and {CHART_PATH2}")
