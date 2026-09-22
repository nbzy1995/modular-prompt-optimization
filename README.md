# Modular Prompt Optimization for Hallucination Reduction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compare individual and combined prompt instructions on factual questions. This repository contains a prompt-composition runner, saved SimpleQA responses and judge labels, and a script that summarizes the saved labels.

## Features

- Compose Chain-of-Thought, single-call verification, expert persona, and uncertainty instructions in a specified order.
- Save model responses and resume interrupted runs from checkpoints.
- Summarize saved `CORRECT`, `INCORRECT`, and `NOT_ATTEMPTED` labels for the SimpleQA study.

The prompt order is `ROLE -> TASK -> ordered modules -> FORMAT -> EXAMPLES -> QUESTION`. The role module comes first; other requested modules keep their order.

## Quick start

With Python 3.9 or later and [uv](https://docs.astral.sh/uv/), fetch SimpleQA and summarize the saved labels without an API key:

```bash
uv run --no-project python src/data/fetch_simpleqa.py
uv run --no-project python experiments/spike/summarize.py
```

The downloaded files are saved in `dataset/`, which Git ignores.

## Usage

To run a new experiment, put `SCALEDOWN_API_KEY=your-key` in a local `.env` file, then run:

```bash
uv run --no-project --with requests --with python-dotenv python experiment.py \
  --task simpleqa_small --model scaledown-gpt-4o \
  --optimizers cot --output-path runs
```

This calls the ScaleDown API once per question. Responses are written to `runs/result/`, and interrupted runs can resume from `runs/checkpoints/`. Use `--optimizers none` for the baseline or a comma-separated list such as `expert_persona,cot` for a combination.

## Available options

- **Prompt modules:** `cot`, `cove`, `expert_persona`, `uncertainty`, or `none` for the baseline. `cove` is a single-call verification instruction.
- **Configured tasks:** `simpleqa_small` (100 questions), `simpleqa` (full dataset), and `wikidata`.
- **Providers:** The example uses ScaleDown's GPT-4o route; [`src/llms.py`](src/llms.py) also contains a Gemini adapter.

## Architecture

- [`src/data/`](src/data/) contains the dataset preparation scripts. External datasets are linked below and excluded from Git.
- [`src/prompt_optimizer.py`](src/prompt_optimizer.py) composes prompts; [`src/task_runner.py`](src/task_runner.py) runs questions and writes checkpoints and responses.
- [`experiments/result/`](experiments/result/) contains the saved prompts and model responses.
- [`experiments/spike/graded/`](experiments/spike/graded/) contains saved per-question judge labels.
- [`experiments/spike/summarize.py`](experiments/spike/summarize.py) validates the saved labels and computes the table below.

## Experiment and results

| Field | Saved study |
|---|---|
| Benchmark | First 100 SimpleQA rows |
| Prompt factors | CoT, single-call CoVe-style verification, expert persona, uncertainty prompting |
| Conditions | Baseline, three standalone techniques, four two-module combinations |
| Repeats | One saved run per condition |
| Model route | `scaledown-gpt-4o`, mapped by the historical adapter to `gpt-4o` at ScaleDown with `scaledown.rate=0.0` |
| Scoring | Saved `CORRECT`, `INCORRECT`, and `NOT_ATTEMPTED` judge labels |

Three questions used as examples in the saved prompts were excluded from scoring; the current prompt uses a different example.

| Condition | Correct | Incorrect | Not attempted | Accuracy | Incorrect / attempted | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 28 | 68 | 1 | 28.9% | 70.8% | 1.0% |
| CoT | 34 | 62 | 1 | 35.1% | 64.6% | 1.0% |
| CoVe-style | 29 | 68 | 0 | 29.9% | 70.1% | 0.0% |
| Uncertainty | 30 | 66 | 1 | 30.9% | 68.8% | 1.0% |
| CoT + uncertainty | 33 | 62 | 2 | 34.0% | 65.3% | 2.1% |
| CoVe-style + uncertainty | 33 | 63 | 1 | 34.0% | 65.6% | 1.0% |
| Expert persona + CoT | 33 | 62 | 2 | 34.0% | 65.3% | 2.1% |
| Expert persona + uncertainty | 31 | 65 | 1 | 32.0% | 67.7% | 1.0% |

Chain-of-Thought had the highest accuracy among the tested conditions: 35.1%, compared with 28.9% for the baseline. None of the tested two-module combinations performed better than Chain-of-Thought alone.

### Limitations

- The evaluation used only the first 100 SimpleQA questions.
- Each condition was run once.
- The judge model used for the saved labels was not recorded.

## Extension

Add a prompt module to `OPTIMIZER_PROMPTS` in [`src/prompt_optimizer.py`](src/prompt_optimizer.py). Add a task's prompt components to `TASK_MAPPING` in [`src/utils.py`](src/utils.py).

## Data sources

- [SimpleQA](https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py): `fetch_simpleqa.py` converts OpenAI's CSV into the two local JSON files used here.
- [MultiSpanQA](https://multi-span.github.io/): the source file for `preprocess_multispanqa.py` is `MultiSpanQA_data/train.json` in the official download.
- [QUEST](https://github.com/google-research/language/blob/master/language/quest/README.md): the source file for `preprocess_wikidata_category.py` is `train.jsonl`.
- [Wikidata Query Service](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service): `generate_wikidata.py` uses the query definitions in [`src/data/wikidata_queries.csv`](src/data/wikidata_queries.csv).

## Testing

```bash
uv run --no-project --with pytest --with requests --with google-generativeai pytest tests/
```

## License

The repository code is licensed under the [MIT License](LICENSE). Saved results also contain SimpleQA questions and answers; their MIT notice is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
