# Modular Prompt Optimization Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

A framework for evaluating modular combinations of prompt optimization techniques on LLM hallucination reduction. Designed for systematic experimentation with comprehensive evaluation metrics.

## Features

- **Hallucination Evaluation**: correct/incorrect/abstention responses with precision, recall, F1, and hallucination rates
- **Multiple Optimizers**: Chain-of-Thought, Chain-of-Verification, Expert Persona, Uncertainty Quantification, and their arbitrary combinations
- **Multiple Dataset**: Full support for OpenAI's SimpleQA hallucination benchmark dataset, and more
- **Multi-LLM Support**: OpenAI GPT, Google Gemini, via unified interface
- **Automatic Checkpointing**: Resume interrupted experiments, progress tracking

## Quick Start

### Prerequisites
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast Python package management:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv
```

### Setup
```bash
# Clone and setup environment (uv handles everything automatically)
git clone <repository-url>
cd modular-prompt-optimization
uv sync

# Add API keys to .env file
OPENAI_API_KEY=your-key
GOOGLE_API_KEY=your-key  
SCALEDOWN_API_KEY=your-key
```

## Usage

### Run Experiments
```bash
# Basic experiment
uv run experiment.py --task=simpleqa --model=scaledown-gpt-4o --optimizers=cot --output-path=results

# Multiple optimizers  
uv run experiment.py --task=simpleqa --model=scaledown-gpt-4o --optimizers=expert_persona,cot --output-path=results
```

### Evaluate Results
```bash
uv run evaluate.py -r results/scaledown-gpt-4o_simpleqa_cot_results.json -d dataset/simpleqa.json -t simpleqa
```

Example output:
```
🚨 HALLUCINATION ANALYSIS:
   Hallucination Rate: 0.684 (68.4% of attempted answers)
   Abstention Rate: 0.020 (model says 'I don't know')
   
🎯 CORE PERFORMANCE METRICS:
   Precision: 0.316 (accuracy when attempting answers)
   F1 Score: 0.308
```

### Analysis
```bash
cd experiments/
jupyter notebook simpleqa_hallucination_analysis.ipynb
```

## Available Options

**Models**: `scaledown-gpt-4o`, `gemini2.5_flash_lite`, `llama2`, `llama2_70b`

**Tasks**: `simpleqa`, `wikidata`, `multispanqa`, `wikidata_category`

**Optimizers**: `cot`, `cove`, `expert_persona`, `uncertainty` (combinable with commas)

## Architecture

- `src/llms.py` - LLM provider implementations
- `src/task_runner.py` - Experiment orchestration with checkpointing  
- `src/prompt_optimizer.py` - Modular optimization techniques
- `evaluate.py` - Enhanced evaluation metrics
- `experiments/` - Analysis notebook and results

## Evaluation Metrics

The framework provides detailed hallucination analysis:

- **Response Classification**: Correct, Incorrect (hallucinations), Abstentions
- **Core Metrics**: Precision, Recall, F1 Score
- **Hallucination Metrics**: Hallucination rate, abstention rate, calibration metrics
- **Interactive Analysis**: Inspect specific response types and compare optimizers

## Extension

Add new optimizers by extending `OPTIMIZER_PROMPTS` in `src/prompt_optimizer.py`.
Add new tasks by adding `TaskConfig` entries to `TASK_MAPPING` in `src/utils.py`.

## Testing

```bash
# Run unit tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src

# Install dev dependencies
uv sync --group dev
```

GitHub Actions CI automatically runs tests on Python 3.8+ for all commits and pull requests.

## Requirements

- Python 3.8+ (managed automatically by uv)
- [uv](https://docs.astral.sh/uv/) package manager
- Dependencies are defined in `pyproject.toml`

All dependencies are automatically managed by uv - no manual pip installs needed!