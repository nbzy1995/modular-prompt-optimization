# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

The project uses a Python virtual environment. Set it up as follows:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with required API keys:
```
OPENAI_API_KEY=sk-abc.....
HF_API_KEY=hf_abc.......
GOOGLE_API_KEY=your_google_api_key
SCALEDOWN_API_KEY=your_scaledown_api_key
```

## Core Commands

### Running Experiments
```bash
python3 main.py --model=MODEL --task=TASK --optimization_mode=MODE [--temperature=0.07] [--top-p=0.9] [--fresh-start=True]
```

### Evaluation
```bash
python3 src/evaluate.py -r RESULT_PATH -d DATASET_PATH -t DATASET_TYPE
```

### Available Options
- **Models**: `llama2`, `llama2_70b`, `llama-65b`, `gpt3`, `gemini2.5_flash_lite`, `scaledown-gpt-4o`
- **Tasks**: `wikidata`, `wikidata_category`, `multispanqa`, `simpleqa`
- **Optimization Modes**: `cove`, `cot`
- **CoVe Settings**: `joint`, `two_step`, `factored` (configured in TaskConfig)

## Architecture Overview

### Core Components

**Data Pipeline (`src/data/`)**:
- `data_processor.py`: Core utilities for loading JSON/JSONL datasets and extracting questions/answers
- Dataset-specific preprocessors for different question-answering tasks
- Functions handle both dict-based (wikidata) and list-based (multispanqa) data formats

**Prompt Optimization (`src/prompt_optim/`)**:
- `cove/`: Chain-of-Verification implementation with multiple execution modes
  - `cove_chains.py`: Base ChainOfVerification class with three modes (joint, two_step, factored)
  - `cove_chains_google.py`: Google Gemini-specific implementation
  - `cove_chains_hf.py`: HuggingFace model implementation
- Each optimization technique follows a pattern: baseline → plan → execute → verify

**Configuration System (`src/utils.py`)**:
- `TaskConfig`: Defines prompts, token limits, and commands for each task
- `ModelConfig`: Model-specific configurations including prompt formats
- Task-specific prompt templates stored in `src/prompts.py`
- Supports different prompt formats (Standard, GPT, Llama) via model configs

**Evaluation (`src/evaluate.py`)**:
- Metrics computation for both open-ended and list-based answers
- Precision, recall, F1-score calculation
- Supports comparison between baseline and optimized responses

### Data Flow

1. `main.py` loads dataset questions based on task selection
2. Model-specific chain classes apply selected prompt optimization technique
3. Results saved to `experiments/result/` as JSON files
4. Optional evaluation compares optimized vs baseline responses using ground truth

### Key Design Patterns

- **Chain Pattern**: All optimization techniques inherit from base ChainOfVerification class
- **Strategy Pattern**: Different models implement same interface but with model-specific API calls
- **Template Method**: Common flow (baseline → optimize → verify) with task-specific prompts
- **Configuration-Driven**: Tasks and models defined via dataclass configs rather than hardcoded

### File Organization

```
src/
├── data/           # Dataset loading and preprocessing
├── prompt_optim/   # Optimization technique implementations
├── prompts.py      # All prompt templates
├── utils.py        # Configuration classes and utilities
└── evaluate.py     # Evaluation metrics and comparison

dataset/            # Processed datasets ready for experiments
experiments/        # Jupyter notebooks and result files
tests/              # Test files (to be implemented)
```

## Development Notes

- Model implementations must inherit from `ChainOfVerification` base class
- New tasks require adding TaskConfig with all three CoVe modes (joint, two_step, factored)
- Checkpoint system saves progress to `experiments/checkpoints/` to resume interrupted runs
- Results are automatically saved with naming convention: `{model}_{task}_{mode}_results.json`