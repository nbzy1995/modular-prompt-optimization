# Modular Prompt Optimization Library

A modular LLM prompt optimization library containing various prompt optimization techniques and their combinations. This library provides a complete pipeline for evaluating LLM prompt techniques on question-answering tasks, with a focus on hallucination reduction.

## :fire: Quickstart

First, create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then, create a `.env` file in the project root with your API keys:

```bash
OPENAI_API_KEY=sk-abc.....
HF_API_KEY=hf_abc.......
GOOGLE_API_KEY=your_google_api_key
SCALEDOWN_API_KEY=your_scaledown_api_key
```

## Usage

### Running Experiments

```bash
python3 main.py --model=MODEL --task=TASK --optimizers=OPTIMIZERS [--temperature=0.07] [--top-p=0.9] [--fresh-start=True]
```

### Examples

```bash
# Run CoVe optimization on wikidata with Gemini
python3 main.py --model=gemini2.5_flash_lite --task=wikidata --optimizers=cove

# Combine multiple optimizers on multispanqa
python3 main.py --model=scaledown-gpt-4o --task=multispanqa --optimizers=expert_persona,cot,uncertainty

# Resume interrupted experiment (automatic checkpointing)
python3 main.py --model=gemini2.5_flash_lite --task=wikidata --optimizers=cove

# Force restart experiment
python3 main.py --model=gemini2.5_flash_lite --task=wikidata --optimizers=cove --fresh-start=True
```

### Evaluation

```bash
python3 src/evaluate.py -r RESULT_PATH -d DATASET_PATH -t DATASET_TYPE
```

## Available Options

### Models
- `gemini2.5_flash_lite` - Google Gemini 2.5 Flash Lite
- `scaledown-gpt-4o` - GPT-4o via Scaledown API
- `llama2`, `llama2_70b`, `llama-65b` - Llama models (requires HuggingFace setup)
- `gpt3` - OpenAI GPT-3

### Tasks
- `wikidata` - WikiData question-answering
- `wikidata_category` - WikiData category classification
- `multispanqa` - Multi-span question-answering
- `simpleqa` - Simple question-answering
- `test` - Test dataset (uses wikidata prompts)

### Optimizers
- `expert_persona` - Expert persona prompting
- `cot` - Chain-of-Thought prompting  
- `uncertainty` - Uncertainty-aware prompting
- `cove` - Chain-of-Verification
- Multiple optimizers can be combined with commas: `expert_persona,cot,uncertainty`

## Features

### Automatic Checkpointing
- Experiments automatically save progress to `checkpoints/` directory
- Resume interrupted experiments by running the same command
- Use `--fresh-start=True` to ignore existing checkpoints

### Result Management
- Results automatically saved to `result/{model}_{task}_{optimizers}_results.json`
- Each result contains baseline and optimized responses for comparison
- Progress tracking with detailed logging

### Modular Architecture
- Easy to add new optimization techniques
- Support for multiple LLM providers
- Configuration-driven task and model management

## Architecture

### Data Flow
1. Load dataset questions based on selected task
2. Create appropriate LLM provider (Google, Scaledown, etc.)
3. For each question: generate baseline response → apply optimizers → generate optimized response
4. Save results with automatic checkpointing
5. Optional evaluation against ground truth

### Core Components

- **LLM Providers** (`src/llms.py`): Unified interface for different LLM APIs
- **Task Runner** (`src/task_runner.py`): Experiment orchestration with checkpointing
- **Prompt Optimizer** (`src/prompt_optimizer.py`): Modular optimization techniques
- **Data Pipeline** (`src/data/`): Dataset loading and preprocessing utilities
- **Configuration** (`src/utils.py`): Task and model configuration management

### Directory Structure

```
src/
├── data/              # Dataset utilities and preprocessors  
├── llms.py           # LLM provider implementations
├── task_runner.py    # Experiment orchestration
├── prompt_optimizer.py # Optimization techniques
├── prompts.py        # Prompt templates
├── utils.py          # Configuration classes
└── evaluate.py       # Evaluation metrics

dataset/              # Processed datasets
result/               # Experiment results (auto-generated)
checkpoints/          # Progress checkpoints (auto-generated)
tests/                # Test files
```

## Development

### Adding New Optimizers

1. Add prompt template to `src/prompts.py`
2. Register in `OPTIMIZER_PROMPTS` dict in `src/prompt_optimizer.py`
3. The optimizer will be automatically available via CLI

### Adding New Tasks

1. Add `TaskConfig` entry to `TASK_MAPPING` in `src/utils.py`
2. Add corresponding prompt template in `src/prompts.py`
3. Add dataset file path to `file_path_mapping` in `main.py`

### Adding New LLM Providers

1. Implement new class inheriting from `LLM` base class in `src/llms.py`
2. Add model detection logic to `LLMProviderFactory.create_provider()`
3. The provider will be automatically available via CLI

## Testing

```bash
python -m pytest tests/
```