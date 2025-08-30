# Modular Prompt Optimization Library

This is a LLM prompt optimization library containing a variety of LLM prompt optimization techniques, as well as combinations of them. This library also provides the full pipeline of evaluating LLM prompt techniques on various tasks, especially on hallucination reduction.


## :fire: Quickstart

First, create a python virtual environment and install the requirements:
```bash
python -m venv .venv
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

Then, create a file named `.env` to store API keys. This file should look like:
```
OPENAI_API_KEY=sk-abc.....
HF_API_KEY=hf_abc.......
```

An example full experiment that apply CoVe technique on wiki-qa dataset using Gemini is in `experiment/cove_experiment.ipynb`


## Usage

```bash
python3 main.py --model=MODEL --task=TASK --setting=SETTING [--temperature=0.07] [--top-p=0.9]
```


### Available Options
- **Prompt Optimization Techniques**: 
    - Cove: `joint`, `two_step`, `factored`
    - CoT: `simple`, `CoT+Uncertainty`,`Persona+CoT`
    - Persona: `expert persona`, `Persona+Uncertainty`
    - Uncertainty: `simple`

- **LLM Models**: `llama2`, `gpt3`, `gemini2.5`
- **Tasks**: `wikidata`, `wikidata_category`, `multispanqa`



## Organization

TODO: to be revised

__Data Flow__
1. Load dataset questions based on task
2. Apply selected prompt techniques (Cove, CoT, etc.)
3. Save LLM responses as JSON
4. (optional) Evaluate response with metrics


`src/`: Core Library

1. Data Processing (`src/data/`):
   - Question/answer extraction utilities
   - For various datasets

2. Prompt Optimizer: (`src/prompt_optim/`):
    - individual modules, CoVe, CoT, etc.
    - combining modules.

3. Evaluator: (`src/evaluate.py`):
    - evaluate response on ground truth, using various metrics.

4. Configuration: (`src/util`):
    - setup LLMs, tasks, metrics.


`datasets`: 

`experiments`: Example experiments with final eval results 

`tests`: TODO: to be implemented
