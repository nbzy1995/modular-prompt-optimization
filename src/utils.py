import dataclasses
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# TODO: more organized prompt import and task config.
from src.prompts import (
    BASELINE_PROMPT_WIKI_CATEGORY,
    BASELINE_PROMPT_MULTI_QA,
    WIKIDATA_FINAL_ANSWER_FORMAT,
    WIKIDATA_EXAMPLES_PROMPT,
    WIKIDATA_QUESTION_PROMPT,
    WIKIDATA_TASK_PROMPT
)

@dataclasses.dataclass
class TaskConfig:
    id: str
    max_tokens: int
    baseline_prompt: str = ""
    final_answer_format: str = ""
    baseline_command: str = " Answer: "
    # Hierarchical components (optional)
    task_prompt: str = ""
    examples_prompt: str = ""
    question_prompt: str = ""


TASK_MAPPING = {
    "wikidata": TaskConfig(
        id="wikidata",
        max_tokens=150,
        task_prompt=WIKIDATA_TASK_PROMPT,
        final_answer_format=WIKIDATA_FINAL_ANSWER_FORMAT,
        examples_prompt=WIKIDATA_EXAMPLES_PROMPT,
        question_prompt=WIKIDATA_QUESTION_PROMPT,
    ),
    "multispanqa": TaskConfig(
        id="multispanqa",
        max_tokens=200,
        baseline_prompt=BASELINE_PROMPT_MULTI_QA,
    ),
    "wikidata_category": TaskConfig(
        id="wikidata_category",
        max_tokens=100,
        baseline_prompt=BASELINE_PROMPT_WIKI_CATEGORY,
    ),
    "test": TaskConfig(
        id="test",
        max_tokens=150,
        task_prompt=WIKIDATA_TASK_PROMPT,
        final_answer_format=WIKIDATA_FINAL_ANSWER_FORMAT,
        examples_prompt=WIKIDATA_EXAMPLES_PROMPT,
        question_prompt=WIKIDATA_QUESTION_PROMPT,
    ),
}


def get_absolute_path(path_relative_to_project_root):
    import os
    current_directory = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    final_directory = os.path.join(
        current_directory,
        rf'../{path_relative_to_project_root}'
    )
    return final_directory

