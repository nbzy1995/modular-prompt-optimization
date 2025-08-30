import dataclasses
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

from src.prompts import (
    BASELINE_PROMPT_WIKI,
    BASELINE_PROMPT_WIKI_CATEGORY,
    BASELINE_PROMPT_MULTI_QA,
)

@dataclasses.dataclass
class TaskConfig:
    id: str
    max_tokens: int
    baseline_prompt: str
    baseline_command: str = " Answer: "


TASK_MAPPING = {
    "wikidata": TaskConfig(
        id="wikidata",
        max_tokens=150,
        baseline_prompt=BASELINE_PROMPT_WIKI,
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
        baseline_prompt=BASELINE_PROMPT_WIKI,  # Use wikidata prompt for testing
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

