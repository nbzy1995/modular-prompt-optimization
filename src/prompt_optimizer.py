from typing import List
from .prompts import (
    EXPERT_PERSONA_PROMPT,
    UNCERTAINTY_PROMPT,
    COT_PROMPT,
    COVE_PROMPT,
)


OPTIMIZER_PROMPTS = {
    "expert_persona": EXPERT_PERSONA_PROMPT,
    "cot": COT_PROMPT, 
    "uncertainty": UNCERTAINTY_PROMPT,
    "cove": COVE_PROMPT,
    "none": "",  # Baseline prompt without optimization
}


def parse_optimizers(optimizers_string: str) -> List[str]:
    """Parse comma-separated optimizer string into list of optimizer names."""
    if not optimizers_string:
        return []
    
    optimizers = [opt.strip() for opt in optimizers_string.split(",")]
    
    # Validate all optimizers exist
    invalid_optimizers = [opt for opt in optimizers if opt not in OPTIMIZER_PROMPTS]
    if invalid_optimizers:
        valid_opts = ", ".join(OPTIMIZER_PROMPTS.keys())
        raise ValueError(f"Invalid optimizers: {invalid_optimizers}. Valid options: {valid_opts}")
    
    return optimizers


def optimize_prompt(question, optimizers_list: List[str], task_config=None) -> str:
    """
    Build prompt with sequential logic:
        ROLE → TASK → [optimizers] → FORMAT → EXAMPLES → QUESTION
    
    Args:
        optimizers_list: List of optimizer names to apply (can be empty)
        task_config: TaskConfig object containing task info
        
    Returns:
        Final optimized prompt with hierarchical structure
    """
    components = _get_task_components(task_config)
    if not components:
        raise ValueError("Task prompt components not defined for this task.")
    
    prompt_parts = []
    
    # 1. ROLE (if expert_persona in optimizers)
    if optimizers_list and "expert_persona" in optimizers_list:
        prompt_parts.append(EXPERT_PERSONA_PROMPT)
    
    # 2. TASK
    prompt_parts.append(components["task_prompt"])
    
    # 3. Add other optimizers in user-specified order
    if optimizers_list:
        for optimizer_name in optimizers_list:
            if optimizer_name in ["expert_persona", "none"]:
                continue
            optimizer_prompt = OPTIMIZER_PROMPTS.get(optimizer_name)
            if optimizer_prompt:
                prompt_parts.append(optimizer_prompt)
    
    # 4. FORMAT
    if components["final_answer_format"]:
        prompt_parts.append(components["final_answer_format"])

    # 5. EXAMPLES
    prompt_parts.append(components["examples_prompt"])
    
    # 6. QUESTION (placeholder - will be filled with actual question)
    prompt_parts.append(components["question_prompt"].format(question=question))

    return "\n\n".join(prompt_parts)

def _get_task_components(task_config):
    """Get task-specific components from task_config."""
    if not task_config:
        return None
        
    # Check if task has hierarchical components defined
    if task_config.task_prompt and task_config.examples_prompt and task_config.question_prompt:
        return {
            "task_prompt": task_config.task_prompt,
            "examples_prompt": task_config.examples_prompt, 
            "question_prompt": task_config.question_prompt,
            "final_answer_format": task_config.final_answer_format
        }
    
    # TODO: Add components for other tasks
    return None
