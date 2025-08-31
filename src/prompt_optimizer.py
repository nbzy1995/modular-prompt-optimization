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


def optimize_prompt(baseline_prompt: str, optimizers_list: List[str]) -> str:
    """
    Optimize a baseline prompt by concatenating optimizer prompts.
    
    Args:
        baseline_prompt: The base prompt for the task
        optimizers_list: List of optimizer names to apply
        
    Returns:
        Final optimized prompt with all optimizers concatenated
    """
    if not optimizers_list:
        return baseline_prompt
    
    # Handle special case of "none" optimizer - return baseline prompt unchanged
    if optimizers_list == ["none"]:
        return baseline_prompt
    
    # Start with baseline prompt
    optimized_prompt = baseline_prompt
    
    # Append each optimizer prompt (skip empty prompts like "none")
    for optimizer_name in optimizers_list:
        optimizer_prompt = OPTIMIZER_PROMPTS[optimizer_name]
        if optimizer_prompt:  # Only add non-empty prompts
            optimized_prompt += f"\n\n{optimizer_prompt}"
    
    return optimized_prompt