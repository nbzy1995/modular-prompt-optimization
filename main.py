import argparse
import os
import sys
from dotenv import dotenv_values


from src.llms import LLMProviderFactory

from src.utils import get_absolute_path
from src.data.data_processor import (
    read_json,
    get_questions_from_list,
    get_questions_from_dict,
)


CONFIG = dotenv_values(".env")
google_access_token = CONFIG.get("GOOGLE_API_KEY")
scaledown_api_key = CONFIG.get("SCALEDOWN_API_KEY")

file_path_mapping = {
    "wikidata": get_absolute_path("dataset/wikidata_questions.json"),
    "multispanqa": get_absolute_path("dataset/multispanqa_dataset.json"),
    "wikidata_category": get_absolute_path("dataset/wikidata_category_dataset.json"),
    "test": get_absolute_path("temp_one_question.json"),  # For testing
}

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m",
        "--model",
        type=str,
        help="LLM to use for predictions.",
        default="scaledown-gpt-4o",
        choices=["llama2", "llama2_70b", "llama-65b", "gpt3", "gemini2.5_flash_lite", "scaledown-gpt-4o"],
    )
    argParser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Task to evaluate on.",
        default="wikidata",
        choices=["wikidata", "wikidata_category", "multispanqa", "simpleqa", "test"],
    )
    argParser.add_argument(
        "-o",
        "--optimizers",
        type=str,
        help="Comma-separated list of prompt optimizers to apply (e.g., 'expert_persona,cot,uncertainty').",
        default="cove",
    )
    argParser.add_argument(
        "-temp", "--temperature", type=float, help="Temperature.", default=0.0
    )
    argParser.add_argument("-p", "--top-p", type=float, help="Top-p.", default=0.9)
    argParser.add_argument(
        "--fresh-start",
        type=bool, 
        default=False,
        help="1 if force start fresh experiment, ignoring any existing checkpoint.",
    )
    args = argParser.parse_args()

    # --------------------------------------------------

    # Handle fresh start flag
    if args.fresh_start:
        # Remove existing checkpoint for fresh start - use current working directory
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        checkpoint_file = os.path.join(
            checkpoint_dir, f"{args.model}_{args.task}_{args.optimizers.replace(',', '_')}_checkpoint.json"
        )
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"🗑️ Removed existing checkpoint for fresh start")


    # --------------------------------------------------
    # 1. Load task and dataset
    # --------------------------------------------------
    data = read_json(file_path_mapping[args.task])
    if args.task == "wikidata":
        questions = get_questions_from_dict(data)
    else:
        questions = get_questions_from_list(data)



    # --------------------------------------------------
    # 2. Setup LLM model
    # --------------------------------------------------

    llm = LLMProviderFactory.create_provider(
        model_id=args.model,
        temperature=args.temperature,
        configuration=CONFIG
    )
    print(f"🤖 Created {llm.get_model_info()['provider']} for model {args.model}")


    # --------------------------------------------------
    # 3. Run prompt optimization experiments
    # --------------------------------------------------
    from src.task_runner import TaskRunner

    # Create and run task with optimizers
    task_runner = TaskRunner(
        llm=llm,
        task=args.task,
        questions=questions,
        optimizers=args.optimizers
    )

    task_runner.run_experiments()


    # --------------------------------------------------
    # 4. Evaluate response on dataset labels
    # --------------------------------------------------
