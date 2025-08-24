import argparse
import os
import sys
from dotenv import dotenv_values

from src.utils import get_absolute_path
from src.data.data_processor import (
    read_json,
    get_questions_from_list,
    get_questions_from_dict,
)


CONFIG = dotenv_values(get_absolute_path(".env"))
hf_access_token = CONFIG.get("HF_API_KEY")
openai_access_token = CONFIG.get("OPENAI_API_KEY")
google_access_token = CONFIG.get("GOOGLE_API_KEY")

file_path_mapping = {
    "wikidata": get_absolute_path("dataset/wikidata_questions.json"),
    "multispanqa": get_absolute_path("dataset/multispanqa_dataset.json"),
    "wikidata_category": get_absolute_path("dataset/wikidata_category_dataset.json"),
}

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m",
        "--model",
        type=str,
        help="LLM to use for predictions.",
        default="llama2",
        choices=["llama2", "llama2_70b", "llama-65b", "gpt3", "gemini2.5_flash_lite"],
    )
    argParser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Task.",
        default="wikidata",
        choices=["wikidata", "wikidata_category", "multispanqa"],
    )
    argParser.add_argument(
        "-s",
        "--setting",
        type=str,
        help="Setting.",
        default="joint",
        choices=["joint", "two_step", "factored"],
    )
    argParser.add_argument(
        "-temp", "--temperature", type=float, help="Temperature.", default=0.07
    )
    argParser.add_argument("-p", "--top-p", type=float, help="Top-p.", default=0.9)
    argParser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Force start fresh experiment, ignoring any existing checkpoint.",
    )
    args = argParser.parse_args()

    data = read_json(file_path_mapping[args.task])
    if args.task == "wikidata":
        questions = get_questions_from_dict(data)
    else:
        questions = get_questions_from_list(data)

    # Handle fresh start flag for Google models
    if args.model == "gemini2.5_flash_lite" and args.fresh_start:
        # Remove existing checkpoint for fresh start - use current working directory
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        checkpoint_file = os.path.join(
            checkpoint_dir, f"{args.model}_{args.task}_{args.setting}_checkpoint.json"
        )
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"🗑️ Removed existing checkpoint for fresh start")

    if args.model == "gpt3":
        print("❌ OpenAI implementation not available")
        sys.exit(1)
    elif args.model == "gemini2.5_flash_lite":
        from src.prompt_optim.cove.cove_chains_google import ChainOfVerificationGoogle
        chain_google = ChainOfVerificationGoogle(
            model_id=args.model,
            temperature=args.temperature,
            task=args.task,
            setting=args.setting,
            questions=questions,
            google_access_token=google_access_token,
        )
        chain_google.run_chain()
    else:
        from src.prompt_optim.cove.cove_chains_hf import ChainOfVerificationHuggingFace
        chain_hf = ChainOfVerificationHuggingFace(
            model_id=args.model,
            top_p=args.top_p,
            temperature=args.temperature,
            task=args.task,
            setting=args.setting,
            questions=questions,
            hf_access_token=hf_access_token,
        )
        chain_hf.run_chain()
