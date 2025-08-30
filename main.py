import argparse
import os
import sys
from dotenv import dotenv_values, load_dotenv
load_dotenv()

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
    "simpleqa": get_absolute_path("dataset/clean_simple_qa_100.json"),
    "test": get_absolute_path("temp_one_question.json"),  # For testing
}

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m", "--model",
        type=str,
        help="LLM to use for predictions.",
        default="gemini-2.5-flash",
        choices=["llama2", "llama2_70b", "llama-65b", "gpt3", "gemini2.5-flash-lite", "scaledown-gpt-4o"],
    )
    argParser.add_argument(
        "-t", "--task",
        type=str,
        help="Task to evaluate on.",
        default="simpleqa",
        choices=["wikidata", "wikidata_category", "multispanqa", "simpleqa"],
    )
    argParser.add_argument(
        "-mode", "--optimization_mode",
        type=str,
        help="The prompt optimization method to use.",
        default="cove",
        choices=["cove",
        "persona",
        "uncertainty",
        "combined",
        "cot",
        "persona_cot",
        "cot_uncertainty",],
    )
    argParser.add_argument("-temp", "--temperature", type=float, help="Temperature.", default=0.0)
    argParser.add_argument("-p", "--top-p", type=float, help="Top-p.", default=0.9)

    # Flags / QoL
    argParser.add_argument("--fresh-start", action="store_true",
                           help="Delete existing checkpoint before running.")
    argParser.add_argument("--limit", type=int, default=None,
                           help="Only run first N questions (for quick iteration).")

    args = argParser.parse_args()

    # --------------------------------------------------
    # 1. Load task and dataset
    # --------------------------------------------------
    task_path = file_path_mapping[args.task]
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Dataset file not found: {task_path}")
    data = read_json(task_path)
    if args.task == "wikidata":
        questions = get_questions_from_dict(data)
    else:
        questions = get_questions_from_list(data)
    if args.limit:
        questions = questions[:args.limit]

    # --------------------------------------------------
    # 2. Setup LLM model
    # --------------------------------------------------
    try:
        llm = LLMProviderFactory.create_provider(
            model_id=args.model,
            temperature=args.temperature,
            configuration=CONFIG
        )
        print(f"🤖 Created {llm.get_model_info()['provider']} for model {args.model}")
    except ValueError as e:
        print(f"❌ Error creating LLM provider: {e}")
        sys.exit(1)

    # --------------------------------------------------
    # 3. Fresh start: remove checkpoint
    # --------------------------------------------------
    if args.fresh_start:
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(
            checkpoint_dir, f"{args.model}_{args.task}_{args.optimization_mode}_checkpoint.json"
        )
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("🗑️ Removed existing checkpoint for fresh start")

    # --------------------------------------------------
    # 4. Run optimization method
    # --------------------------------------------------
    if args.optimization_mode == "cove":
        from src.prompt_optim.cove.cove_chains import ChainOfVerification
        ChainOfVerification(llm=llm, task=args.task, questions=questions).run_chain()

    elif args.optimization_mode == "persona":
        from src.prompt_optim.common.persona.persona_chain import PersonaChain
        PersonaChain(llm=llm, task=args.task, questions=questions, model_id=args.model).run()

    elif args.optimization_mode == "uncertainty":
        from src.prompt_optim.common.uncertainity.uncertainty_chain import UncertaintyChain
        UncertaintyChain(llm=llm, task=args.task, questions=questions, model_id=args.model).run()

    elif args.optimization_mode == "combined":
        from src.prompt_optim.common.PersonaUncertainty.combined_chain import CombinedChain
        CombinedChain(llm=llm, task=args.task, questions=questions, model_id=args.model).run()

    elif args.optimization_mode == "cot":
        from src.prompt_optim.common.CoT.cot_chain import ChainOfThought
        ChainOfThought(llm=llm, task=args.task, questions=questions, model_id=args.model).run()

    elif args.optimization_mode == "persona_cot":
        from src.prompt_optim.common.PersonaCoT.persona_cot_chain import PersonaCoTChain
        PersonaCoTChain(llm=llm, task=args.task, questions=questions, model_id=args.model).run()

    elif args.optimization_mode == "cot_uncertainty":
        from src.prompt_optim.common.CoTUncertainty.cot_uncertainty_chain import CoTUncertaintyChain
        CoTUncertaintyChain(llm=llm, task=args.task, questions=questions, model_id=args.model).run()

    else:
        print(f"❌ Unknown optimization mode: {args.optimization_mode}")
        sys.exit(1)



    # --------------------------------------------------
    # 5. (Evaluation happens elsewhere in the repo)
    # --------------------------------------------------
