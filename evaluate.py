import argparse
import os, sys
from typing import Dict, List
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.data_processor import (
    read_json,
    get_cleaned_final_answer,
    get_answers_from_dict,
    get_answers_from_list,
    get_simpleqa_answers,
)


def compute_metrics_for_open_answer(
    answers: List[str], true_answers: List[str]
) -> Dict[str, float]:
    precision_list = []
    recall_list = []
    f1_score_list = []

    for answer, true_answer in zip(answers, true_answers):
        answer = set(answer.split(" "))
        true_answer = set(true_answer.split(" "))

        tp = len(answer.intersection(true_answer))
        fp = len(answer.difference(true_answer))
        fn = len(true_answer.difference(answer))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall) if tp > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    return {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1_score": np.mean(f1_score_list),
    }


def compute_metrics_for_list_answer(
    answers: List[List[str]], true_answers: List[List[str]]
) -> Dict[str, float]:
    positive_answers = []
    negative_answers = []
    for answer, true_answer in zip(answers, true_answers):
        positive = 0
        negative = 0
        for item in answer:
            print(f"Item: {item}, True Answer: {true_answer}")
            if item in true_answer:
                positive += 1
            else:
                negative += 1
        positive_answers.append(positive)
        negative_answers.append(negative)

    tp = np.sum(positive_answers)
    fp = np.sum(negative_answers)

    return {
        "positive_avg": np.mean(positive_answers),
        "negative_avg": np.mean(negative_answers),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "total_tp": tp,
        "total_fp": fp
    }


def compute_metrics_for_simpleqa(
    answers: List[str], true_answers: List[str]
) -> Dict[str, float]:
    """
    Compute exact match accuracy for SimpleQA dataset.
    SimpleQA requires precise factual answers.
    """
    exact_matches = []
    normalized_matches = []
    
    for answer, true_answer in zip(answers, true_answers):
        # Exact match
        exact_match = 1.0 if answer.strip() == true_answer.strip() else 0.0
        exact_matches.append(exact_match)
        
        # Normalized match (case-insensitive, whitespace normalized)
        answer_norm = answer.strip().lower()
        true_answer_norm = true_answer.strip().lower()
        normalized_match = 1.0 if answer_norm == true_answer_norm else 0.0
        normalized_matches.append(normalized_match)
    
    return {
        "exact_match_accuracy": np.mean(exact_matches),
        "normalized_match_accuracy": np.mean(normalized_matches),
        "total_questions": len(answers),
        "correct_exact": int(np.sum(exact_matches)),
        "correct_normalized": int(np.sum(normalized_matches))
    }


def evaluate(result_path: str, dataset_path: str, dataset_type: str):
    if not (os.path.exists(dataset_path) and os.path.exists(result_path)):
        raise ValueError("Dataset or results path does not exist.")
    dataset = read_json(dataset_path)
    results = read_json(result_path)

    # Labels
    if dataset_type == "wikidata":
        true_answers = get_answers_from_dict(dataset)
    elif dataset_type == "wikidata_category" or dataset_type == "multispan_qa":
        true_answers = get_answers_from_list(dataset)
    elif dataset_type in ["simpleqa", "simpleqa_small"]:
        true_answers = get_simpleqa_answers(dataset)

    # Predictions
    if dataset_type == "multispan_qa":
        answers = [result["Final Answer Section"] for result in results]
        metrics = compute_metrics_for_open_answer(answers, true_answers)
    elif dataset_type in ["simpleqa", "simpleqa_small"]:
        answers = [result["Final Answer Section"] for result in results]
        metrics = compute_metrics_for_simpleqa(answers, true_answers)
    else:
        answers = get_cleaned_final_answer(results, "Final Answer Section")
        metrics = compute_metrics_for_list_answer(answers, true_answers)

    print(f"metrics: {metrics}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-r", "--result-path", type=str, help="Path to the result file."
    )
    argParser.add_argument(
        "-d", "--dataset-path", type=str, help="Path to the original dataet."
    )
    argParser.add_argument(
        "-t",
        "--dataset-type",
        type=str,
        help="Type of the dataet.",
        choices=["wikidata", "wikidata_category", "multispan_qa", "simpleqa","simpleqa_small"],
    )

    args = argParser.parse_args()

    evaluate(args.result_path, args.dataset_path, args.dataset_type)