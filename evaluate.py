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


def classify_simpleqa_response(answer: str, true_answer: str) -> str:
    """
    Classify a SimpleQA response into one of 4 categories:
    - 'correct': Factually accurate answer
    - 'incorrect': Wrong factual information (hallucination)
    - 'abstention': Model acknowledges uncertainty ("I don't know" etc.)
    """
    answer_clean = answer.strip().lower()
    true_answer_clean = true_answer.strip().lower()
    
    # Abstention patterns - model acknowledges uncertainty
    abstention_patterns = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "i'm uncertain", "i am uncertain", "i cannot determine", "i can't determine",
        "i'm not certain", "i am not certain", "uncertain", "unknown", 
        "not sure", "don't know", "do not know", "can't say", "cannot say",
        "i cannot answer", "i can't answer", "insufficient information",
        "not enough information", "unable to determine", "i'm unsure", "i am unsure"
    ]
    
    # Check for abstention
    for pattern in abstention_patterns:
        if pattern in answer_clean:
            return 'abstention'
    
    # Check for correct answer (normalized comparison)
    if answer_clean == true_answer_clean:
        return 'correct'
    
    # Additional fuzzy matching for correct answers (handle minor variations)
    # Remove common punctuation and extra spaces
    import re
    answer_normalized = re.sub(r'[^\w\s]', '', answer_clean).strip()
    true_answer_normalized = re.sub(r'[^\w\s]', '', true_answer_clean).strip()
    
    if answer_normalized == true_answer_normalized:
        return 'correct'
    
    return 'incorrect'


def compute_metrics_for_simpleqa(
    answers: List[str], true_answers: List[str]
) -> Dict[str, float]:
    """
    Enhanced SimpleQA evaluation with hallucination detection and calibration metrics.
    
    Classifies responses and computes precision, recall, F1, and hallucination rates.
    Following WikiData evaluation approach but adapted for factual QA.
    """
    # Classify all responses
    classifications = []
    for answer, true_answer in zip(answers, true_answers):
        classification = classify_simpleqa_response(answer, true_answer)
        classifications.append(classification)
    
    # Count categories
    correct_count = classifications.count('correct')
    incorrect_count = classifications.count('incorrect')  # Hallucinations
    abstention_count = classifications.count('abstention')
    total_questions = len(answers)

    # Attempted answers (not abstentions)
    attempted_count = correct_count + incorrect_count
    
    # Core metrics (following WikiData precision/recall approach)
    precision = correct_count / attempted_count if attempted_count > 0 else 0.0
    recall = correct_count / total_questions  # Coverage of correct knowledge
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Hallucination metrics
    hallucination_rate = incorrect_count / attempted_count if attempted_count > 0 else 0.0
    abstention_rate = abstention_count / total_questions
    
    # Calibration metrics (OpenAI SimpleQA style)
    attempted_accuracy = correct_count / attempted_count if attempted_count > 0 else 0.0
    overall_accuracy = correct_count / total_questions
    
    # Detailed breakdown
    results = {
        # Core performance metrics
        "precision": precision,
        "recall": recall, 
        "f1_score": f1_score,
        
        # Hallucination and calibration metrics
        "hallucination_rate": hallucination_rate,
        "abstention_rate": abstention_rate,
        "attempted_accuracy": attempted_accuracy,
        "overall_accuracy": overall_accuracy,
        
        # Raw counts for detailed analysis
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "incorrect_answers": incorrect_count,  # Hallucinations
        "abstentions": abstention_count,
        "attempted_answers": attempted_count,
        
        # Breakdown by category (percentages)
        "correct_pct": correct_count / total_questions * 100,
        "incorrect_pct": incorrect_count / total_questions * 100,
        "abstention_pct": abstention_count / total_questions * 100,
    }
    
    return results


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
        print(f"metrics: {metrics}")
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