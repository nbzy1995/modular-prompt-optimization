"""Validate and summarize the saved SimpleQA judge labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GRADED_DIR = Path(__file__).resolve().parent / "graded"
RESULT_DIR = REPO_ROOT / "experiments" / "result"
SMALL_DATASET = REPO_ROOT / "dataset" / "simpleqa_small.json"
FULL_DATASET = REPO_ROOT / "dataset" / "simpleqa.json"

CONDITIONS = (
    ("none", "Baseline"),
    ("cot", "CoT"),
    ("cove", "CoVe-style"),
    ("uncertainty", "Uncertainty"),
    ("cot_uncertainty", "CoT + uncertainty"),
    ("cove_uncertainty", "CoVe-style + uncertainty"),
    ("expert_persona_cot", "Expert persona + CoT"),
    ("expert_persona_uncertainty", "Expert persona + uncertainty"),
)

VALID_GRADES = {"CORRECT", "INCORRECT", "NOT_ATTEMPTED"}

HISTORICAL_OVERLAPS = {
    "Who received the IEEE Frank Rosenblatt Award in 2010?",
    "What's the name of the women's liberal arts college in Cambridge, Massachusetts?",
    "How much money, in euros, was the surgeon held responsible for Stella Obasanjo's death ordered to pay her son?",
}


def fail(message: str) -> None:
    raise SystemExit(f"Integrity check failed: {message}")


def load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"could not read {path}: {error}")


def normalize(value: str) -> str:
    return " ".join(value.split())


def load_dataset() -> tuple[list[dict], dict[str, str]]:
    small = load_json(SMALL_DATASET)
    full = load_json(FULL_DATASET)
    if not isinstance(small, list) or not isinstance(full, list):
        fail("SimpleQA datasets must be JSON lists")
    if len(small) != 100:
        fail(f"expected 100 rows in {SMALL_DATASET}, found {len(small)}")
    if small != full[:100]:
        fail("simpleqa_small.json is not the first 100 rows of simpleqa.json")

    targets: dict[str, str] = {}
    for index, row in enumerate(small):
        if not isinstance(row, dict) or not {"problem", "answer"} <= row.keys():
            fail(f"dataset row {index} is missing problem or answer")
        question = normalize(row["problem"])
        if question in targets:
            fail(f"duplicate dataset question: {question}")
        targets[question] = normalize(row["answer"])

    if set(targets) & HISTORICAL_OVERLAPS != HISTORICAL_OVERLAPS:
        fail("the three historical prompt-example overlaps are not all in the dataset")
    return small, targets


def load_condition(condition: str, targets: dict[str, str]) -> dict[str, str]:
    path = GRADED_DIR / f"{condition}_graded.json"
    records = load_json(path)
    if not isinstance(records, list):
        fail(f"{path.name} must contain a JSON list")
    if len(records) != 100:
        fail(f"expected 100 records in {path.name}, found {len(records)}")

    result_path = RESULT_DIR / f"scaledown-gpt-4o_simpleqa_small_{condition}_results.json"
    results = load_json(result_path)
    if not isinstance(results, list) or len(results) != len(records):
        fail(f"result count in {result_path.name} does not match {path.name}")
    predictions: dict[str, str] = {}
    for result in results:
        if not isinstance(result, dict) or "Question" not in result:
            fail(f"invalid result in {result_path.name}")
        question = normalize(result["Question"])
        if question in predictions:
            fail(f"duplicate question in {result_path.name}: {question}")
        prediction = result.get("Final Answer Section") or result.get("Optimized Answer")
        if not isinstance(prediction, str):
            fail(f"missing prediction in {result_path.name}: {question}")
        predictions[question] = prediction
    if set(predictions) != set(targets):
        fail(f"question set in {result_path.name} does not match SimpleQA")

    grades: dict[str, str] = {}
    for index, record in enumerate(records):
        required = {"Question", "target", "predicted", "grade"}
        if not isinstance(record, dict) or not required <= record.keys():
            fail(f"{path.name} record {index} is missing required fields")
        question = normalize(record["Question"])
        if question in grades:
            fail(f"duplicate question in {path.name}: {question}")
        if question not in targets:
            fail(f"unknown question in {path.name}: {question}")
        if normalize(record["target"]) != targets[question]:
            fail(f"target mismatch in {path.name}: {question}")
        if record["predicted"] != predictions[question]:
            fail(f"prediction mismatch between {path.name} and {result_path.name}: {question}")
        grade = record["grade"]
        if grade not in VALID_GRADES:
            fail(f"invalid grade in {path.name}: {grade!r}")
        grades[question] = grade

    if set(grades) != set(targets):
        fail(f"question set in {path.name} does not match simpleqa_small.json")
    overlap_count = len(set(grades) & HISTORICAL_OVERLAPS)
    if overlap_count != 3:
        fail(f"expected 3 historical overlaps in {path.name}, found {overlap_count}")
    return grades


def percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def main() -> None:
    _, targets = load_dataset()

    expected_files = {f"{condition}_graded.json" for condition, _ in CONDITIONS}
    actual_files = {path.name for path in GRADED_DIR.glob("*_graded.json")}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        fail(f"graded file set mismatch; missing={missing}, unexpected={unexpected}")

    condition_grades = {
        condition: load_condition(condition, targets) for condition, _ in CONDITIONS
    }
    remainders = {
        condition: set(grades) - HISTORICAL_OVERLAPS
        for condition, grades in condition_grades.items()
    }
    first_condition = CONDITIONS[0][0]
    expected_remainder = remainders[first_condition]
    if len(expected_remainder) != 97:
        fail(f"expected a 97-question remainder, found {len(expected_remainder)}")
    for condition, questions in remainders.items():
        if questions != expected_remainder:
            fail(f"post-exclusion question set differs for {condition}")

    print("| Condition | Correct | Incorrect | Not attempted | Accuracy | Incorrect / attempted | Abstention |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for condition, label in CONDITIONS:
        remaining_grades = [
            grade
            for question, grade in condition_grades[condition].items()
            if question in expected_remainder
        ]
        counts = Counter(remaining_grades)
        correct = counts["CORRECT"]
        incorrect = counts["INCORRECT"]
        not_attempted = counts["NOT_ATTEMPTED"]
        attempted = correct + incorrect
        print(
            f"| {label} | {correct} | {incorrect} | {not_attempted} | "
            f"{percent(correct, 97)} | {percent(incorrect, attempted)} | "
            f"{percent(not_attempted, 97)} |"
        )

if __name__ == "__main__":
    main()
