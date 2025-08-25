import os
import json
import pandas as pd
import re

def load_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_accuracy(results, task):
    correct, total = 0, len(results)

    for item in results:
        gt = str(item["ground_truth"]).strip()
        pred = str(item.get("prediction", item.get("raw_answer", ""))).strip()

        if task == "commonsenseqa":
            # Extract first option letter (A–E) from prediction
            match = re.search(r"\b([A-E])\b", pred.upper())
            if match:
                pred_clean = match.group(1)
                if pred_clean == gt:
                    correct += 1

        elif task == "multiarith":
            if pred == gt or pred == gt.replace(".0", ""):
                correct += 1

    return (correct / total * 100) if total > 0 else 0.0


    return (correct / total * 100) if total > 0 else 0.0

def evaluate_experiment(experiment_dir="./results"):
    table = {}

    for filename in os.listdir(experiment_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(experiment_dir, filename)
        data = load_results(filepath)

        fname = filename.lower()
        if "csqa" in fname:
            task = "CommonSenseQA"
            task_key = "commonsenseqa"
        elif "multiarith" in fname or "multi_arith" in fname:
            task = "MultiArith"
            task_key = "multiarith"
        else:
            print(f"Skipped file: {filename} (task not recognized)")
            continue
        if "cot" in fname:
            setting = "zero-shot-cot"
        else:
            setting = "zero-shot"

        acc = compute_accuracy(data, task_key)

        if task not in table:
            table[task] = {}
        table[task][setting] = acc

    # Convert to dataframe like paper
    df = pd.DataFrame(table).T
    return df
