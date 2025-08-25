import json
import random
from typing import List, Dict

def load_multiarith(path: str, limit: int = 50, seed: int = 0) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    samples = [
        {
            "question": item["sQuestion"].strip(),
            "answer": str(item["lSolutions"][0]),
        }
        for item in data
    ]
    random.Random(seed).shuffle(samples)  
    return samples[:limit]

def load_commonsense(path: str, limit: int = 50, seed: int = 0) -> List[Dict]:
    samples = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["question"]["stem"]
            ans = obj["answerKey"]
            choices = {c["label"]: c["text"] for c in obj["question"]["choices"]}
            samples.append({
                "question": q.strip(),
                "answer": ans,
                "choices": choices,
            })
    random.Random(seed).shuffle(samples)   
    return samples[:limit]
