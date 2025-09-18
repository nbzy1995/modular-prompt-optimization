import json
from typing import Dict, List
import re

def read_jsonlines(path: str) -> Dict[str, str]:
    records = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records

def read_json(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_questions_from_dict(data: Dict[str, str]) -> List[str]:
    return list(data.keys())

def get_questions_from_list(data: List[Dict[str, str]]) -> List[str]:
    questions = [entry['question'] for entry in data]
    return questions

def get_answers_from_dict(data: Dict[str, str]) -> List[str]:
    return list(data.items())

def get_answers_from_list(data: List[Dict[str, str]]) -> List[str]:
    answers = [entry['answer'] for entry in data]
    return answers

def get_items_from_answer(result: str) -> List[str]:
    answers = result.split("\n")
    answers = [re.sub("([\d.]*\d+)\.\ ", "", answer) for answer in answers]
    return answers

def get_cleaned_final_answer(results: List[str], answer_slice: str) -> List[List[str]]:
    return [get_items_from_answer(result[answer_slice]) for result in results]

def load_simpleqa_dataset(path: str) -> List[Dict[str, str]]:
    """Load SimpleQA dataset from JSON file"""
    data = read_json(path)
    return data

def get_simpleqa_questions(data: List[Dict[str, str]]) -> List[str]:
    """Extract questions from SimpleQA dataset"""
    return [item['problem'] for item in data]

def get_simpleqa_answers(data: List[Dict[str, str]]) -> List[str]:
    """Extract answers from SimpleQA dataset"""
    return [item['answer'] for item in data]

def get_simpleqa_metadata(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Extract metadata from SimpleQA dataset"""
    import ast
    metadata_list = []
    for item in data:
        try:
            # Parse metadata string to dict
            if isinstance(item['metadata'], str):
                metadata = ast.literal_eval(item['metadata'])
            else:
                metadata = item['metadata']
            metadata_list.append(metadata)
        except:
            # Fallback if parsing fails
            metadata_list.append({'topic': 'Unknown', 'answer_type': 'Other'})
    return metadata_list
