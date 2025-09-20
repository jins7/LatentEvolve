"""
Data processing utilities
"""
from datasets import load_dataset, load_from_disk, Dataset
import os
import pandas as pd
import random
import glob
import json

math_prompt = """Please provide a answer, and must put your final answer within \\boxed{}"""
choice_prompt = """Please answer the following multiple-choice question by selecting the correct option (A, B, C or D). Your response must ends up with the following format:The correct answer is {your answer option letter here}."""

def extract_answer_str(true_answer, dataset_path):
    if "gsm8k" in dataset_path:
        if not isinstance(true_answer, str):
            return ''

        last_hash_pos = true_answer.rfind('####')
        if last_hash_pos == -1:
            return ''

        answer = true_answer[last_hash_pos + 4:].strip()
    else:
        answer = true_answer
    return answer

def process_json_dataset(data_path, tokenizer, recursive=True):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found")

    if os.path.isdir(data_path):
        pattern = os.path.join(data_path, "**", "*.json") if recursive else os.path.join(data_path, "*.json")
        json_files = [p for p in glob.glob(pattern, recursive=recursive) if os.path.isfile(p)]
        if not json_files:
            raise FileNotFoundError(f"No .json files found under directory: {data_path}")
    else:
        if not data_path.lower().endswith(".json"):
            raise ValueError(f"Expected a .json file, got: {data_path}")
        json_files = [data_path]

    all_formatted = []
    all_questions = []
    all_answers = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                question = obj.get("question", None)
                answer = obj.get("answer", None)
                if not isinstance(question, str) or not isinstance(answer, str):
                    continue

                messages = [
                    {"role": "system", "content": math_prompt},
                    {"role": "user", "content": question},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                all_formatted.append(formatted)
                all_questions.append(question)
                all_answers.append(answer)

    dataset_dict = {
        "formatted": all_formatted,
        "question": all_questions,
        "answer": all_answers,
    }
    return dataset_dict
