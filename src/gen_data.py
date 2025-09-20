from datasets import load_dataset, load_from_disk, Dataset
import os
import pandas as pd
import random
import glob
import json
from transformers import AutoTokenizer


def load_and_process_dataset(data_name_or_path, tokenizer):
    if "gsm8k" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)['train']
        except:
            dataset = load_dataset("openai/gsm8k", "socratic")["train"]
        question_col = "question"
        answer_col = "answer"

    elif "MATH-500" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)['test']
        except:
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        question_col = "problem"
        answer_col = "answer"

    elif "scibench" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)['train']
        except:
            dataset = load_dataset("xw27/scibench")["train"]
        question_col = "problem_text"
        answer_col = "answer_number"

    elif "aime" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)['train']
        except:
            dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
        question_col = "Problem"
        answer_col = "Answer"

    elif "jama" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)['test']
        except:
            dataset = load_dataset("xw27/scibench")["test"]
        question_col = "question"
        answer_col = "answer_idx"
        option_A = "opa"
        option_B = "opb"
        option_C = "opc"
        option_D = "opd"

    else:
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    def preprocess_function(examples):
        formatted = []
        questions = examples[question_col]
        for q in questions:
            messages = [
                {"role": "system",
                 "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q},
            ]
            formatted.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
        return {"formatted": formatted, "question": questions, "answer": examples[answer_col]}

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    return dataset


def process_aime_dataset_to_json(aime_data_path, output_path):
    try:
        if os.path.isdir(aime_data_path):
            dataset = load_from_disk(aime_data_path)['train']
        else:
            dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    except Exception as e:
        print(f"Error loading AIME dataset: {e}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            json_item = {
                "question": item["Problem"],
                "answer": item["Answer"]
            }
            f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
    
    print(f"AIME dataset processed and saved to {output_path}")
    print(f"Total samples: {len(dataset)}")


def _write_jsonl(file_path, items):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
