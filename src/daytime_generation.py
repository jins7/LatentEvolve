from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from rewards.reward import RewardModel
from ori_generation import original_generation
from opt_generation import optimized_generation
from datasets import Dataset
from judge import *
import os
from process_data import process_json_dataset
import argparse
import numpy as np
import random
import torch.nn.functional as F
import json
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Dataset to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=2000, help="End index of the data to evaluate")

    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--k", type=float, default=0.1,
                        help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--rule_format_string", type=str, default=None, help="the answer format that should follow")

    parser.add_argument("--reward_threshold", type=float, default=-0.2,
                        help="Threshold for reward to stop optimization")

    parser.add_argument("--top_n", type=int, default=3, help="Top-N for history similarity")
    parser.add_argument("--alpha", type=float, default=0.3, help="Fusion ratio for history and current hidden states")
    parser.add_argument("--min_history_size", type=int, default=10, help="Minimum history size to start fusion")
    parser.add_argument("--output_json_path", type=str, help="Save the output json file")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_sentence_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

def main(args):
    if args.rule_format_string == "boxed":
        rule_format_string = r'\\boxed{(.*)}'
    else:
        if args.rule_format_string:
            raise ValueError("Unknown format")
        rule_format_string = None

    if args.seed:
        set_seed(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    reward_model = RewardModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        data_name=args.dataset,
        rule_format_string=rule_format_string
    )

    print("Loading dataset...")

    dataset = Dataset.from_dict(process_json_dataset(args.dataset,
                          tokenizer=tokenizer))
    print(f"Example: {dataset[0]}")

    original_correct = 0
    optimized_correct = 0
    total = 0
    update_count = 0
    original_length = 0
    optimized_length = 0
    fitten_length = 0
    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.dataset.split("/")[-1]

    latent_data = []

    output_dir = f"{args.output_dir}/{model_name}-{data_name}-k{args.k}-lr{args.lr}-SolIdx{args.solver_prompt_idx}"

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = min(args.end_data_idx, len(dataset))

    print(f"Start to evaluate {args.dataset} from {start_data_idx} to {end_data_idx}...")

    sent_tokenizer = AutoTokenizer.from_pretrained("../../models/qwen/Qwen2___5-1___5B-Instruct")
    sent_model = AutoModel.from_pretrained("../../models/qwen/Qwen2___5-1___5B-Instruct").to(device)
    history = []
    min_history_size = args.min_history_size
    N = args.top_n
    alpha = args.alpha

    data_idx_list = range(start_data_idx, end_data_idx)

    for i in tqdm(data_idx_list):
        example = dataset[i]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f"{output_dir}/test"):
            os.makedirs(f"{output_dir}/test")

        true_answer = example["answer"]

        print(f"Question: {example['question']}")
        print(f"True answer: {true_answer}")
        if true_answer is None:
            continue

        prompt_vec = get_sentence_embedding(example['question'], sent_tokenizer, sent_model, device)
        history_hidden = None
        do_fusion = False
        if len(history) >= min_history_size:
            do_fusion = True
            history_vecs = torch.stack([h['vec'] for h in history])
            sims = F.cosine_similarity(prompt_vec.unsqueeze(0), history_vecs)
            topk = torch.topk(sims, min(N, len(history)))
            weights = torch.softmax(topk.values, dim=0)
            hiddens_lists = [history[j]['hidden_states'] for j in topk.indices.tolist()]
            max_len = max(len(h_list) for h_list in hiddens_lists)
            hiddens_padded = []
            for h_list in hiddens_lists:
                if len(h_list) < max_len:
                    pad_len = max_len - len(h_list)
                    h_tensor = torch.stack(h_list)
                    if len(h_tensor.shape) == 3:
                        padding = torch.zeros(pad_len, h_tensor.shape[1], h_tensor.shape[2], device=h_tensor.device,
                                              dtype=h_tensor.dtype)
                    else:
                        padding = torch.zeros(pad_len, h_tensor.shape[1], device=h_tensor.device, dtype=h_tensor.dtype)
                    h_padded = torch.cat([h_tensor, padding], dim=0)
                else:
                    h_padded = torch.stack(h_list)
                hiddens_padded.append(h_padded)
            weighted_hidden = sum(w * h for w, h in zip(weights, hiddens_padded))
            history_hidden = weighted_hidden

        original_output, hidden_states_list, input_ids = original_generation(
            input_text=example["formatted"],
            model=model,
            tokenizer=tokenizer,
            device=device, )

        k = args.k
        update_length = min(int(k * len(hidden_states_list)), 300)
        orig_latent = [h.cpu().tolist() for h in copy.deepcopy(hidden_states_list[:update_length])]
        if update_length > 0 and do_fusion and history_hidden is not None:
            current_hidden = torch.stack([state for state in hidden_states_list[:update_length]])
            assert isinstance(history_hidden, torch.Tensor), "history_hidden should be a tensor"
            max_len = max(current_hidden.shape[0], history_hidden.shape[0])

            if current_hidden.shape[0] < max_len:
                pad_len = max_len - current_hidden.shape[0]
                if len(current_hidden.shape) == 3:
                    padding = torch.zeros(pad_len, current_hidden.shape[1], current_hidden.shape[2],
                                          device=current_hidden.device, dtype=current_hidden.dtype)
                else:
                    padding = torch.zeros(pad_len, current_hidden.shape[1], device=current_hidden.device,
                                          dtype=current_hidden.dtype)
                current_hidden_padded = torch.cat([current_hidden, padding], dim=0)
            else:
                current_hidden_padded = current_hidden

            if history_hidden.shape[0] < max_len:
                pad_len = max_len - history_hidden.shape[0]
                if len(history_hidden.shape) == 3:
                    padding = torch.zeros(pad_len, history_hidden.shape[1], history_hidden.shape[2],
                                          device=history_hidden.device, dtype=history_hidden.dtype)
                else:
                    padding = torch.zeros(pad_len, history_hidden.shape[1], device=history_hidden.device,
                                          dtype=history_hidden.dtype)
                history_hidden_padded = torch.cat([history_hidden, padding], dim=0)
            else:
                history_hidden_padded = history_hidden

            fused_hidden = alpha * current_hidden_padded + (1 - alpha) * history_hidden_padded

            original_len = min(current_hidden.shape[0], len(hidden_states_list[:update_length]))
            for idx in range(original_len):
                hidden_states_list[idx] = fused_hidden[idx]

        optimized_output, reward_history, new_original_length, new_optimized_length, new_update_length = optimized_generation(
            reward_model=reward_model,
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=example["question"],
            input_text=example["formatted"],
            original_answer=original_output,
            original_hidden_states_list=hidden_states_list,
            input_ids=input_ids,
            max_num_steps=args.max_num_steps,
            lr=args.lr,
            grad_clip=args.grad_clip,
            k=args.k,
            reward_threshold=args.reward_threshold,
        )

        update_count += (len(reward_history) - 1)

        is_ori_correct, check_it = is_answer_equal(original_output, true_answer, data_name)
        is_opt_correct, check_it = is_answer_equal(optimized_output, true_answer, data_name)

        if is_ori_correct:
            original_correct_add = True
        else:
            original_correct_add = False
        if is_opt_correct:
            optimized_correct_add = True
        else:
            optimized_correct_add = False

        original_correct += original_correct_add
        optimized_correct += optimized_correct_add

        total += 1

        original_length += new_original_length
        optimized_length += new_optimized_length
        fitten_length += (new_optimized_length - new_update_length) if len(reward_history) > 1 else 0

        if new_update_length > 0 and optimized_correct_add:
            optimized_hidden = hidden_states_list[:new_update_length]
            prompt_text = example['question']
            opt_latent = [h.squeeze(0).cpu().tolist() for h in optimized_hidden]
            latent_data.append((prompt_text, orig_latent, opt_latent))

    os.makedirs("json", exist_ok=True)
    with open(args.output_json_path, "w", encoding="utf-8") as f:
        json.dump(latent_data, f, ensure_ascii=False, indent=2)

    print(f"Daytime accuracy: {optimized_correct / total:.4f}")


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)