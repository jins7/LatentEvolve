import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm
from ori_generation import original_generation
from judge import *
from process_data import *
from nighttime_model import LatentModel
from collections import OrderedDict
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Latent Evolution Generation')

    parser.add_argument('--dataset_path', type=str, default="../../datasets/test/gsm8k.json",
                        help='Path to the dataset file')
    parser.add_argument('--k_ratio', type=float, default=0.2,
                        help='K ratio for sampling')

    parser.add_argument('--qwen_model_path', type=str, 
                        default="../../models/Qwen/Qwen3-4b-Instruct-2507",
                        help='Path to the Qwen model')
    parser.add_argument('--latent_model_path', type=str,
                        default="../../models/latent/4b_latent_model_mmlu_750.pt",
                        help='Path to the latent model checkpoint')
    parser.add_argument('--latent_model_base', type=str,
                        default="../../models/qwen/Qwen2___5-1___5B-Instruct",
                        help='Path to the latent model base')

    parser.add_argument('--k', type=int, default=15,
                        help='Number of latent vectors to use')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum input length for tokenization')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate')

    parser.add_argument('--out_dim', type=int, default=2560,
                        help='Output dimension for latent model')
    parser.add_argument('--num_tokens', type=int, default=15,
                        help='Number of tokens for latent model')
    parser.add_argument('--reduce_dim', type=int, default=512,
                        help='Reduced dimension for latent model')

    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). If None, auto-detect')
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.device is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    qwen_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.qwen_model_path))
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
    qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_path, torch_dtype=torch.bfloat16, device_map=DEVICE)
    qwen_model.eval()

    latent_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.latent_model_path))
    latent_model_base = os.path.abspath(os.path.join(os.path.dirname(__file__), args.latent_model_base))
    latent_tokenizer = AutoTokenizer.from_pretrained(latent_model_base)
    latent_model = None

    dataset = Dataset.from_dict(process_json_dataset(args.dataset_path, tokenizer=qwen_tokenizer))
    print(f"Loaded dataset size: {len(dataset)}")

    correct = 0
    total = 0

    for i in tqdm(range(0, len(dataset)), desc="Evaluating"):
        example = dataset[i]
        prompt = example["question"] + "\nPlease provide a step-by-step answer, and must put your final answer within \\boxed{}"
        # prompt = example["question"] + "\nPlease step-by-step answer the following multiple-choice question by selecting the correct option (A, B, C or D). Your response must ends up with the following format:The correct answer is {your answer option letter here}."
        raw_prompt = example["question"]
        formatted_prompt = example["formatted"]
        with torch.no_grad():
            original_output, hidden_states_list, input_ids = original_generation(
                input_text=formatted_prompt,
                model=qwen_model,
                tokenizer=qwen_tokenizer,
                device=DEVICE)
        latent_dim = hidden_states_list[0].squeeze(0).shape[-1]
        k = args.k
        orig_latent = [h.squeeze(0).cpu().tolist() for h in hidden_states_list[:k]]
        if len(orig_latent) < k:
            pad = [[0.0] * latent_dim for _ in range(k - len(orig_latent))]
            orig_latent_15 = orig_latent + pad
        else:
            orig_latent_15 = orig_latent[:k]
        orig_latent_tensor = torch.tensor(orig_latent_15, dtype=torch.float32).unsqueeze(0)
        latent_input = latent_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length, padding='max_length')
        input_ids = latent_input['input_ids']
        attention_mask = latent_input['attention_mask']
        if latent_model is None:
            latent_model = LatentModel(latent_model_base, latent_dim, out_dim=args.out_dim, num_tokens=args.num_tokens, reduce_dim=args.reduce_dim)
            state_dict = torch.load(latent_model_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[new_key] = v
            latent_model.load_state_dict(new_state_dict)
            latent_model.eval()
        with torch.no_grad():
            pred_latent, pred_emb = latent_model(input_ids, attention_mask, orig_latent_tensor)
        prompt_input_ids = qwen_tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        with torch.no_grad():
            prompt_emb = qwen_model.model.embed_tokens(prompt_input_ids)
        full_emb = torch.cat([pred_emb.to(DEVICE), prompt_emb], dim=1).to(torch.bfloat16)
        pad_token_id = qwen_tokenizer.pad_token_id if qwen_tokenizer.pad_token_id is not None else qwen_tokenizer.eos_token_id
        prompt_attention_mask = (prompt_input_ids != pad_token_id).long()
        latent_attention_mask = torch.ones((1, pred_emb.shape[1]), dtype=prompt_attention_mask.dtype, device=prompt_attention_mask.device)
        full_attention_mask = torch.cat([latent_attention_mask, prompt_attention_mask], dim=1)
        with torch.no_grad():
            output_ids = qwen_model.generate(inputs_embeds=full_emb, attention_mask=full_attention_mask, max_new_tokens=args.max_new_tokens)
            output_text = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        true_answer = example["answer"]
        judge, check_instruction = is_answer_equal(output_text, true_answer, args.dataset_path)
        true_str = extract_answer_str(true_answer, args.dataset_path)
        math_score = compute_score(output_text, true_str)

        final_judge = judge or (math_score >= 1.0)

        print(f"Prompt: {prompt}\nGenerated: {output_text}\nTrue Answer: {true_answer}\nFinal Judge: {final_judge}\n{'-'*40}")
        total += 1
        if final_judge:
            correct += 1
    print(f"Final accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    main()

