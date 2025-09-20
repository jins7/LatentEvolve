python evolve_generation.py \
    --dataset_path "../../datasets/test/gsm8k.json" \
    --qwen_model_path "../../models/Qwen/Qwen3-4b-Instruct-2507" \
    --latent_model_path "../../models/latent/evolve_model.pt" \
    --latent_model_base "../../models/qwen/Qwen2___5-1___5B-Instruct" \
    --k 20 \
    --max_length 256 \
    --max_new_tokens 1024 \
    --out_dim 2048 \
    --num_tokens 20 \
    --reduce_dim 256 \
    --device "cuda"

