import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

def parse_args():
    parser = argparse.ArgumentParser(description='Latent Model Training')

    parser.add_argument('--model_path', type=str, 
                        default='../../models/qwen/Qwen2___5-1___5B-Instruct',
                        help='Path to the base model')
    parser.add_argument('--latent_json_path', type=str,
                        default='json/4b_latent_mmlu_1000.json',
                        help='Path to the latent JSON data file')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint for continuing training')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help='Max norm for gradient clipping')

    parser.add_argument('--out_dim', type=int, default=2560,
                        help='Output dimension for latent model')
    parser.add_argument('--num_tokens', type=int, default=15,
                        help='Number of tokens for latent model')
    parser.add_argument('--reduce_dim', type=int, default=512,
                        help='Reduced dimension for latent model')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum input length for tokenization')

    parser.add_argument('--use_reducer_for_target', action='store_true',
                        help='Use reducer for target in training')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue training from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). If None, auto-detect')

    parser.add_argument('--output_dir', type=str, 
                        default='../../models/latent',
                        help='Directory to save the trained model')
    parser.add_argument('--output_filename', type=str,
                        default='4b_latent_model_mmlu_1000.pt',
                        help='Filename for the saved model')
    
    return parser.parse_args()

class LatentDataset(Dataset):
    def __init__(self, latent_json_path, tokenizer, max_length=128):
        with open(latent_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, orig_latent, opt_latent = self.data[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        orig_latent = torch.tensor(orig_latent, dtype=torch.float32)
        opt_latent = torch.tensor(opt_latent, dtype=torch.float32)
        return input_ids, attention_mask, orig_latent, opt_latent

class LatentModel(nn.Module):
    def __init__(self, model_name_or_path, latent_dim, out_dim=2560, num_tokens=15, reduce_dim=512):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.reduce_dim = reduce_dim
        self.latent_reducer = nn.Linear(latent_dim, reduce_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size + reduce_dim, num_tokens * reduce_dim),
            nn.ReLU(),
        )
        self.projector = nn.Linear(reduce_dim, out_dim)

    def forward(self, input_ids, attention_mask, orig_latent):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        orig_latent_pooled = orig_latent.mean(dim=1)
        orig_latent_reduced = self.latent_reducer(orig_latent_pooled)
        fusion_input = torch.cat([pooled, orig_latent_reduced], dim=-1)
        pred_latent = self.fusion(fusion_input)
        pred_latent = pred_latent.view(-1, self.num_tokens, self.reduce_dim)
        pred_emb = self.projector(pred_latent)
        return pred_latent, pred_emb

def collate_fn(batch):
    input_ids, attention_mask, orig_latent, opt_latent = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    orig_latent = [torch.tensor(x, dtype=torch.float32) for x in orig_latent]
    opt_latent = [torch.tensor(x, dtype=torch.float32) for x in opt_latent]
    orig_latent_padded = pad_sequence(orig_latent, batch_first=True)
    opt_latent_padded = pad_sequence(opt_latent, batch_first=True)
    return input_ids, attention_mask, orig_latent_padded, opt_latent_padded

def train(args):
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.model_path))
    latent_json_path = os.path.join(os.path.dirname(__file__), args.latent_json_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open(latent_json_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)[0]
        latent_dim = len(sample[1][0])
    dataset = LatentDataset(latent_json_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = LatentModel(model_path, latent_dim, out_dim=args.out_dim, num_tokens=args.num_tokens, reduce_dim=args.reduce_dim).to(device)

    if args.continue_training and args.checkpoint_path:
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.checkpoint_path))
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if torch.cuda.device_count() > 1:
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model.train()
    scaler = GradScaler()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for input_ids, attention_mask, orig_latent, opt_latent in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            orig_latent = orig_latent.to(device)
            opt_latent = opt_latent.to(device)
            max_k = opt_latent.shape[1]
            if max_k < 15:
                pad = torch.zeros(opt_latent.shape[0], 15-max_k, opt_latent.shape[2], device=opt_latent.device, dtype=opt_latent.dtype)
                opt_latent_15 = torch.cat([opt_latent, pad], dim=1)
            else:
                opt_latent_15 = opt_latent[:, :15, :]
            with autocast(dtype=torch.bfloat16):
                pred_latent, _ = model(input_ids, attention_mask, orig_latent)
                if args.use_reducer_for_target:
                    if hasattr(model, 'module'):
                        target = model.module.latent_reducer(opt_latent_15)
                    else:
                        target = model.latent_reducer(opt_latent_15)
                else:
                    target = opt_latent_15
                if torch.isnan(pred_latent).any() or torch.isnan(target).any():
                    print("NaN detected in pred_latent or target")
                if torch.isinf(pred_latent).any() or torch.isinf(target).any():
                    print("Inf detected in pred_latent or target")
                loss = criterion(pred_latent, target)
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf loss detected! Skipping this batch.")
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
            del input_ids, attention_mask, orig_latent, opt_latent, pred_latent, opt_latent_15, loss, target
            torch.cuda.empty_cache()
        print(f'Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}')

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_filename)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()
