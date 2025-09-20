import torch
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
def optimized_generation(
        reward_model, model, tokenizer, device,
        question, input_text, original_answer, 
        original_hidden_states_list, input_ids, start_index=0, 
        max_num_steps=10, lr=0.03, max_new_tokens=1024,
        grad_clip=None, k=0.1, reward_threshold=-0.2):

    eos_token = tokenizer.eos_token
    stop_words.append(eos_token)
    reward_history = []
    initial_reward = reward_model.get_reward(question, original_answer)
    
    print(f"-- Original Output: {original_answer} -- Initial Reward: {initial_reward}")
    reward_history.append(initial_reward)
    current_reward = initial_reward
    
    original_length = len(original_hidden_states_list)
    optimized_length = 0
    
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    base_input_ids = inputs.input_ids.clone()

    update_length = min(int(k * original_length), 300)
    if update_length <= 0:
        print("Update Length Zero!!!")
        final_answer = original_answer
        return final_answer, reward_history, original_length, optimized_length, update_length

    optimized_hidden_states = torch.nn.Parameter(torch.stack(
        [state.clone().detach().requires_grad_(True)
        for state in original_hidden_states_list[start_index: min(start_index + update_length, len(original_hidden_states_list))]])
    )

    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)
    
    original_seq = []
    original_seq.extend(input_ids[0][len(base_input_ids[-1]): len(base_input_ids[-1]) + start_index])
    
    input_ids = input_ids[:, : len(base_input_ids[-1]) + start_index]
    base_input_ids = input_ids.clone()
    new_answer = None

    for _ in range(max_num_steps):
        input_ids = base_input_ids.clone()
        if current_reward > reward_threshold:
            final_answer = new_answer if new_answer is not None else original_answer
            optimized_length = len(tokenizer.encode(final_answer))
            print(f"-- Final Answer: {final_answer}, -- Current Reward: {current_reward}")
            return final_answer, reward_history, original_length, optimized_length, update_length
        
        optimizer.zero_grad()
        
        logits = model.lm_head(optimized_hidden_states)
        probs = torch.softmax(logits, dim=-1) + 1e-8
        
        next_token_ids = torch.argmax(probs, dim=-1)
        next_token_ids = next_token_ids.squeeze(-1)
        log_pi_xz = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)

        loss = - current_reward * log_pi_xz.sum()
        print(f"-- Loss: {loss.item()}")
        loss.backward(retain_graph=True)
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        optimizer.step()

        generated_seq = []
        generated_seq.extend(original_seq)
        with torch.no_grad():
            next_tokens = torch.argmax(model.lm_head(optimized_hidden_states), 
                                       dim=-1)
            next_tokens = next_tokens.squeeze(-1)
            generated_seq.extend(next_tokens.tolist())
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)

        with torch.no_grad():
            cnt = 0
            while True:
                outputs = model.model(input_ids, output_hidden_states=True)
                hidden_states = outputs[0][:, -1]
                logits = model.lm_head(hidden_states)
                next_token_id = torch.argmax(logits, dim=-1)
                new_token = tokenizer.decode(next_token_id.item())
                generated_seq.append(next_token_id.item())
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                cnt += 1
                if new_token == eos_token:
                    break
                if cnt > max_new_tokens:
                    break
        del outputs, hidden_states, next_token_id, new_token
        del logits, next_tokens, input_ids
        torch.cuda.empty_cache()

        new_answer = tokenizer.decode(generated_seq)
        current_reward = reward_model.get_reward(question, new_answer)
        print(f"-- New Answer: {new_answer}, -- Current Reward: {current_reward}")
            
        reward_history.append(current_reward)
        
    final_answer = new_answer
    optimized_length = len(tokenizer.encode(final_answer))
    print(f"-- Final answer: {final_answer}")
    return final_answer, reward_history, original_length, optimized_length, update_length

