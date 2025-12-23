import torch
import torch.optim as optim
import torch.nn.utils as utils 
import numpy as np
from torch.distributions import Categorical
import random

# Assuming these exist in your project files
from dataset import ArithmeticTokenizer, ArithmeticDataset
from model import MiniTransformer

# --- Configuration ---
LR = 1e-5                
EPISODES = 7000          
BATCH_SIZE = 128         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE = 1.0        # High temp for exploration
ENTROPY_COEF = 0.01      # Prevent mode collapse
REPLAY_BUFFER_SIZE = 500 # Keep last 500 hard problems
REPLAY_PROB = 0.25       # 25% chance to retrain on a hard problem

def compute_reward(prompt_str, generated_answer_str, correct_val):
    try:
        ans_only = generated_answer_str.replace(prompt_str, "").strip()
        if not ans_only: return -0.1
            
        clean_str = "".join([c for c in ans_only if c.isdigit() or c == '-'])
        if not clean_str or clean_str == '-': return -0.1

        pred_val = int(clean_str)
        
        # 1. Exact Match
        if pred_val == correct_val:
            return 1.0 
            
        # 2. Partial Credit (Reward Shaping)
        reward = 0.0
        # Correct number of digits (e.g. 4 digit answer for 4 digit sum)
        if len(str(abs(pred_val))) == len(str(abs(correct_val))):
            reward += 0.1
        
        return reward
    except:
        return -0.1

def get_correct_val(prompt_str):
    lhs = prompt_str.replace('=', '')
    if '+' in lhs:
        parts = lhs.split('+')
        return int(parts[0]) + int(parts[1])
    elif '-' in lhs:
        parts = lhs.split('-')
        return int(parts[0]) - int(parts[1])
    return 0

def select_dataset(episode, ds_easy, ds_med, ds_hard):
    # Phase 1: Easy -> Med (1200-1700)
    if episode < 1200: return ds_easy, "Easy"
    elif episode < 1700:
        prob = (episode - 1200) / 500.0
        return (ds_med if np.random.rand() < prob else ds_easy), "Mix E->M"
        
    # Phase 2: Med -> Hard (3200-3700)
    elif episode < 3200: return ds_med, "Med"
    elif episode < 3700:
        prob = (episode - 3200) / 500.0
        return (ds_hard if np.random.rand() < prob else ds_med), "Mix M->H"
        
    else: return ds_hard, "Hard"

def train_rl():
    print(f"--- Starting RL with Prioritized Replay on {DEVICE} ---")
    tokenizer = ArithmeticTokenizer()
    model = MiniTransformer(tokenizer, embed_dim=192, num_heads=6, num_layers=6).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("pretrained_arithmetic.pth"))
        print("✅ Loaded model weights.")
    except:
        print("❌ Could not load weights. Starting random.")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    model.train()
    
    ds_easy = ArithmeticDataset(tokenizer, num_samples=10000, max_digits=2)
    ds_med  = ArithmeticDataset(tokenizer, num_samples=10000, max_digits=3)
    ds_hard = ArithmeticDataset(tokenizer, num_samples=20000, max_digits=4)
    
    # Replay Buffer for Hard Examples
    hard_buffer = [] 
    
    running_reward = 0.0
    reward_history = []
    
    batch_loss = 0
    batch_count = 0
    optimizer.zero_grad()

    for episode in range(1, EPISODES + 1):
        # 1. Select Dataset (Curriculum)
        ds, phase = select_dataset(episode, ds_easy, ds_med, ds_hard)
        
        # 2. Logic: New Sample OR Replay Hard Sample?
        is_replay = False
        if len(hard_buffer) > 50 and np.random.rand() < REPLAY_PROB:
            # Replay a past failure!
            item = random.choice(hard_buffer)
            is_replay = True
            phase = "Replay"
        else:
            # Standard new sample
            idx = np.random.randint(0, len(ds))
            item = ds[idx]

        prompt_ids = item['prompt_ids'].to(DEVICE).unsqueeze(0)
        prompt_str = item['prompt_str']
        correct_val = get_correct_val(prompt_str)
        
        # 3. Generate
        curr_ids = prompt_ids
        log_probs = []
        entropies = []
        actions = []
        
        for _ in range(9): # Increased to 9 for 4-digit sums
            logits, _ = model(curr_ids)
            next_token_logits = logits[:, -1, :] / TEMPERATURE
            m = Categorical(logits=next_token_logits)
            action = m.sample()
            
            log_probs.append(m.log_prob(action))
            entropies.append(m.entropy())
            actions.append(action.item())
            
            curr_ids = torch.cat([curr_ids, action.unsqueeze(1)], dim=1)
            if action.item() == tokenizer.eos_token_id: break
        
        generated_str = tokenizer.decode(actions)
        reward = compute_reward(prompt_str, generated_str, correct_val)
        
        # 4. Update Replay Buffer
        # If model failed (reward < 1.0) and it's NOT already a replay step, save it
        if reward < 0.99 and not is_replay:
            hard_buffer.append(item)
            if len(hard_buffer) > REPLAY_BUFFER_SIZE:
                hard_buffer.pop(0) # Remove oldest

        # Smooth Reward logging
        running_reward = 0.05 * reward + 0.95 * running_reward
        reward_history.append(running_reward)

        # 5. Calculate Loss
        if len(log_probs) > 0:
            # Advantage: Be better than average!
            advantage = reward - running_reward
            
            policy_loss = []
            for lp in log_probs:
                policy_loss.append(-lp * advantage)
            
            p_loss = torch.stack(policy_loss).sum()
            e_loss = -ENTROPY_COEF * torch.stack(entropies).sum()
            
            total_loss = p_loss + e_loss
            batch_loss += total_loss
            batch_count += 1

        # 6. Optimization Step
        if batch_count >= BATCH_SIZE:
            (batch_loss / BATCH_SIZE).backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            batch_loss = 0
            batch_count = 0

        if episode % 100 == 0:
            print(f"{episode:<6} | {phase:<8} | Rw: {running_reward:.4f} | {prompt_str} {generated_str}")

    torch.save(model.state_dict(), "rl_arithmetic_replay.pth")
    np.save("rl_rewards_replay.npy", np.array(reward_history))
    print("\nRL Training Complete.")

if __name__ == "__main__":
    train_rl()