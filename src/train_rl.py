import torch
import torch.optim as optim
import torch.nn.utils as utils 
import numpy as np
import random
import mlflow
import mlflow.pytorch
from torch.distributions import Categorical

# Import project-specific classes
from dataset import ArithmeticTokenizer, ArithmeticDataset
from model import MiniTransformer

# --- Configuration ---
LR = 1e-5                
EPISODES = 7000          
BATCH_SIZE = 128         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURE = 1.0        
ENTROPY_COEF = 0.01      
REPLAY_BUFFER_SIZE = 500 
REPLAY_PROB = 0.25       

def compute_reward(prompt_str, gen_str, correct_val, reverse_target=True):
    """
    Denser Reward Shaping: Rewards matching digits from right-to-left.
    """
    try:
        pred_raw = gen_str.split('=')[-1].strip()
        pred_clean = "".join([c for c in pred_raw if c.isdigit() or c == '-'])
        
        if not pred_clean or pred_clean == '-': 
            return -0.1
            
        pred_val_str = pred_clean[::-1] if reverse_target else pred_clean
        pred_val = int(pred_val_str)
        
        if pred_val == correct_val:
            return 1.0 
            
        reward = 0.0
        target_str = str(abs(correct_val))
        
        # Compare units-first
        p_units = pred_clean if reverse_target else pred_clean[::-1]
        t_units = target_str[::-1]
        
        for i in range(min(len(p_units), len(t_units))):
            if p_units[i] == t_units[i]:
                reward += 0.15
            else:
                break 
        
        return min(reward, 0.9)
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
    elif '*' in lhs:
        parts = lhs.split('*')
        return int(parts[0]) * int(parts[1])
    return 0

def select_dataset(episode, ds_easy, ds_med, ds_hard):
    if episode < 1200: return ds_easy, "Easy"
    elif episode < 1700:
        prob = (episode - 1200) / 500.0
        return (ds_med if np.random.rand() < prob else ds_easy), "Mix E->M"
    elif episode < 3200: return ds_med, "Med"
    elif episode < 3700:
        prob = (episode - 3200) / 500.0
        return (ds_hard if np.random.rand() < prob else ds_med), "Mix M->H"
    else: return ds_hard, "Hard"

def train_rl():
    mlflow.set_experiment("Arithmetic_LLM_Scaling")
    
    with mlflow.start_run(run_name="RL_Finetuning_Phase"):
        print(f"--- Starting RL with Prioritized Replay & MLflow on {DEVICE} ---")
        
        # Log Hyperparameters
        mlflow.log_params({
            "lr": LR,
            "episodes": EPISODES,
            "batch_size": BATCH_SIZE,
            "temp": TEMPERATURE,
            "entropy_coef": ENTROPY_COEF,
            "replay_prob": REPLAY_PROB,
            "reverse_target": True
        })

        tokenizer = ArithmeticTokenizer()
        # Scale: Must match your updated model config (e.g., 8 layers, 256 dim)
        model = MiniTransformer(tokenizer, embed_dim=256, num_heads=8, num_layers=8, max_len=32).to(DEVICE)
        
        try:
            model.load_state_dict(torch.load("pretrained_arithmetic.pth"))
            print("✅ Loaded model weights.")
        except:
            print("❌ Could not load weights. Starting random.")

        optimizer = optim.AdamW(model.parameters(), lr=LR)
        model.train()
        
        ds_easy = ArithmeticDataset(tokenizer, num_samples=10000, max_digits=2, reverse_target=True)
        ds_med  = ArithmeticDataset(tokenizer, num_samples=10000, max_digits=3, reverse_target=True)
        ds_hard = ArithmeticDataset(tokenizer, num_samples=20000, max_digits=4, reverse_target=True)
        
        hard_buffer = [] 
        running_reward = 0.0
        batch_loss = 0
        batch_count = 0
        optimizer.zero_grad()

        for episode in range(1, EPISODES + 1):
            ds, phase = select_dataset(episode, ds_easy, ds_med, ds_hard)
            
            is_replay = False
            if len(hard_buffer) > 50 and np.random.rand() < REPLAY_PROB:
                item = random.choice(hard_buffer)
                is_replay = True
                phase = "Replay"
            else:
                idx = np.random.randint(0, len(ds))
                item = ds[idx]

            prompt_ids = item['prompt_ids'].to(DEVICE).unsqueeze(0)
            prompt_str = item['prompt_str']
            correct_val = get_correct_val(prompt_str)
            
            # Generation
            curr_ids = prompt_ids
            log_probs, entropies, actions = [], [], []
            
            for _ in range(10):
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
            reward = compute_reward(prompt_str, generated_str, correct_val, reverse_target=True)
            
            if reward < 0.99 and not is_replay:
                hard_buffer.append(item)
                if len(hard_buffer) > REPLAY_BUFFER_SIZE:
                    hard_buffer.pop(0)

            running_reward = 0.05 * reward + 0.95 * running_reward
            advantage = reward - running_reward
            
            # Log metrics
            mlflow.log_metric("episode_reward", reward, step=episode)
            mlflow.log_metric("running_reward", running_reward, step=episode)

            if len(log_probs) > 0:
                policy_loss = [-lp * advantage for lp in log_probs]
                p_loss = torch.stack(policy_loss).sum()
                e_loss = -ENTROPY_COEF * torch.stack(entropies).sum()
                total_loss = p_loss + e_loss
                batch_loss += total_loss
                batch_count += 1

            if batch_count >= BATCH_SIZE:
                avg_batch_loss = (batch_loss / BATCH_SIZE)
                avg_batch_loss.backward()
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                mlflow.log_metric("batch_loss", avg_batch_loss.item(), step=episode)
                batch_loss, batch_count = 0, 0

            if episode % 100 == 0:
                print(f"{episode:<6} | {phase:<8} | Rw: {running_reward:.4f} | {prompt_str} {generated_str}")

        # Finalize and Log Model to UI
        torch.save(model.state_dict(), "rl_arithmetic_replay.pth")
        mlflow.pytorch.log_model(model, artifact_path="model") 
        mlflow.log_artifact("rl_arithmetic_replay.pth")
        print("\nRL Training Complete. Model and metrics logged to MLflow.")

if __name__ == "__main__":
    train_rl()