"""
RL Fine-Tuning for Arithmetic LLM - v3 with Scratchpad CoT

Features:
- Config-driven hyperparameters from configs/hyperparams.yaml
- Scratchpad step bonus in reward computation
- Corrected curriculum: 2→3→4 digit (no 5-digit in v3)
- Temperature and entropy coefficient scheduling
- Replay buffer with 2000 capacity
- Mixed precision training (AMP)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import numpy as np
import random
import mlflow
import mlflow.pytorch
from torch.distributions import Categorical

from src.config import get_config
from src.dataset import ArithmeticTokenizer, ArithmeticDataset, compute_scratchpad_reward
from src.model import ArithTransformer


# Global config (needed for nested functions)
CONFIG = None


def get_correct_val(prompt_str):
    """Extract correct answer from prompt string."""
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
    """Select dataset based on 5-phase curriculum (v3: 2→3→4 digit only)."""
    curriculum = CONFIG.rl.curriculum
    
    # Phase 1: Easy (2-digit)
    p1 = curriculum.get('phase1', {})
    if episode <= p1.end_episode:
        return ds_easy, "Easy"
    
    # Mix 1: Easy → Medium (1000 episodes)
    m1 = curriculum.get('mix1', {})
    if episode <= m1.end_episode:
        duration = m1.end_episode - m1.start_episode
        prob = min(1.0, (episode - m1.start_episode) / duration)
        return (ds_med if np.random.rand() < prob else ds_easy), f"Mix E->M ({prob:.1f})"
    
    # Phase 2: Medium (3-digit)
    p2 = curriculum.get('phase2', {})
    if episode <= p2.end_episode:
        return ds_med, "Medium"
    
    # Mix 2: Medium → Hard (1000 episodes)
    m2 = curriculum.get('mix2', {})
    if episode <= m2.end_episode:
        duration = m2.end_episode - m2.start_episode
        prob = min(1.0, (episode - m2.start_episode) / duration)
        return (ds_hard if np.random.rand() < prob else ds_med), f"Mix M->H ({prob:.1f})"
    
    # Phase 3: Hard (4-digit)
    p3 = curriculum.get('phase3', {})
    if episode <= p3.end_episode:
        return ds_hard, "Hard"
    
    # Default: Hard
    return ds_hard, "Hard"


def get_scheduled_value(start_val, end_val, episode, total_episodes):
    """Linear schedule from start_val to end_val."""
    progress = min(1.0, episode / total_episodes)
    return start_val + (end_val - start_val) * progress


def compute_reward_v3(prompt_str, gen_str, correct_val):
    """Compute reward for v3 scratchpad generation."""
    total, step_bonus, answer = compute_scratchpad_reward(prompt_str, gen_str, correct_val)
    return total, step_bonus, answer


def train_rl():
    global CONFIG
    CONFIG = get_config()
    
    mlflow.set_experiment("Arithmetic_LLM_Scaling_v3")
    with mlflow.start_run(run_name="RL_Finetuning_v3_Scratchpad"):
        mlflow.log_params({
            "rl/learning_rate": CONFIG.rl.learning_rate,
            "rl/total_episodes": CONFIG.rl.total_episodes,
            "rl/batch_size": CONFIG.rl.batch_size,
            "rl/temperature_start": CONFIG.rl.temperature,
            "rl/temperature_end": CONFIG.rl.temperature_end,
            "rl/entropy_coef_start": CONFIG.rl.entropy_coef,
            "rl/entropy_coef_end": CONFIG.rl.entropy_coef_end,
            "rl/max_grad_norm": CONFIG.rl.max_grad_norm,
            "rl/replay_buffer_size": CONFIG.rl.replay_buffer.size,
            "rl/max_new_tokens": CONFIG.rl.max_new_tokens,
            "rl/scratchpad_step_bonus": CONFIG.rl.reward.scratchpad_step_bonus,
        })
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"RL Training on: {device}")
        
        # Initialize tokenizer (v3 with 21 tokens)
        tokenizer = ArithmeticTokenizer(vocab_size=21)
        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
        
        # Initialize model (v3 with vocab_size=21, max_len=128)
        model = ArithTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=CONFIG.model.embed_dim,
            num_heads=CONFIG.model.num_heads,
            num_layers=CONFIG.model.num_layers,
            dim_feedforward=CONFIG.model.dim_feedforward,
            max_len=CONFIG.model.max_len,
            dropout=CONFIG.model.dropout
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        mlflow.log_param("model_params", sum(p.numel() for p in model.parameters()))
        
        # Load pretrained weights if available
        pretrained_path = CONFIG.paths.pretrained_model
        if os.path.exists(pretrained_path):
            try:
                model.load_state_dict(torch.load(pretrained_path, map_location=device))
                print(f"✅ Loaded pretrained weights from: {pretrained_path}")
            except Exception as e:
                print(f"❌ Could not load weights: {e}")
                print("   Starting from random initialization")
        else:
            print(f"⚠️ Pretrained model not found at: {pretrained_path}")
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG.rl.learning_rate)
        model.train()
        
        # Create datasets (CORRECTED: only 3 levels, no 5-digit)
        ds_easy = ArithmeticDataset(
            tokenizer, 
            num_samples=CONFIG.rl.dataset_sizes["easy"], 
            max_digits=2, 
            reverse_target=False,
            use_scratchpad=True
        )
        ds_med = ArithmeticDataset(
            tokenizer, 
            num_samples=CONFIG.rl.dataset_sizes["medium"], 
            max_digits=3, 
            reverse_target=False,
            use_scratchpad=True
        )
        ds_hard = ArithmeticDataset(
            tokenizer, 
            num_samples=CONFIG.rl.dataset_sizes["hard"], 
            max_digits=4, 
            reverse_target=False,
            use_scratchpad=True
        )
        
        print(f"Datasets: Easy={len(ds_easy)}, Medium={len(ds_med)}, Hard={len(ds_hard)}")
        print(f"Curriculum: 2→3→4 digit (no 5-digit in v3)")
        
        # Replay buffer (2000 for v3)
        hard_buffer = []
        
        # Training state
        running_reward = 0.0
        batch_loss = 0
        batch_count = 0
        optimizer.zero_grad()
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda') if CONFIG.model.use_mixed_precision else None
        
        total_episodes = CONFIG.rl.total_episodes
        
        print(f"\nStarting v3 RL training for {total_episodes} episodes...")
        print(f"Temperature: {CONFIG.rl.temperature} → {CONFIG.rl.temperature_end}")
        print(f"Entropy coef: {CONFIG.rl.entropy_coef} → {CONFIG.rl.entropy_coef_end}")
        print(f"Max new tokens: {CONFIG.rl.max_new_tokens}")
        print("-" * 70)
        
        for episode in range(1, total_episodes + 1):
            # Get scheduled values
            temperature = get_scheduled_value(
                CONFIG.rl.temperature, 
                CONFIG.rl.temperature_end, 
                episode, 
                total_episodes
            )
            entropy_coef = get_scheduled_value(
                CONFIG.rl.entropy_coef,
                CONFIG.rl.entropy_coef_end,
                episode,
                total_episodes
            )
            
            # Select dataset based on 5-phase curriculum
            ds, phase = select_dataset(episode, ds_easy, ds_med, ds_hard)
            
            # Replay buffer sampling
            is_replay = False
            if len(hard_buffer) > 50 and np.random.rand() < CONFIG.rl.replay_buffer.sampling_prob:
                item = random.choice(hard_buffer)
                is_replay = True
                phase = "Replay"
            else:
                idx = np.random.randint(0, len(ds))
                item = ds[idx]
            
            prompt_ids = item['prompt_ids'].to(device).unsqueeze(0)
            prompt_str = item['prompt_str']
            correct_val = get_correct_val(prompt_str)
            
            # Generation with model (no gradients to save memory)
            curr_ids = prompt_ids
            actions_list = []
            
            with torch.no_grad():
                for _ in range(CONFIG.rl.max_new_tokens):
                    if CONFIG.model.use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            logits, _ = model(curr_ids)
                    else:
                        logits, _ = model(curr_ids)
                    
                    next_token_logits = logits[:, -1, :] / temperature
                    m = Categorical(logits=next_token_logits)
                    action = m.sample()
                    
                    actions_list.append(action.item())
                    curr_ids = torch.cat([curr_ids, action.unsqueeze(1)], dim=1)
                    if action.item() == tokenizer.eos_token_id:
                        break
            
            generated_str = tokenizer.decode(actions_list)
            reward, step_bonus, answer_reward = compute_reward_v3(prompt_str, generated_str, correct_val)
            
            # Update replay buffer
            if reward < CONFIG.rl.replay_buffer.threshold and not is_replay:
                hard_buffer.append(item)
                if len(hard_buffer) > CONFIG.rl.replay_buffer.size:
                    hard_buffer.pop(0)
            
            # EMA baseline for advantage
            running_reward = CONFIG.rl.baseline_smoothing * reward + (1 - CONFIG.rl.baseline_smoothing) * running_reward
            advantage = reward - running_reward
            
            # Log metrics
            mlflow.log_metric("episode_reward", reward, step=episode)
            mlflow.log_metric("running_reward", running_reward, step=episode)
            mlflow.log_metric("scratchpad_step_bonus", step_bonus, step=episode)
            mlflow.log_metric("temperature", temperature, step=episode)
            mlflow.log_metric("entropy_coef", entropy_coef, step=episode)
            
            # Policy loss (one forward pass with gradients)
            if len(actions_list) > 0:
                if CONFIG.model.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        all_logits, _ = model(curr_ids)
                else:
                    all_logits, _ = model(curr_ids)
                
                # Get logits for generated tokens
                # Prediction for action at index N is at logit index N-1
                start_idx = prompt_ids.size(1) - 1
                end_idx = curr_ids.size(1) - 1
                relevant_logits = all_logits[:, start_idx:end_idx, :] / temperature
                
                # Target actions
                actions_tensor = curr_ids[:, start_idx+1:].squeeze(0)
                
                # Distribution
                m = Categorical(logits=relevant_logits.squeeze(0))
                log_probs = m.log_prob(actions_tensor)
                entropies = m.entropy()
                
                # REINFORCE loss: -Advantage * log_p
                p_loss = (-log_probs * advantage).sum()
                e_loss = (-entropy_coef * entropies).sum()
                total_loss = p_loss + e_loss
                
                if CONFIG.model.use_mixed_precision:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                batch_loss += total_loss.item()
                batch_count += 1
            
            # Gradient update
            if batch_count >= CONFIG.rl.batch_size:
                if CONFIG.model.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    utils.clip_grad_norm_(model.parameters(), CONFIG.rl.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    utils.clip_grad_norm_(model.parameters(), CONFIG.rl.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                mlflow.log_metric("batch_loss", batch_loss / batch_count, step=episode)
                batch_loss, batch_count = 0, 0
            
            # Logging
            if episode % 100 == 0:
                print(f"Ep {episode:<7} | {phase:<15} | "
                      f"Rw: {running_reward:.3f} | "
                      f"StpBon: {step_bonus:.2f} | "
                      f"T: {temperature:.2f} | "
                      f"{prompt_str[:20]} → {generated_str[:40]}...")
        
        # Save final model
        os.makedirs(CONFIG.paths.checkpoint_dir, exist_ok=True)
        model_path = CONFIG.paths.rl_model
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "rl_model")
        mlflow.log_artifact(model_path)
        
        print(f"\n✅ RL Training complete. Model saved to: {model_path}")
        print(f"   Final running reward: {running_reward:.4f}")


if __name__ == "__main__":
    train_rl()