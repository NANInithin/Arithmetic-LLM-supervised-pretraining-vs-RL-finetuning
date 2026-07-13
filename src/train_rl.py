"""
RL Fine-Tuning for Arithmetic LLM - v4 with Scratchpad CoT

Features:
- Config-driven hyperparameters from configs/hyperparams.yaml
- Scratchpad step bonus in reward computation
- 7-phase curriculum: 2→3→4→5 digit
- Temperature and entropy coefficient scheduling
- Replay buffer with 4000 capacity
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


def select_dataset(episode, ds_easy, ds_med, ds_hard, ds_very_hard):
    """Select dataset based on 7-phase curriculum (v4: 2→3→4→5 digit)."""
    curriculum = CONFIG.rl.curriculum

    # curriculum maps phase-name -> CurriculumPhase dataclass (NOT a dict), so
    # read its bounds via attributes. Helpers tolerate a missing phase.
    def _end(phase_name):
        phase = curriculum.get(phase_name)
        return phase.end_episode if phase is not None else 0

    def _mix_prob(phase_name):
        phase = curriculum.get(phase_name)
        if phase is None:
            return 1.0
        start, end = phase.start_episode, phase.end_episode
        duration = max(1, end - start)
        return min(1.0, (episode - start) / duration)

    # Phase 1: Easy (2-digit)
    if episode <= _end('phase1'):
        return ds_easy, "Easy"

    # Mix 1: Easy → Medium
    if episode <= _end('mix1'):
        prob = _mix_prob('mix1')
        return (ds_med if np.random.rand() < prob else ds_easy), f"Mix E->M ({prob:.1f})"

    # Phase 2: Medium (3-digit)
    if episode <= _end('phase2'):
        return ds_med, "Medium"

    # Mix 2: Medium → Hard
    if episode <= _end('mix2'):
        prob = _mix_prob('mix2')
        return (ds_hard if np.random.rand() < prob else ds_med), f"Mix M->H ({prob:.1f})"

    # Phase 3: Hard (4-digit)
    if episode <= _end('phase3'):
        return ds_hard, "Hard"

    # Mix 3: Hard → Very Hard
    if episode <= _end('mix3'):
        prob = _mix_prob('mix3')
        return (ds_very_hard if np.random.rand() < prob else ds_hard), f"Mix H->VH ({prob:.1f})"

    # Phase 4: Very Hard (5-digit)
    if episode <= _end('phase4'):
        return ds_very_hard, "Very Hard"

    # Default: Very Hard
    return ds_very_hard, "Very Hard"


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
    
    mlflow.set_experiment("Arithmetic_LLM_Scaling_v4")
    with mlflow.start_run(run_name="RL_Finetuning_v4_Scratchpad"):
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
            dropout=CONFIG.model.dropout,
            # Match supervised: honour the config flag (batch-1 RL uses trivial
            # memory either way, and skipping recompute is marginally faster).
            use_gradient_checkpointing=CONFIG.model.use_gradient_checkpointing,
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
        
        # Create datasets (v4: 4 levels including 5-digit very_hard)
        ds_easy = ArithmeticDataset(
            tokenizer,
            num_samples=CONFIG.rl.dataset_sizes.get("easy", 15000),
            max_digits=2,
            reverse_target=False,
            use_scratchpad=True
        )
        ds_med = ArithmeticDataset(
            tokenizer,
            num_samples=CONFIG.rl.dataset_sizes.get("medium", 15000),
            max_digits=3,
            reverse_target=False,
            use_scratchpad=True
        )
        ds_hard = ArithmeticDataset(
            tokenizer,
            num_samples=CONFIG.rl.dataset_sizes.get("hard", 30000),
            max_digits=4,
            reverse_target=False,
            use_scratchpad=True
        )
        ds_very_hard = ArithmeticDataset(
            tokenizer,
            num_samples=CONFIG.rl.dataset_sizes.get("very_hard", 30000),
            max_digits=5,
            reverse_target=False,
            use_scratchpad=True
        )
        
        print(f"Datasets: Easy={len(ds_easy)}, Medium={len(ds_med)}, Hard={len(ds_hard)}, Very Hard={len(ds_very_hard)}")
        print(f"Curriculum: 2→3→4→5 digit (7 phases)")
        
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
        
        print(f"\nStarting v4 RL training for {total_episodes} episodes...")
        print(f"Temperature: {CONFIG.rl.temperature} → {CONFIG.rl.temperature_end}")
        print(f"Entropy coef: {CONFIG.rl.entropy_coef} → {CONFIG.rl.entropy_coef_end}")
        print(f"Max new tokens: {CONFIG.rl.max_new_tokens}")
        print(f"Curriculum: 2→3→4→5 digit (7 phases)")
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
            
            # Select dataset based on 7-phase curriculum
            ds, phase = select_dataset(episode, ds_easy, ds_med, ds_hard, ds_very_hard)
            
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
        # NOTE: intentionally NOT calling mlflow.pytorch.log_model (version-fragile
        # serialization; nothing loads via mlflow.pytorch). See train_supervised.
        # The weights are still tracked in MLflow via log_artifact below.
        mlflow.log_artifact(model_path)
        
        print(f"\n✅ RL Training complete. Model saved to: {model_path}")
        print(f"   Final running reward: {running_reward:.4f}")


if __name__ == "__main__":
    train_rl()