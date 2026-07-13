"""
Supervised Pretraining for Arithmetic LLM - v4 with Scratchpad CoT

Features:
- Config-driven hyperparameters from configs/hyperparams.yaml
- 5% validation split with early stopping
- Linear warmup + cosine decay LR scheduler
- Mixed precision training (AMP)
- Gradient checkpointing for memory efficiency
- Scratchpad chain-of-thought generation for multi-digit arithmetic
- Dynamic per-batch padding (no fixed pad_to_length for performance)
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
from torch.nn.utils.rnn import pad_sequence

from src.config import get_config
from src.dataset import ArithmeticTokenizer, ArithmeticDataset, compute_scratchpad_reward
from src.model import ArithTransformer


class PadCollate:
    """Picklable collate_fn that dynamically pads a batch to its longest seq.

    Defined at module level (not a lambda) so it can be pickled by DataLoader
    worker processes — required for num_workers > 0 on Windows (spawn) and Modal.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        return pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )


def _seed_worker(worker_id):
    """Give each DataLoader worker a distinct Python RNG seed.

    The dataset generates samples on the fly with the `random` module. Without
    this, all workers would inherit the same seed and produce duplicate data.
    torch derives a distinct base seed per worker, which we forward to `random`.
    """
    import random
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)


class CosineWarmupScheduler:
    """LR Scheduler: Linear warmup followed by cosine decay."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """Early stopping callback based on validation loss."""
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def train_supervised():
    CONFIG = get_config()
    
    mlflow.set_experiment("Arithmetic_LLM_Scaling_v4")
    with mlflow.start_run(run_name="Supervised_Pretraining_v4_Scratchpad"):
        mlflow.log_params({
            "embed_dim": CONFIG.model.embed_dim,
            "num_heads": CONFIG.model.num_heads,
            "num_layers": CONFIG.model.num_layers,
            "head_dim": CONFIG.model.head_dim,
            "dim_feedforward": CONFIG.model.dim_feedforward,
            "max_len": CONFIG.model.max_len,
            "vocab_size": CONFIG.model.vocab_size,
            "use_scratchpad": CONFIG.training.use_scratchpad,
            "batch_size": CONFIG.training.batch_size,
            "learning_rate": CONFIG.training.learning_rate,
            "warmup_steps": CONFIG.training.warmup_steps,
            "min_lr": CONFIG.training.min_lr,
            "epochs": CONFIG.training.epochs,
            "val_split": CONFIG.training.val_split,
            "num_samples": CONFIG.training.num_samples,
            "max_digits": CONFIG.training.max_digits_supervised,
            "gradient_accumulation_steps": CONFIG.training.gradient_accumulation_steps,
        })
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")
        mlflow.log_param("device", str(device))
        
        # Initialize tokenizer (v3 with 21 tokens)
        tokenizer = ArithmeticTokenizer(vocab_size=21)
        print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
        
        # Create dataset with dynamic padding (no pad_to_length)
        full_ds = ArithmeticDataset(
            tokenizer, 
            num_samples=CONFIG.training.num_samples,
            max_digits=CONFIG.training.max_digits_supervised,
            reverse_target=False,  # Use scratchpad mode
            use_scratchpad=CONFIG.training.use_scratchpad,
            # NOTE: pad_to_length removed for performance
        )
        
        # Split into train/val (5% validation)
        val_size = int(len(full_ds) * CONFIG.training.val_split)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], 
                                        generator=torch.Generator().manual_seed(42))
        
        print(f"Dataset: {train_size} train, {val_size} val samples")
        print(f"Using scratchpad: {CONFIG.training.use_scratchpad}")
        mlflow.log_param("train_samples", train_size)
        mlflow.log_param("val_samples", val_size)
        
        # Create data loaders with dynamic padding via collate_fn.
        # Parallel workers overlap the on-the-fly sample generation/tokenization
        # with GPU compute, so the A100 isn't starved waiting for batches.
        # persistent_workers keeps them alive across epochs (avoids re-spawn cost).
        num_workers = min(8, os.cpu_count() or 1)
        collate = PadCollate(tokenizer.pad_token_id)
        loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG.training.batch_size,
            shuffle=True,
            collate_fn=collate,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CONFIG.training.batch_size,
            shuffle=False,
            collate_fn=collate,
            **loader_kwargs,
        )
        
        # Initialize model (v3 with vocab_size=21, max_len=128)
        model = ArithTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=CONFIG.model.embed_dim,
            num_heads=CONFIG.model.num_heads,
            num_layers=CONFIG.model.num_layers,
            dim_feedforward=CONFIG.model.dim_feedforward,
            max_len=CONFIG.model.max_len,
            dropout=CONFIG.model.dropout,
            # Honour the config flag instead of the constructor default (True).
            # On a 40GB A100 the ~310M model uses only ~20GB, so checkpointing's
            # ~30% recompute overhead buys memory we don't need — turn it off in
            # hyperparams.yaml (model.use_gradient_checkpointing: false) to speed
            # up training. Re-enable if you ever hit an OOM.
            use_gradient_checkpointing=CONFIG.model.use_gradient_checkpointing,
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        mlflow.log_param("model_params", sum(p.numel() for p in model.parameters()))
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG.training.learning_rate,
            weight_decay=CONFIG.training.weight_decay,
            betas=(CONFIG.training.beta1, CONFIG.training.beta2),
            eps=CONFIG.training.eps
        )
        
        # LR scheduler with warmup + cosine decay
        total_steps = len(train_loader) * CONFIG.training.epochs
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=CONFIG.training.warmup_steps,
            total_steps=total_steps,
            min_lr=CONFIG.training.min_lr
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=CONFIG.training.early_stopping.patience,
            min_delta=CONFIG.training.early_stopping.min_delta
        )
        
        # Loss criterion
        criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.training.ignore_index)
        
        # Mixed precision scaler
        scaler = torch.amp.GradScaler('cuda') if CONFIG.model.use_mixed_precision else None
        
        # Gradient accumulation
        grad_accum_steps = CONFIG.training.gradient_accumulation_steps
        
        print(f"\nStarting v4 scratchpad training...")
        print(f"  Samples: {CONFIG.training.num_samples}, Epochs: {CONFIG.training.epochs}")
        print(f"  Batch size: {CONFIG.training.batch_size}, Grad accum: {grad_accum_steps}")
        print(f"  Effective batch: {CONFIG.training.batch_size * grad_accum_steps}")
        global_step = 0
        
        for epoch in range(CONFIG.training.epochs):
            model.train()
            total_train_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                
                if CONFIG.model.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        logits, _ = model(x)
                        loss = criterion(
                            logits.reshape(-1, tokenizer.vocab_size), 
                            y.reshape(-1)
                        )
                        loss = loss / grad_accum_steps
                    
                    scaler.scale(loss).backward()
                else:
                    logits, _ = model(x)
                    loss = criterion(
                        logits.reshape(-1, tokenizer.vocab_size),
                        y.reshape(-1)
                    )
                    loss = loss / grad_accum_steps
                    loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    if CONFIG.model.use_mixed_precision:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_train_loss += loss.item() * grad_accum_steps
                global_step += 1
                
                if batch_idx % 100 == 0:
                    mlflow.log_metric("train_loss", loss.item() * grad_accum_steps, step=global_step)
                    mlflow.log_metric("learning_rate", scheduler.get_lr(), step=global_step)
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    x, y = batch[:, :-1], batch[:, 1:]
                    
                    if CONFIG.model.use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            logits, _ = model(x)
                            loss = criterion(
                                logits.reshape(-1, tokenizer.vocab_size),
                                y.reshape(-1)
                            )
                    else:
                        logits, _ = model(x)
                        loss = criterion(
                            logits.reshape(-1, tokenizer.vocab_size),
                            y.reshape(-1)
                        )
                    
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            mlflow.log_metric("epoch_train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("epoch_val_loss", avg_val_loss, step=epoch)
            
            current_lr = scheduler.get_lr()
            print(f"Epoch {epoch+1}/{CONFIG.training.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            if early_stopping(avg_val_loss):
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                mlflow.log_param("early_stop_epoch", epoch + 1)
                break
        
        # Save final model
        os.makedirs(CONFIG.paths.checkpoint_dir, exist_ok=True)
        model_path = CONFIG.paths.pretrained_model
        torch.save(model.state_dict(), model_path)
        # NOTE: intentionally NOT calling mlflow.pytorch.log_model. Its formats
        # are version-fragile (pt2 needs an input_example; "cloudpickle" is
        # rejected by newer MLflow — only "pickle"/"pt2" are valid) and nothing
        # here loads via mlflow.pytorch — everything uses torch.load(state_dict).
        # The weights are still tracked in MLflow via log_artifact below.
        mlflow.log_artifact(model_path)
        
        print(f"\n✅ Training complete. Model saved to: {model_path}")
        print(f"   Final validation loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    train_supervised()