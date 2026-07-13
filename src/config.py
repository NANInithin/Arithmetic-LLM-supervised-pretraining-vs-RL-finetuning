"""
Configuration loader for v3 architecture hyperparams.

v3 includes scratchpad chain-of-thought for multi-digit arithmetic.
Performance corrected: removed pad_to_length, reduced samples/epochs.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration (v4: ~310M params)."""
    embed_dim: int = 896
    num_heads: int = 14
    head_dim: int = 64
    num_layers: int = 24
    dim_feedforward: int = 3584
    max_len: int = 256           # Room for 5-digit scratchpad sequences
    vocab_size: int = 21          # Scratchpad tokens: |, [, ], C, B
    positional_encoding: str = "rope"
    normalization: str = "rmsnorm"
    ffn_activation: str = "swiglu"
    dropout: float = 0.1
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 5
    min_delta: float = 1e-4


@dataclass
class TrainingConfig:
    """Supervised training configuration (v4)."""
    num_samples: int = 300000       # 2× data for 2× model
    val_split: float = 0.05
    max_digits_supervised: int = 5  # Full 5-digit curriculum
    use_scratchpad: bool = True      # Scratchpad CoT mode
    batch_size: int = 128            # A100 headroom enables larger batches
    max_len: int = 256
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    lr_scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 800          # Proportional to larger dataset
    min_lr: float = 1e-6             # Lower final LR for better convergence
    epochs: int = 20                 # More data + bigger model needs more passes
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    criterion: str = "CrossEntropyLoss"
    ignore_index: int = 15          # PAD token ID (unchanged)
    gradient_accumulation_steps: int = 4  # Effective batch = 128*4 = 512


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration for RL."""
    size: int = 2000                 # Increased
    threshold: float = 0.99
    sampling_prob: float = 0.25


@dataclass
class RewardConfig:
    """Reward configuration for RL."""
    correct_answer: float = 1.0
    correct_digit_count: float = 0.1
    invalid_output: float = -0.1
    partial_credit_per_digit: float = 0.15
    scratchpad_step_bonus: float = 0.2  # NEW bonus


@dataclass
class CurriculumPhase:
    """Single curriculum phase."""
    max_digits: Optional[int] = None
    start_episode: int = 0
    end_episode: int = 0
    from_max_digits: Optional[int] = None
    to_max_digits: Optional[int] = None


@dataclass
class RLConfig:
    """RL fine-tuning configuration (v4)."""
    total_episodes: int = 15000      # More episodes for 5-digit curriculum
    batch_size: int = 128
    use_model_config: bool = True
    learning_rate: float = 5e-6
    temperature: float = 1.0
    temperature_end: float = 0.5       # Lower final temp for exploitation
    max_new_tokens: int = 60         # 5-digit scratchpad needs more tokens
    entropy_coef: float = 0.01
    entropy_coef_end: float = 0.003    # Stronger entropy decay
    max_grad_norm: float = 1.0
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    curriculum: Dict[str, CurriculumPhase] = field(default_factory=dict)
    dataset_sizes: Dict[str, int] = field(default_factory=lambda: {
        "easy": 15000,
        "medium": 15000,
        "hard": 30000,
        "very_hard": 30000
    })
    reward: RewardConfig = field(default_factory=RewardConfig)
    baseline_smoothing: float = 0.05


@dataclass
class EvaluationConfig:
    """Evaluation configuration (v4)."""
    num_samples: int = 500
    max_digits: int = 5
    print_first_n: int = 50
    use_scratchpad: bool = True      # Match training mode
    models_to_evaluate: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PathsConfig:
    """Paths configuration (v4)."""
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    plots_dir: str = "plots"
    logs_dir: str = "logs"
    mlruns_dir: str = "mlruns"
    pretrained_model: str = "checkpoints/pretrained_v4.pth"
    rl_model: str = "checkpoints/rl_finetuned_v4.pth"


@dataclass
class Config:
    """Full configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _dict_to_curriculum_phase(d: Dict[str, Any]) -> CurriculumPhase:
    """Convert dict to CurriculumPhase."""
    return CurriculumPhase(
        max_digits=d.get("max_digits"),
        start_episode=d.get("start_episode", 0),
        end_episode=d.get("end_episode", 0),
        from_max_digits=d.get("from_max_digits"),
        to_max_digits=d.get("to_max_digits")
    )


def load_config(config_path: str = "configs/hyperparams.yaml") -> Config:
    """
    Load configuration from YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Parse model config
    model_cfg = raw.get('model', {})
    model = ModelConfig(
        embed_dim=int(model_cfg.get('embed_dim', 768)),
        num_heads=int(model_cfg.get('num_heads', 12)),
        head_dim=int(model_cfg.get('head_dim', 64)),
        num_layers=int(model_cfg.get('num_layers', 16)),
        dim_feedforward=int(model_cfg.get('dim_feedforward', 3072)),
        max_len=int(model_cfg.get('max_len', 128)),
        vocab_size=int(model_cfg.get('vocab_size', 21)),
        positional_encoding=model_cfg.get('positional_encoding', 'rope'),
        normalization=model_cfg.get('normalization', 'rmsnorm'),
        ffn_activation=model_cfg.get('ffn_activation', 'swiglu'),
        dropout=float(model_cfg.get('dropout', 0.1)),
        use_mixed_precision=bool(model_cfg.get('use_mixed_precision', True)),
        use_gradient_checkpointing=bool(model_cfg.get('use_gradient_checkpointing', True))
    )
    
    # Parse training config
    train_cfg = raw.get('training', {})
    early_stop_cfg = train_cfg.get('early_stopping', {})
    early_stopping = EarlyStoppingConfig(
        enabled=bool(early_stop_cfg.get('enabled', True)),
        patience=int(early_stop_cfg.get('patience', 5)),
        min_delta=float(early_stop_cfg.get('min_delta', 1e-4))
    )
    training = TrainingConfig(
        num_samples=int(train_cfg.get('num_samples', 300000)),
        val_split=float(train_cfg.get('val_split', 0.05)),
        max_digits_supervised=int(train_cfg.get('max_digits_supervised', 5)),
        use_scratchpad=bool(train_cfg.get('use_scratchpad', True)),
        batch_size=int(train_cfg.get('batch_size', 128)),
        max_len=int(train_cfg.get('max_len', 256)),
        optimizer=train_cfg.get('optimizer', 'AdamW'),
        learning_rate=float(train_cfg.get('learning_rate', 3e-4)),
        weight_decay=float(train_cfg.get('weight_decay', 0.1)),
        beta1=float(train_cfg.get('beta1', 0.9)),
        beta2=float(train_cfg.get('beta2', 0.95)),
        eps=float(train_cfg.get('eps', 1e-8)),
        lr_scheduler=train_cfg.get('lr_scheduler', 'cosine_with_warmup'),
        warmup_steps=int(train_cfg.get('warmup_steps', 800)),
        min_lr=float(train_cfg.get('min_lr', 1e-6)),
        epochs=int(train_cfg.get('epochs', 20)),
        early_stopping=early_stopping,
        criterion=train_cfg.get('criterion', 'CrossEntropyLoss'),
        ignore_index=int(train_cfg.get('ignore_index', 15)),
        gradient_accumulation_steps=int(train_cfg.get('gradient_accumulation_steps', 4))
    )
    
    # Parse RL config
    rl_cfg = raw.get('rl', {})
    replay_cfg = rl_cfg.get('replay_buffer', {})
    replay_buffer = ReplayBufferConfig(
        size=int(replay_cfg.get('size', 2000)),
        threshold=float(replay_cfg.get('threshold', 0.99)),
        sampling_prob=float(replay_cfg.get('sampling_prob', 0.25))
    )
    reward_cfg = rl_cfg.get('reward', {})
    reward = RewardConfig(
        correct_answer=float(reward_cfg.get('correct_answer', 1.0)),
        correct_digit_count=float(reward_cfg.get('correct_digit_count', 0.1)),
        invalid_output=float(reward_cfg.get('invalid_output', -0.1)),
        partial_credit_per_digit=float(reward_cfg.get('partial_credit_per_digit', 0.15)),
        scratchpad_step_bonus=float(reward_cfg.get('scratchpad_step_bonus', 0.2))
    )
    curriculum_dict = rl_cfg.get('curriculum', {})
    curriculum = {
        k: _dict_to_curriculum_phase(v) 
        for k, v in curriculum_dict.items()
    }
    rl = RLConfig(
        total_episodes=int(rl_cfg.get('total_episodes', 15000)),
        batch_size=int(rl_cfg.get('batch_size', 128)),
        use_model_config=bool(rl_cfg.get('use_model_config', True)),
        learning_rate=float(rl_cfg.get('learning_rate', 5e-6)),
        temperature=float(rl_cfg.get('temperature', 1.0)),
        temperature_end=float(rl_cfg.get('temperature_end', 0.5)),
        max_new_tokens=int(rl_cfg.get('max_new_tokens', 60)),
        entropy_coef=float(rl_cfg.get('entropy_coef', 0.01)),
        entropy_coef_end=float(rl_cfg.get('entropy_coef_end', 0.003)),
        max_grad_norm=float(rl_cfg.get('max_grad_norm', 1.0)),
        replay_buffer=replay_buffer,
        curriculum=curriculum,
        dataset_sizes={k: int(v) for k, v in rl_cfg.get('dataset_sizes', {"easy": 15000, "medium": 15000, "hard": 30000, "very_hard": 30000}).items()},
        reward=reward,
        baseline_smoothing=float(rl_cfg.get('baseline_smoothing', 0.05))
    )
    
    # Parse evaluation config
    eval_cfg = raw.get('evaluation', {})
    evaluation = EvaluationConfig(
        num_samples=eval_cfg.get('num_samples', 500),
        max_digits=eval_cfg.get('max_digits', 5),
        print_first_n=eval_cfg.get('print_first_n', 50),
        use_scratchpad=eval_cfg.get('use_scratchpad', True),
        models_to_evaluate=eval_cfg.get('models_to_evaluate', [])
    )
    
    # Parse paths config
    paths_cfg = raw.get('paths', {})
    checkpoint_dir = paths_cfg.get('checkpoint_dir', 'checkpoints')
    
    # Allow Modal (or other cloud runners) to redirect checkpoints to a mounted volume.
    # When redirected, model checkpoints are placed inside that volume.
    modal_checkpoint_dir = os.environ.get('ARITH_LLM_CHECKPOINT_DIR')
    if modal_checkpoint_dir:
        pretrained_model = os.path.join(modal_checkpoint_dir, 'pretrained_v4.pth')
        rl_model = os.path.join(modal_checkpoint_dir, 'rl_finetuned_v4.pth')
        checkpoint_dir = modal_checkpoint_dir
    else:
        pretrained_model = paths_cfg.get('pretrained_model', 'checkpoints/pretrained_v4.pth')
        rl_model = paths_cfg.get('rl_model', 'checkpoints/rl_finetuned_v4.pth')
    
    paths = PathsConfig(
        checkpoint_dir=checkpoint_dir,
        results_dir=paths_cfg.get('results_dir', 'results'),
        plots_dir=paths_cfg.get('plots_dir', 'plots'),
        logs_dir=paths_cfg.get('logs_dir', 'logs'),
        mlruns_dir=paths_cfg.get('mlruns_dir', 'mlruns'),
        pretrained_model=pretrained_model,
        rl_model=rl_model
    )
    
    return Config(model=model, training=training, rl=rl, evaluation=evaluation, paths=paths)


def get_config() -> Config:
    """Load config from default path."""
    return load_config()


if __name__ == "__main__":
    config = load_config()
    print("Loaded v4 config:")
    print(f"  Model: embed_dim={config.model.embed_dim}, vocab_size={config.model.vocab_size}")
    print(f"  Training: samples={config.training.num_samples}, epochs={config.training.epochs}")
    print(f"  RL: episodes={config.rl.total_episodes}, max_new_tokens={config.rl.max_new_tokens}")