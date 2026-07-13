# Arithmetic LLM: Supervised Pretraining vs RL Fine-Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive PyTorch implementation comparing supervised pretraining and reinforcement learning fine-tuning approaches for teaching transformer models to perform multi-digit arithmetic operations. This project demonstrates the power of combining both paradigms to achieve strong accuracy on complex arithmetic tasks including addition, subtraction, and multiplication with up to 5-digit operands.

**v4 Update:** The current architecture is a **~310M parameter decoder-only Transformer** trained with **scratchpad chain-of-thought** on **Modal A100-40GB GPUs**. It targets 5-digit arithmetic through a 7-phase curriculum. Local training is no longer recommended due to 8 GB VRAM constraints; use Modal cloud training instead.

---

## 📊 Results Summary (v4 — Modal H100, 5-digit scratchpad CoT)

Final full-run results (500 random test samples per model). Full log in [`docs/run_details.md`](docs/run_details.md).

| Metric | Pretrained Model | RL Fine-Tuned |
|--------|------------------|---------------|
| **Overall Accuracy (500 samples)** | **95.00%** (475/500) | **94.60%** (473/500) |
| 1–4 digit accuracy | **100%** | **100%** |
| 5-digit accuracy | **99.30%** | **97.90%** |
| Training | 20 epochs supervised | 15,000 RL episodes |

**Headline:** v4 fixes the catastrophic v3 collapse — from **4.00%** (v3) to **95.00%** overall, with **perfect 1–4 digit** and **~99% 5-digit** accuracy, decisively beating the v2 baseline (67.6%). This validates the v4 design: fixed train/inference mismatch (evaluate *with* scratchpad), 2× model capacity (~310M), and a full 5-digit curriculum.

**RL finding (deferred to v5):** On v4, RL fine-tuning did **not** improve over supervised pretraining (95.00% → 94.60%); multiplication regressed slightly (89.9% → 88.6%). With the supervised model already at ~95%, REINFORCE added policy-gradient noise without meaningful headroom. Investigating/redesigning RL is a v5 item.

> ⚠️ **Sampling variance:** the eval set is regenerated randomly each run (~±2% on 500 samples). Example: the pretrained model read 97.4% in one run and 95.0% in another with identical weights. Small deltas (like the pretrained-vs-RL gap) are within noise — a fixed/seeded eval set is a future improvement.

### Accuracy by digit complexity (final v4 run)

| Digits | Pretrained | RL Fine-Tuned |
|--------|-----------|---------------|
| 1-digit | 100% | 100% |
| 2-digit | 100% | 100% |
| 3-digit | 100% | 100% |
| 4-digit | 100% | 100% |
| 5-digit | 99.30% | 97.90% |
| 6-digit \* | 33.3% | 23.8% |
| 7–8 digit \* | 0–11% | 0–11% |

\* 6+ digit is **beyond the 5-digit training curriculum** — pure length extrapolation, not a target.

### Accuracy by operation (final v4 run)

| Operation | Pretrained | RL Fine-Tuned |
|-----------|-----------|---------------|
| Addition (+) | 95.7% | 95.8% |
| Subtraction (−) | 100% | 100% |
| Multiplication (×) | 89.9% | 88.6% |

**Key insight:** subtraction is solved (100%); multiplication is the hardest operation (longest scratchpads) and the main source of remaining error. Supervised pretraining alone essentially solves the in-distribution (≤5-digit) task.

### Sample predictions (final v4 run, with scratchpad → answer)

```
34938-6553=  → 28385   ✅   (5-digit subtraction)
7588+41794=  → 49382   ✅   (5-digit addition)
71*45=       → 3195    ✅   (multiplication)
2650-2622=   → 0028→28 ✅   (leading zeros handled)
5356*16=     → 85696   ✅
717*5391=    → 6       ❌   (large × large — truncated/hard)
```

> **Note on images:** the PNGs previously embedded here (`loss.png`, `episode_reward.png`, `Pretrained_accuracy.png`, etc.) are from an earlier (v2-era) run and do **not** reflect these v4 numbers. For live v4 metrics use MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (GPU recommended for faster training)
- 4GB+ GPU memory
- 2GB+ disk space for models and MLflow artifacts

### Installation

# 1. Clone the repository
git clone https://github.com/NANInithin/Arithmetic-LLM-supervised-pretraining-vs-RL-finetuning.git

cd Arithmetic-LLM-supervised-pretraining-vs-RL-finetuning

# 2. Create virtual environment (recommended)
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

### Run Complete Pipeline (Local — for testing only)

Execute the entire training and evaluation pipeline with a single command:

python run_pipeline.py

**Note:** The v4 model (~310M params) requires ~12-16 GB VRAM during training. Local runs on an RTX 4060 8 GB will OOM. For full v4 training, use Modal (see below).

**Pipeline Stages:**
1. **Cleanup:** Removes old local log files (`supervised_loss.npy`, `rl_rewards_replay.npy`)
2. **Metadata Logging:** Logs `configs/hyperparams.yaml` and `requirements.txt` to MLflow
3. **Supervised Pretraining:** 20 epochs on 300k samples
4. **RL Fine-Tuning:** 15,000 episodes with 7-phase 2→3→4→5 curriculum
5. **Evaluation:** Tests both models on 500 random 5-digit arithmetic problems with scratchpad decoding
6. **Results:** Displays accuracy metrics and MLflow tracking URL

Run individual stages:

python run_pipeline.py --stage supervised
python run_pipeline.py --stage rl
python run_pipeline.py --stage eval

### Run on Modal Cloud (Recommended for v4)

1. Install Modal and authenticate:
   ```bash
   pip install modal
   modal setup
   ```

2. Run the full pipeline on A100-40GB:
   ```bash
   modal run modal_train.py
   ```

3. Run a specific stage:
   ```bash
   modal run modal_train.py --stage supervised
   modal run modal_train.py --stage rl
   modal run modal_train.py --stage eval
   ```

Checkpoints are persisted to the `arithmetic-llm-checkpoints` Modal Volume and mounted at `/checkpoints`.

### View Results in MLflow

Start the MLflow UI to visualize training metrics (using the SQLite backend for metrics):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore:
- Loss curves and reward progression
- Model accuracy metrics
- Logged hyperparameters
- Model artifacts and weights

---

## 📂 Project Structure

Arithmetic-LLM-supervised-pretraining-vs-RL-finetuning/
│
├── README.md                      # This file
├── GETTING_STARTED.md             # Detailed setup guide
├── LICENSE                        # MIT License
│
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── dataset.py                 # ArithmeticDataset & ArithmeticTokenizer
│   ├── model.py                   # MiniTransformer (8-layer, 256-dim)
│   ├── train_supervised.py        # Supervised pretraining with reverse mode
│   ├── train_rl.py                # RL fine-tuning with curriculum & replay
│   ├── evaluate.py                # Final evaluation with accuracy metrics
│   └── plot.py                    # Visualization utilities
│
├── configs/                       # Configuration files
│   └── hyperparams.yaml           # Complete hyperparameter documentation
│
├── run_pipeline.py                # Master orchestration script (local + Modal)
├── modal_train.py                 # Modal cloud training wrapper
├── setup.py                       # Package installation configuration
├── requirements.txt               # Python dependencies
│
├── mlruns/                        # MLflow tracking (auto-created)
│   ├── 0/                         # Default experiment
│   └── artifacts/                 # Model checkpoints and logs
│
└── outputs/                       # Generated visualizations (auto-created)
    └── training_results.png

---

## 🏗️ Architecture

### MiniTransformer (Scaled)

A decoder-only Transformer optimized for arithmetic reasoning:

- **Embedding Dimension:** 256 (scaled from 192)
- **Attention Heads:** 8 (scaled from 6)
- **Layers:** 8 (scaled from 6)
- **Total Parameters:** ~2.8M (scaled from 1.5M)
- **Max Sequence Length:** 32 tokens (supports multiplication)
- **Vocabulary Size:** 16 tokens
  - Digits: `0-9`
  - Operations: `+`, `-`, `*`
  - Special: `=`, `<EOS>`, `<PAD>`

**Key Design Choice:** Reverse digit generation (right-to-left) aligns with manual arithmetic computation, reducing error propagation.

MiniTransformer(
    embed_dim=256,          # Embedding dimension
    num_heads=8,            # Multi-head attention
    num_layers=8,           # Transformer encoder layers
    max_len=32,             # Maximum sequence length
    dropout=0.1,            # Dropout rate
    vocab_size=16           # Token vocabulary
)

### Curriculum Learning Strategy

Five-phase progressive training with smooth transitions:

| Phase | Episodes | Difficulty | Operands Range | Smooth Transition |
|-------|----------|------------|----------------|-------------------|
| Easy | 1-1200 | 2-digit | 0-99 | - |
| Mix E→M | 1200-1700 | 2→3 digit | Gradual blend | Linear (500 ep) |
| Medium | 1700-3200 | 3-digit | 0-999 | - |
| Mix M→H | 3200-3700 | 3→4 digit | Gradual blend | Linear (500 ep) |
| Hard | 3700-7000 | 4-digit | 0-9999 | - |

**Innovation:** Smooth 500-episode transitions prevent "curriculum cliff" collapse that occurs with abrupt difficulty switches.

---

## 🔬 Training Methodology

### Phase 1: Supervised Pretraining

**Objective:** Establish strong foundation through teacher-forcing

**Configuration:**
- **Batch Size:** 512
- **Learning Rate:** 3e-4 (AdamW optimizer)
- **Epochs:** 25
- **Weight Decay:** 0.1 (regularization)
- **Loss Function:** Cross-entropy with padding ignore
- **Reverse Mode:** Targets generated right-to-left (e.g., `"123"` → `"321"`)

**Operations Supported:**
- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)

**Expected Outcome:** ~99% accuracy on 4-digit arithmetic

### Phase 2: RL Fine-Tuning

**Objective:** Policy refinement through reward optimization and curriculum scaling

**Configuration:**
- **Episodes:** 7000
- **Learning Rate:** 1e-5 (10x smaller than supervised)
- **Batch Size:** 128 (gradient accumulation)
- **Temperature:** 1.0 (balanced exploration)
- **Entropy Coefficient:** 0.01 (prevents mode collapse)
- **Gradient Clipping:** 1.0 max norm

### Dense Reward Shaping

Traditional sparse rewards (1.0 for correct, 0.0 for wrong) lead to slow learning. Our dense reward function:

def compute_reward(prediction, target):
    """
    Dense reward with partial credit for digit-by-digit matching
    """
    if exact_match(prediction, target):
        return 1.0  # Perfect answer
    
    # Invalid output penalty
    if not is_valid_number(prediction):
        return -0.1
    
    # Partial credit: Match digits right-to-left (units first)
    reward = 0.0
    pred_units = reverse(prediction)  # Units place first
    target_units = reverse(target)
    
    for i in range(min(len(pred_units), len(target_units))):
        if pred_units[i] == target_units[i]:
            reward += 0.15  # Per-digit bonus
        else:
            break  # Stop at first mismatch
    
    return min(reward, 0.9)  # Cap partial reward below perfect

**Why this works:**
- Provides learning signal even for incorrect answers
- Aligns with reverse generation strategy (units → tens → hundreds)
- Encourages incremental improvement

### Prioritized Experience Replay

**Problem:** Easy examples dominate training; model forgets edge cases

**Solution:** Maintain buffer of hard examples

**Configuration:**
- **Buffer Size:** 500 examples
- **Storage Threshold:** Reward < 0.99 (all non-perfect attempts)
- **Sampling Probability:** 25% per episode
- **Purpose:** Prevent catastrophic forgetting and focus on tail distribution

**Implementation:**
# Store failures
if reward < 0.99 and not is_replay:
    hard_buffer.append(example)
    if len(hard_buffer) > 500:
        hard_buffer.pop(0)  # FIFO

# Sample from buffer
if len(hard_buffer) > 50 and np.random.rand() < 0.25:
    example = random.choice(hard_buffer)  # Replay
else:
    example = sample_from_current_curriculum()  # Normal

---

## 📈 Results Visualization

### Training Metrics from MLflow

The following visualizations demonstrate the model's learning progression across both training phases:

#### Supervised Pretraining Loss

![Supervised Loss](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/aa2e2090-f92d-4ee1-a5e3-b181162a02e1)

*Supervised training loss decreases from ~1.7 to ~1.15 over 25 epochs, demonstrating stable convergence with reverse digit ordering. The smooth descent indicates effective learning of arithmetic patterns.*

#### RL Fine-Tuning Metrics

**Episode Reward:**

![Episode Reward](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/eaf01a36-9815-4c02-8cbf-f392969fb02a)

*Episode rewards show dense reward shaping in action, with frequent spikes to 1.0 (perfect answers) throughout training. The sustained high rewards across curriculum phases demonstrate effective policy learning.*

**Running Reward:**

![Running Reward](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/4b8523da-aaa7-4e7d-adf4-8e5dd356884d)

*Exponentially smoothed running reward (α=0.05) demonstrates steady improvement from ~0 to ~0.95-1.0, showing effective curriculum learning progression through all five phases without catastrophic forgetting.*

**Batch Loss:**

![Batch Loss](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/fdb1e124-0806-46a7-8e32-2208d48156eb)

*Policy gradient batch loss stabilizes around -0.05 to -0.15 after initial fluctuation, indicating consistent policy improvement through gradient accumulation. The negative values reflect positive advantages from baseline subtraction.*

#### Final Evaluation Accuracy

**Pretrained Model:**

![Pretrained Accuracy](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/060f228d-7fa6-4fe1-8239-e05c12dea06c)

*The supervised pretrained model achieves **99% accuracy** on the 4-digit arithmetic test set (500 examples), demonstrating that reverse digit generation and scaled architecture provide a strong foundation.*

**RL Fine-Tuned Model:**

![RL Fine-Tuned Accuracy](https://agi-prod-file-upload-public-main-use1.s3.amazonaws.com/3a6aa984-dedf-4330-a30f-0dfdd7873dd7)

*After RL fine-tuning with curriculum learning and prioritized replay, the model achieves **100% accuracy** on the same test set, eliminating all remaining edge case errors.*

### Performance Breakdown

| Operation | Pretrained | RL Fine-tuned |
|-----------|------------|---------------|
| Addition | ~99% | 100% |
| Subtraction | ~99% | 100% |
| Multiplication | ~99% | 100% |
| **Overall** | **99%** | **100%** |

**Key Insight:** The pretrained model already achieves near-perfect accuracy through scaled architecture and reverse digit ordering. RL fine-tuning eliminates the final 1% of edge cases through targeted replay and dense reward signals.

---

## 🔧 Configuration & Hyperparameters

All hyperparameters are documented in `configs/hyperparams.yaml`. Key settings:

### Supervised Pretraining

supervised_pretraining:
  # Data Configuration
  num_samples: 100000
  max_digits: 4
  operations: ['+', '-', '*']
  
  # Model Architecture
  embed_dim: 256
  num_heads: 8
  num_layers: 8
  max_len: 32
  
  # Training Configuration
  batch_size: 512
  learning_rate: 3e-4
  epochs: 25
  weight_decay: 0.1

### RL Fine-Tuning

rl_finetuning:
  # Episode Configuration
  total_episodes: 7000
  batch_size: 128
  
  # RL Hyperparameters
  learning_rate: 1e-5
  temperature: 1.0
  entropy_coef: 0.01
  
  # Reward Configuration
  reward:
    correct_answer: 1.0
    correct_digit_bonus: 0.15
    invalid_output: -0.1
  
  # Prioritized Replay Buffer
  replay_buffer:
    size: 500
    threshold: 0.99
    sampling_prob: 0.25

---

## 💡 Technical Highlights

### Why Reverse Digit Generation?

Generating answers **right-to-left** (units → tens → hundreds) improves learning because:

1. **Aligns with manual arithmetic:** Units place computed first in addition/subtraction
2. **Reduces error propagation:** Errors don't cascade from most significant digit
3. **Matches human computation:** Natural mental model of arithmetic
4. **Simplifies carries:** Carry information flows naturally in generation direction

**Example:**
Problem:  1234 + 5678 = ?
Standard: 6 9 1 2  (left-to-right, need all carries first)
Reverse:  2 1 9 6  (right-to-left, compute carries incrementally)

### Why Curriculum Learning?

Progressive difficulty scaling prevents:

- **Early collapse:** Model overwhelmed by hard examples before learning basics
- **Distribution mismatch:** Sudden difficulty jumps cause catastrophic forgetting
- **Sample inefficiency:** Wasting episodes on problems model can't yet solve

**Smooth transitions** (500-episode blending periods) are critical:
# Linear probability ramp prevents "curriculum cliff"
prob_hard = (episode - transition_start) / 500.0
dataset = ds_hard if np.random.rand() < prob_hard else ds_easy

### Why Prioritized Replay?

Storing and replaying hard examples:

- **Prevents forgetting:** Model maintains performance on easy tasks
- **Accelerates tail learning:** Focuses on remaining edge cases
- **Improves worst-case:** Eliminates systematic failures on specific patterns
- **Balances distribution:** Counteracts curriculum's easy-to-hard bias

**Buffer Strategy:**
- Store all non-perfect attempts (reward < 0.99)
- Sample 25% of episodes from buffer
- FIFO eviction when buffer full (500 examples)

---

## ⚙️ Usage Examples

### Example 1: Full Pipeline (Recommended)

python run_pipeline.py

Runs complete workflow:
1. Supervised pretraining → `pretrained_arithmetic.pth`
2. RL fine-tuning → `rl_arithmetic_replay.pth`
3. Evaluation → Accuracy metrics in MLflow
4. MLflow UI available at `http://127.0.0.1:5000`

### Example 2: Individual Training Stages

# Stage 1: Supervised pretraining (~15 minutes on GPU)
python src/train_supervised.py

# Stage 2: RL fine-tuning (~40 minutes on GPU)
python src/train_rl.py

# Stage 3: Evaluation (~1 minute)
python src/evaluate.py

### Example 3: Custom Inference

import torch
from src.model import MiniTransformer
from src.dataset import ArithmeticTokenizer

# Load tokenizer and model
tokenizer = ArithmeticTokenizer()
model = MiniTransformer(
    tokenizer, 
    embed_dim=256, 
    num_heads=8, 
    num_layers=8,
    max_len=32
).to("cuda")

# Load trained weights
model.load_state_dict(torch.load("rl_arithmetic_replay.pth"))
model.eval()

# Test custom problem
prompt = "1234+5678="
input_ids = tokenizer.encode(prompt)
output_ids = model.generate(input_ids, max_new_tokens=10)
result = tokenizer.decode(output_ids).split('=')[-1]

# Reverse back to normal order
final_result = result[::-1]
print(f"{prompt} {final_result}")  # Output: "1234+5678= 6912"

---

## 🛠️ Troubleshooting

### GPU Memory Issues

If encountering OOM errors, reduce batch sizes:

# In src/train_supervised.py
BATCH_SIZE = 256  # Default: 512

# In src/train_rl.py
BATCH_SIZE = 64   # Default: 128

### MLflow UI Not Starting

Check port availability:

# Check if port 5000 is in use
lsof -i :5000

# Use alternative port
mlflow ui --port 5001

### Model Checkpoint Not Found

Verify checkpoint files exist:

ls -lh *.pth
# Expected output:
# pretrained_arithmetic.pth (after supervised training)
# rl_arithmetic_replay.pth (after RL fine-tuning)

### Low RL Performance

If RL accuracy is below pretrained:

1. **Check learning rate:** Should be 10x smaller than supervised (1e-5)
2. **Verify baseline:** Ensure advantage calculation uses exponential smoothing
3. **Inspect curriculum:** Smooth transitions prevent catastrophic forgetting
4. **Monitor replay buffer:** Should contain 100+ examples after phase 1

---

## 📊 Performance Benchmarks

### Training Time (Actual Benchmark)

| Stage | Duration | GPU Memory | Details |
|-------|----------|------------|---------|
| Supervised Pretraining | ~29 min | ~2.5 GB | 25 epochs, batch_size=512 |
| RL Fine-Tuning | ~19 min | ~3.0 GB | 7000 episodes, batch_size=128 |
| Evaluation | <1 min | ~1.5 GB | 500 test samples per model |
| **Total Pipeline** | **~49 min** | **~3.0 GB** | From logs: 14:53:33 → 15:42:52 |

**Note:** Timings from actual run with PyTorch 2.7.1+cu118 on CUDA-capable GPU. CPU training will be significantly slower (~3-4x).

### Accuracy Progression

| Checkpoint | Overall Accuracy | Notes |
|------------|------------------|-------|
| Epoch 25 (Pretrained) | **98.60%** (493/500) | Strong baseline from supervised learning |
| RL Episode 7000 (Final) | **99.80%** (499/500) | Edge case refinement via curriculum + replay |

**Evaluation Details:**
- Test set: 500 random problems with 4-digit operands max
- Operations: Addition, subtraction, multiplication (including 2-digit × 2-digit)
- Reverse digit ordering maintained in predictions
- Both models tested on identical test set for fair comparison

---

## 📝 Limitations & Future Work

### Current Limitations

1. **Model Size:** Only 2.8M parameters; larger models (GPT-2 scale) would perform better
2. **Operations:** Addition, subtraction, multiplication only; division not supported
3. **Operand Range:** Capped at 4-digit numbers; longer operands require architecture changes
4. **No Scratchpad:** Direct answer generation without intermediate steps
5. **Single Operation:** Cannot handle chains like `5+3*2-1`

### Future Improvements

#### Near-Term Enhancements
- [ ] **Division Support:** Extend to division with quotient and remainder
- [ ] **Longer Operands:** Scale to 5-6 digit arithmetic with extended context window
- [ ] **Mixed Precision:** FP16 training for faster throughput
- [ ] **Distributed Training:** Multi-GPU support for larger batch sizes

#### Long-Term Research Directions
- [ ] **Chain-of-Thought:** Scratchpad for intermediate reasoning steps
- [ ] **Multi-Step Problems:** Complex expressions like `(123+456)*789`
- [ ] **Symbolic Reasoning:** Variables and algebraic expressions (`2x+5=15`)
- [ ] **Word Problems:** Natural language to arithmetic translation
- [ ] **Decimal Arithmetic:** Floating-point precision tasks
- [ ] **Negative Numbers:** Full support for signed operands
- [ ] **Model Distillation:** Transfer knowledge from GPT-3.5/GPT-4

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

### High-Priority Contributions
- [ ] Alternative reward shaping strategies (comparison study)
- [ ] PPO/A2C implementation for RL phase comparison
- [ ] Hyperparameter optimization (Optuna/Ray Tune integration)
- [ ] Unit tests for dataset and model components
- [ ] Benchmarking against larger models (GPT-2, GPT-3.5)

### How to Contribute

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/YourFeature`
3. **Commit changes:** `git commit -am 'Add YourFeature'`
4. **Push to branch:** `git push origin feature/YourFeature`
5. **Open Pull Request** with detailed description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Citation

**Author:** Nithin NAN  
**GitHub:** [@NANInithin](https://github.com/NANInithin)  
**Project:** [Arithmetic-LLM-supervised-pretraining-vs-RL-finetuning](https://github.com/NANInithin/Arithmetic-LLM-supervised-pretraining-vs-RL-finetuning)

### Citation

If you use this code in your research, please cite:

@software{arithmetic_llm_2026,
  title={Arithmetic LLM: Supervised Pretraining vs RL Fine-Tuning},
  author={NAN, Nithin},
  year={2026},
  publisher={GitHub},
  url={https://github.com/NANInithin/Arithmetic-LLM-supervised-pretraining-vs-RL-finetuning},
  note={Achieves 100\% accuracy on 4-digit arithmetic through curriculum learning and prioritized replay}
}

---

## 🙏 Acknowledgments

- **PyTorch Team:** Excellent deep learning framework
- **MLflow Team:** Comprehensive experiment tracking platform
- **Curriculum Learning:** Inspired by Bengio et al. (2009)
- **Arithmetic Reasoning Research:** Wei et al. (2024) - "Teaching Arithmetic to Small Transformers"
- **RL Community:** Policy gradient and reward shaping literature

---

## 📚 References

1. **Curriculum Learning:** Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum learning.* ICML.

2. **Transformers:** Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS.

3. **Policy Gradients:** Sutton, R. S., et al. (2000). *Policy gradient methods for reinforcement learning with function approximation.* NeurIPS.

4. **Arithmetic in Transformers:** Wei, J., et al. (2024). *Teaching arithmetic to small transformers.* ICLR.

5. **Reward Shaping:** Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML.

6. **Prioritized Experience Replay:** Schaul, T., et al. (2016). *Prioritized experience replay.* ICLR.

---

## 📋 Quick Reference Commands

# Installation
pip install -r requirements.txt
pip install -e .

# Full Pipeline
python run_pipeline.py

# Individual Stages
python src/train_supervised.py
python src/train_rl.py
python src/evaluate.py

# MLflow UI
mlflow ui
# Visit: http://127.0.0.1:5000

# Generate Plots
python src/plot.py

---

**⭐ If you find this project useful, please star it on GitHub!**

**🚀 Ready to train? Run `python run_pipeline.py` and achieve 100% accuracy!**