# Getting Started Guide

Welcome! This guide will walk you through running the RL Arithmetic Fine-Tuning project step-by-step.

## ⚡ 5-Minute Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/rl-arithmetic-finetuning.git
cd rl-arithmetic-finetuning
pip install -r requirements.txt
```

### 2. Download Pre-trained Model (Optional)
```bash
# Download pretrained_arithmetic.pth from GitHub Releases
# Place in root directory (optional, model will be trained if missing)
```

### 3. Run Everything
```bash
# Pretraining (30 min)
python src/train_supervised.py

# RL Fine-tuning (2-3 hours)
python src/train_rl.py

# Evaluate
python src/evaluate.py

# Visualize
python src/plot.py
```

---

## 🛠️ Step-by-Step Setup

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**✅ Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Step 2: Run Supervised Pretraining
```bash
python src/train_supervised.py
```

**What happens:**
- Generates 100,000 random arithmetic examples (2-4 digit)
- Trains for 20 epochs with cosine annealing
- Saves model to `pretrained_arithmetic.pth`
- Saves loss history to `supervised_loss.npy`

**Expected output:**
```
Training on cuda...
Epoch 1/20 | Loss: 1.2340 | LR: 0.000300
Epoch 2/20 | Loss: 0.5821 | LR: 0.000298
...
Epoch 20/20 | Loss: 0.1217 | LR: 0.000001
Model saved.
Training loss history saved.
```

**Duration:** ~30 minutes on RTX 4060 / ~60 min on CPU

---

### Step 3: Run RL Fine-Tuning
```bash
python src/train_rl.py
```

**What happens:**
- Loads pretrained model from Step 2
- Runs 7000 RL training episodes with curriculum
- Saves RL model to `rl_arithmetic_replay.pth`
- Saves reward history to `rl_rewards_replay.npy`

**Expected output:**
```
--- Starting RL with Prioritized Replay on cuda ---
✅ Loaded model weights.
100    | Easy     | Rw: 0.5551 | 79+9= 89
200    | Easy     | Rw: 0.4610 | 58-8= 41
...
7000   | Hard     | Rw: 0.3150 | 9886-6805= 2571

RL Training Complete.
```

**Duration:** ~2-3 hours on RTX 4060 / ~6-8 hours on CPU

---

### Step 4: Evaluate Models
```bash
python src/evaluate.py
```

**What happens:**
- Tests both models (pretrained and RL) on 50 random 4-digit problems
- Prints accuracy metrics
- Shows example predictions

**Expected output:**
```
--- Evaluating: pretrained_arithmetic.pth ---
Prompt     | Prediction | Status
----------------------------------------
3+3=       | 6          | ✅
...
Final Accuracy: 48/50 (96.00%)

--- Evaluating: rl_arithmetic_replay.pth ---
Prompt     | Prediction | Status
----------------------------------------
532-4=     | 528        | ✅
...
Final Accuracy: 47/50 (94.00%)
```

---

### Step 5: Visualize Training
```bash
python src/plot.py
```

**What happens:**
- Generates `training_results.png` with two plots:
  1. **Left:** Supervised loss curve (should decline)
  2. **Right:** RL reward trajectory (noisy but averaging ~0.3-0.5)

---

## 📊 Understanding the Results

### Accuracy Breakdown
- **2-digit:** ~99% (easy, model learned well)
- **3-digit:** ~90% (medium, some carry errors)
- **4-digit:** ~48% (hard, requires look-ahead reasoning)

### Why 4-digit is Hard
When computing `1234 + 5678`:
- Model must know: `4+8=12` (carry 1) → `3+7+1=11` (carry 1) → ...
- Without seeing future additions, this is a **look-ahead dependency**
- Small models can't solve this without Chain-of-Thought (intermediate steps)

### RL Fine-tuning Impact
- **Best case:** +2-3% on 4-digit (from 46% → 48%)
- **Trade-off:** Slight drop on 3-digit (92% → 90%)
- **Reason:** RL forces model to focus on hard examples, sacrificing easy ones

---

## 🔧 Customization

### Modify Training Duration
Edit `src/train_rl.py`:
```python
EPISODES = 7000  # Change to 3500 for 1.5 hours or 14000 for 6 hours
```

### Faster Pretraining (Toy Experiment)
Edit `src/train_supervised.py`:
```python
train_ds = ArithmeticDataset(tokenizer, num_samples=10000, max_digits=4)  # Was 100000
EPOCHS = 5  # Was 20
```

### Skip RL (Just Evaluate Pretrained)
```bash
# Simply don't run train_rl.py
python src/evaluate.py
# Will evaluate only the pretrained model
```

### Use Different Hyperparameters
Edit `src/train_rl.py`:
```python
LR = 5e-5           # Was 1e-5 (higher = faster learning, riskier)
TEMPERATURE = 0.5   # Was 1.0 (lower = more focused)
REPLAY_PROB = 0.5   # Was 0.25 (higher = more replay focus)
```

---

## 🐛 Troubleshooting

### "CUDA out of memory" Error
```python
# In train_supervised.py:
BATCH_SIZE = 256  # Was 512 (reduces GPU memory)

# In train_rl.py:
BATCH_SIZE = 64   # Was 128
```

### Model training is too slow
```bash
# CPU vs GPU check
python -c "import torch; print(torch.cuda.is_available())"
# If False, install CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "No such file: pretrained_arithmetic.pth"
This is normal! RL training will fail gracefully if pretraining wasn't done first.
Run `train_supervised.py` first.

### Model accuracy drops suddenly during RL
This is expected! RL training is noisier than supervised learning. The moving average (smoothed reward) should stabilize.

---

## 📈 Monitoring Progress

### During Supervised Training
Watch the loss curve decline:
```
Epoch 1: Loss 1.234
Epoch 5: Loss 0.582
Epoch 10: Loss 0.321
Epoch 15: Loss 0.198
Epoch 20: Loss 0.121
```

Goal: Loss < 0.15 by end

### During RL Training
Watch the moving average reward:
```
Episode 100: Reward 0.55
Episode 1000: Reward 0.48
Episode 3500: Reward 0.35
Episode 7000: Reward 0.31
```

This decline is normal! Reward decreases as curriculum gets harder.

---

## 🚀 Next Steps

### Try Advanced Features
1. **Modify Curriculum:** Change phase transition points in `select_dataset()`
2. **Change Reward Function:** Add reward for "almost correct" answers
3. **Implement Chain-of-Thought:** Modify dataset to include reasoning steps
4. **Test on Multiplication:** Extend tokenizer and dataset

### Run Experiments
```bash
# Experiment 1: No RL (just evaluate pretrained)
python src/evaluate.py

# Experiment 2: Faster RL (fewer episodes)
# Edit EPISODES = 1000 in train_rl.py
python src/train_rl.py

# Experiment 3: Larger model
# Edit embed_dim=256, num_layers=8 in train_supervised.py
python src/train_supervised.py
```

---

## 📚 Learn More

- **README.md:** Full project documentation
- **configs/hyperparams.yaml:** All configurable parameters
- **CONTRIBUTING.md:** How to contribute improvements
- **src/*.py:** Detailed code comments

---

## ❓ FAQ

**Q: How much GPU memory do I need?**  
A: ~2GB minimum (RTX 3050 / A100 with smaller batch size)

**Q: Can I run on CPU?**  
A: Yes, but expect ~10x slower training (CPU takes 5-30 hours)

**Q: Can I use a pretrained model from somewhere?**  
A: Yes! The code automatically loads `pretrained_arithmetic.pth` if it exists

**Q: What if I want to train a larger model?**  
A: Increase `embed_dim` (192→256) and `num_layers` (6→8) in both training scripts

**Q: How do I use the model for inference?**  
A: See the README under "Usage Examples"

---

## 🎉 Success Criteria

✅ You've completed the setup when:
- [ ] Supervised pretraining runs without errors
- [ ] RL fine-tuning runs and saves models
- [ ] Evaluation shows ~94-96% accuracy
- [ ] `training_results.png` is generated
- [ ] Loss and reward curves look reasonable

---

**Next: Read README.md for detailed documentation!**
