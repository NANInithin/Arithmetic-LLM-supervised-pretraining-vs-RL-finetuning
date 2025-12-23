# 🧮 Arithmo-Transformer: Supervised vs. RL Training

A "from-scratch" implementation of a Tiny Transformer trained to solve arithmetic problems (addition and subtraction of up to 4 digits). This project compares **Supervised Pretraining** against **Reinforcement Learning (RL)** fine-tuning strategies.

The model learns to output character-level solutions (e.g., input `12+3=` -> output `15`).

![Training Results](training_results.png)

## 🌟 Key Features
* **Architecture**: A custom GPT-style Transformer built in raw PyTorch (`MiniTransformer`).
* **Tokenization**: Character-level tokenizer specific to arithmetic (`0-9`, `+`, `-`, `=`).
* **Supervised Learning**: Standard Cross-Entropy loss training on generated synthetic data.
* **Reinforcement Learning**: Fine-tuning using a REINFORCE-style approach with:
    * **Curriculum Learning**: Progresses from 2-digit to 4-digit problems.
    * **Prioritized Replay Buffer**: Retrains on "hard" examples where the model previously failed.
    * **Reward Shaping**: Partial credit for correct magnitude (number of digits).

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `dataset.py` | Contains the `ArithmeticTokenizer` and `ArithmeticDataset` for generating infinite synthetic math problems. |
| `model.py` | The `MiniTransformer` architecture (Embedding + TransformerEncoder + Linear Head). |
| `train_supervised.py` | Script for initial pretraining using standard Cross Entropy Loss. |
| `train_rl.py` | Script for RL fine-tuning using Policy Gradients and a Replay Buffer. |
| `evaluate.py` | Tests the model's accuracy on unseen, difficult 4-digit problems. |
| `plot.py` | Visualizes the loss curves and reward history. |

## 🚀 Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
