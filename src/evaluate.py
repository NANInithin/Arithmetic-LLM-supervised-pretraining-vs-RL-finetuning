"""
Evaluation for Arithmetic LLM - v4 with Scratchpad CoT

Features:
- Config-driven hyperparameters from configs/hyperparams.yaml
- Evaluates both pretrained and RL-finetuned models
- Uses ArithTransformer (v4 architecture with vocab_size=21)
- Evaluates WITH scratchpad to match training (extracts answer after '|')
- Greedy decoding (temperature=0.0) for deterministic evaluation
"""

import os
import torch
import mlflow
import mlflow.pytorch

from src.config import get_config
from src.dataset import ArithmeticTokenizer, ArithmeticDataset
from src.model import ArithTransformer


def evaluate_model(model_path, name, max_digits=5, num_samples=None, print_first_n=None):
    """
    Evaluate a single model.
    
    Evaluation is done WITH scratchpad (matching training) and extracts the final
    answer after the last '|' delimiter. Greedy decoding is used for reproducible,
    deterministic results.
    
    Args:
        model_path: Path to model weights
        name: Display name for logging
        max_digits: Maximum digit complexity
        num_samples: Number of samples to evaluate (uses config if None)
        print_first_n: Number of samples to print (uses config if None)
    """
    CONFIG = get_config()
    
    # Sanitize name for MLflow (no parentheses, spaces, or hyphens)
    safe_name = name.replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_')
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ArithmeticTokenizer(vocab_size=21)
    
    # Use config values if not specified
    if num_samples is None:
        num_samples = CONFIG.evaluation.num_samples
    if print_first_n is None:
        print_first_n = CONFIG.evaluation.print_first_n
    
    # Initialize model (v4 architecture)
    model = ArithTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=CONFIG.model.embed_dim,
        num_heads=CONFIG.model.num_heads,
        num_layers=CONFIG.model.num_layers,
        dim_feedforward=CONFIG.model.dim_feedforward,
        max_len=CONFIG.model.max_len,
        dropout=0.0  # No dropout during evaluation
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Loaded model from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        return None
    
    # Create evaluation dataset WITH scratchpad (matches training distribution)
    ds = ArithmeticDataset(
        tokenizer, 
        num_samples=num_samples, 
        max_digits=max_digits, 
        reverse_target=False,  # Forward answer (not reverse)
        use_scratchpad=True    # Match training mode
    )
    
    # Counters
    correct = 0
    total = 0
    
    # Digit-level accuracy tracking
    digit_correct = {}
    digit_total = {}
    
    # Per-operation accuracy tracking
    op_correct = {}
    op_total = {}
    
    print(f"\nRunning evaluation on {num_samples} samples (up to {max_digits}-digit)...")
    print(f"{'Prompt':<15} | {'Prediction':<12} | {'Target':<12} | Status")
    print("-" * 60)
    
    for i in range(len(ds)):
        item = ds[i]
        prompt = item['prompt_str']
        
        # Calculate ground truth and identify operation
        clean_p = prompt.replace('=', '')
        if '+' in clean_p:
            parts = clean_p.split('+')
            val = int(parts[0]) + int(parts[1])
            op = '+'
            operand = parts[0]
        elif '-' in clean_p:
            parts = clean_p.split('-')
            val = int(parts[0]) - int(parts[1])
            op = '-'
            operand = parts[0]
        elif '*' in clean_p:
            parts = clean_p.split('*')
            val = int(parts[0]) * int(parts[1])
            op = '*'
            operand = parts[0]
        else:
            continue
        
        num_digits = max(len(operand), len(str(abs(val))))
        
        # Generate prediction with scratchpad (greedy, stop at EOS)
        input_ids = item['prompt_ids'].to(device).unsqueeze(0)
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids.tolist()[0], 
                max_new_tokens=CONFIG.rl.max_new_tokens,
                temperature=0.0,                       # Greedy decoding for deterministic eval
                eos_token_id=tokenizer.eos_token_id    # Stop at EOS
            )
        
        pred_raw = tokenizer.decode(gen_ids)
        
        # Extract final answer: everything after the last '|'
        if '|' in pred_raw:
            pred_ans = pred_raw.split('|')[-1].strip()
        else:
            # Fallback: extract after '=' if no scratchpad delimiter
            pred_ans = pred_raw.split('=')[-1].strip()
            
        pred_clean = "".join([c for c in pred_ans if c.isdigit() or c == '-'])
        
        # Check correctness
        is_correct = False
        try:
            if pred_clean and int(pred_clean) == val:
                is_correct = True
                correct += 1
        except ValueError:
            pass
        
        total += 1
        
        # Track digit-level accuracy
        digit_total[num_digits] = digit_total.get(num_digits, 0) + 1
        if is_correct:
            digit_correct[num_digits] = digit_correct.get(num_digits, 0) + 1
        
        # Track per-operation accuracy
        op_total[op] = op_total.get(op, 0) + 1
        if is_correct:
            op_correct[op] = op_correct.get(op, 0) + 1
        
        # Print first N examples
        if i < print_first_n:
            status = "✅" if is_correct else "❌"
            print(f"{prompt:<15} | {pred_clean:<12} | {str(val):<12} | {status}")
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    print("-" * 60)
    print(f"\n📊 Results for {name}:")
    print(f"   Overall Accuracy: {correct}/{total} ({accuracy*100:.2f}%)")
    
    # Print digit-level breakdown
    print(f"\n📈 Accuracy by digit complexity:")
    for digits in sorted(digit_total.keys()):
        if digit_total[digits] > 0:
            digit_acc = digit_correct.get(digits, 0) / digit_total[digits]
            print(f"   {digits}-digit: {digit_correct.get(digits, 0)}/{digit_total[digits]} ({digit_acc*100:.2f}%)")
    
    # Print per-operation breakdown
    print(f"\n🔢 Accuracy by operation:")
    for op in sorted(op_total.keys()):
        if op_total[op] > 0:
            op_acc = op_correct.get(op, 0) / op_total[op]
            print(f"   {op}: {op_correct.get(op, 0)}/{op_total[op]} ({op_acc*100:.2f}%)")
    
    # Log to MLflow
    mlflow.log_metric(f"{safe_name}_overall_accuracy", accuracy)
    for digits in sorted(digit_total.keys()):
        if digit_total[digits] > 0:
            digit_acc = digit_correct.get(digits, 0) / digit_total[digits]
            mlflow.log_metric(f"{safe_name}_{digits}digit_accuracy", digit_acc)
    # MLflow metric names may not contain '+' or '*', so map the operation
    # symbols to words before logging (dashes are allowed, but map for clarity).
    op_names = {'+': 'add', '-': 'sub', '*': 'mul'}
    for op in sorted(op_total.keys()):
        if op_total[op] > 0:
            op_acc = op_correct.get(op, 0) / op_total[op]
            mlflow.log_metric(f"{safe_name}_{op_names.get(op, op)}_accuracy", op_acc)
    
    return accuracy


def evaluate_all():
    """Evaluate all configured models."""
    CONFIG = get_config()
    
    mlflow.set_experiment("Arithmetic_LLM_Scaling_v4")
    with mlflow.start_run(run_name="Evaluation_v4"):
        # Log evaluation config
        mlflow.log_params({
            "eval/num_samples": CONFIG.evaluation.num_samples,
            "eval/max_digits": CONFIG.evaluation.max_digits,
            "eval/print_first_n": CONFIG.evaluation.print_first_n,
            "eval/use_scratchpad": CONFIG.evaluation.use_scratchpad,
            "model/embed_dim": CONFIG.model.embed_dim,
            "model/num_layers": CONFIG.model.num_layers,
            "model/vocab_size": CONFIG.model.vocab_size,
        })
        
        results = {}

        # On Modal, checkpoints live on the mounted volume (/checkpoints), but the
        # models_to_evaluate paths in the YAML are relative ("checkpoints/..."),
        # which don't exist inside the container. Redirect by basename when
        # ARITH_LLM_CHECKPOINT_DIR is set — mirrors the redirect in config.py so
        # eval actually finds the models instead of silently reporting nothing.
        modal_ckpt_dir = os.environ.get("ARITH_LLM_CHECKPOINT_DIR")

        # Evaluate each configured model
        for model_info in CONFIG.evaluation.models_to_evaluate:
            model_path = model_info.get("path", "")
            name = model_info.get("name", "Model")

            if modal_ckpt_dir:
                model_path = os.path.join(modal_ckpt_dir, os.path.basename(model_path))

            if os.path.exists(model_path):
                acc = evaluate_model(
                    model_path, 
                    name,
                    max_digits=CONFIG.evaluation.max_digits
                )
                results[name] = acc
            else:
                print(f"\n⚠️ Model not found: {model_path}")
                results[name] = None
        
        # Summary
        print("\n" + "="*60)
        print("📊 FINAL EVALUATION SUMMARY")
        print("="*60)
        for name, acc in results.items():
            if acc is not None:
                print(f"  {name}: {acc*100:.2f}%")
            else:
                print(f"  {name}: Not evaluated (file not found)")
        print("="*60)
        
        return results


if __name__ == "__main__":
    evaluate_all()