import torch
import mlflow
import os
from dataset import ArithmeticTokenizer, ArithmeticDataset
from model import MiniTransformer

# --- Config: Must match train_supervised and train_rl ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 8
MAX_LEN = 32

def evaluate_model(model_path, name, reverse_target=True):
    print(f"\n--- Evaluating: {name} ({model_path}) ---")
    tokenizer = ArithmeticTokenizer()
    
    model = MiniTransformer(
        tokenizer, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return

    ds = ArithmeticDataset(tokenizer, num_samples=500, max_digits=4, reverse_target=reverse_target)
    correct, total = 0, 0
    
    print(f"{'Prompt':<12} | {'Prediction':<10} | {'Status'}")
    print("-" * 40)
    
    for i in range(len(ds)):
        item = ds[i]
        prompt = item['prompt_str']
        
        # Calculate real answer
        clean_p = prompt.replace('=', '')
        if '+' in clean_p:
            val = int(clean_p.split('+')[0]) + int(clean_p.split('+')[1])
        elif '-' in clean_p:
            val = int(clean_p.split('-')[0]) - int(clean_p.split('-')[1])
        elif '*' in clean_p:
            val = int(clean_p.split('*')[0]) * int(clean_p.split('*')[1])
            
        input_ids = item['prompt_ids'].to(DEVICE).unsqueeze(0)
        gen_ids = model.generate(input_ids.tolist()[0], max_new_tokens=10)
        pred_raw = tokenizer.decode(gen_ids).split('=')[-1]
        
        pred_final = pred_raw[::-1] if reverse_target else pred_raw
        
        is_correct = False
        try:
            if int(''.join(filter(str.isdigit, pred_final))) == val:
                is_correct = True
        except: pass
        
        if is_correct: correct += 1
        total += 1
        
        if i < 50:
            status = "✅" if is_correct else "❌"
            print(f"{prompt:<12} | {pred_final:<10} | {status}")

    accuracy = correct / total
    print("-" * 40)
    print(f"Final Accuracy for {name}: {correct}/{total} ({accuracy*100:.2f}%)")
    
    # Expert Tip: Log the final accuracy as a metric to MLflow
    mlflow.log_metric(f"{name}_accuracy", accuracy) #

if __name__ == "__main__":
    # Start a single evaluation run in MLflow
    mlflow.set_experiment("Arithmetic_LLM_Scaling")
    with mlflow.start_run(run_name="Final_Evaluation"):
        evaluate_model("pretrained_arithmetic.pth", "Pretrained")
        evaluate_model("rl_arithmetic_replay.pth", "RL_FineTuned")
        
        # Log requirements for reproducibility
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt") #