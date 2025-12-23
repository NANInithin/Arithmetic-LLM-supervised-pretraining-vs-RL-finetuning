import torch
from dataset import ArithmeticTokenizer
from model import MiniTransformer

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path, num_samples=50):
    print(f"--- Evaluating: {model_path} ---")
    tokenizer = ArithmeticTokenizer()
    model = MiniTransformer(tokenizer, embed_dim=192, num_heads=6, num_layers=6).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0
    
    # Test on hard examples (random 4-digit math)
    from dataset import ArithmeticDataset
    ds = ArithmeticDataset(tokenizer, num_samples=num_samples, max_digits=4)
    
    print(f"{'Prompt':<10} | {'Prediction':<10} | {'Status'}")
    print("-" * 40)
    
    for i in range(len(ds)):
        item = ds[i]
        prompt = item['prompt_str']
        # Calculate real answer
        try:
            if '+' in prompt: val = sum(map(int, prompt[:-1].split('+')))
            else: val = int(prompt[:-1].split('-')[0]) - int(prompt[:-1].split('-')[1])
        except: continue
            
        # Generate
        input_ids = item['prompt_ids'].to(DEVICE).unsqueeze(0)
        gen_ids = model.generate(input_ids.tolist()[0], max_new_tokens=5)
        pred_str = tokenizer.decode(gen_ids).split('=')[-1] # Get part after =
        
        # Check
        is_correct = False
        try:
            if int(''.join(filter(str.isdigit, pred_str))) == val:
                is_correct = True
        except: pass
        
        if is_correct: correct += 1
        total += 1
        
        if i < 10: # Print first 10
            status = "✅" if is_correct else "❌"
            print(f"{prompt:<10} | {pred_str:<10} | {status}")

    print("-" * 40)
    print(f"Final Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")

if __name__ == "__main__":
    # You can compare both models!
    evaluate_model("pretrained_arithmetic.pth")
    evaluate_model("rl_arithmetic_replay.pth")