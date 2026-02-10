import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset import ArithmeticTokenizer, ArithmeticDataset
from model import MiniTransformer

# Config
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 25 # Increased for complexity

def train_supervised():
    mlflow.set_experiment("Arithmetic_LLM_Scaling")
    with mlflow.start_run(run_name="Supervised_Mult_Reverse"):
        tokenizer = ArithmeticTokenizer()
        train_ds = ArithmeticDataset(tokenizer, reverse_target=True)
        
        mlflow.log_params({
            "embed_dim": 256,
            "layers": 8,
            "reverse_mode": True,
            "operations": "add, sub, mult"
        })

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                                  collate_fn=lambda b: pad_sequence([i['input_ids'] for i in b], 
                                  batch_first=True, padding_value=tokenizer.pad_token_id))
        
        model = MiniTransformer(tokenizer).to("cuda")
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to("cuda")
                x, y = batch[:, :-1], batch[:, 1:]
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), "pretrained_arithmetic.pth")

if __name__ == "__main__":
    train_supervised()