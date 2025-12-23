import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our classes
from dataset import ArithmeticTokenizer, ArithmeticDataset
from model import MiniTransformer

# --- Configuration ---
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer globally for collate_fn
tokenizer = ArithmeticTokenizer()

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    return pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

def train_supervised():
    print(f"Training on {DEVICE}...")
    
    # 1. GENERATE MORE DATA (100k samples covers all 4-digit combos)
    #    max_digits=4 is the sweet spot to start.
    train_ds = ArithmeticDataset(tokenizer, num_samples=100000, max_digits=4)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True)
    
    # 2. BALANCED MODEL (4 layers, 4 heads is robust for this size)
    model = MiniTransformer(tokenizer, embed_dim=192, num_heads=6, num_layers=6).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 3. LOSS FUNCTION WITH IGNORE_INDEX (Crucial!)
    #    This prevents the model from "cheating" by predicting padding.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    loss_history = []  # To store loss per epoch for plotting later
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_inputs in train_loader:
            batch_inputs = batch_inputs.to(DEVICE)
            
            x = batch_inputs[:, :-1]
            y = batch_inputs[:, 1:]
            
            optimizer.zero_grad()
            
            # Note: We compute loss manually here to use 'criterion'
            logits, _ = model(x) 
            
            # Reshape for loss: (Batch * Seq, Vocab) vs (Batch * Seq)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
# Test every epoch
        if (epoch+1) % 1 == 0:
            print("-" * 30)
            # We test a mix of difficulties: 1-digit, 2-digit, 3-digit, 4-digit
            test_prompts = [
                "5+3=",          # Easy
                "12+7=",         # Medium
                "123+45=",       # Hard
                "1050+2050=",    # Very Hard
                "9999-1=",        # Subtraction Edge Case
                "250-125="
            ]
            
            for p in test_prompts:
                # Encode and move to device
                inp = torch.tensor(tokenizer.encode(p), dtype=torch.long).to(DEVICE)
                
                # Generate (max_new_tokens=6 to handle 5-digit answers like 19998)
                gen_ids = model.generate(inp.tolist(), max_new_tokens=6)
                
                # Decode
                print(f"  Test: {p:<12} -> {tokenizer.decode(gen_ids)}")
            print("-" * 30)
            
            # If we hit a very low loss, we can stop early
            if avg_loss < 0.05:
                print("Converged! Stopping early.")
                break

    torch.save(model.state_dict(), "pretrained_arithmetic.pth")
    print("Model saved.")
    np.save("supervised_loss.npy", np.array(loss_history))  # Save loss history for plotting
    print("Training loss history saved.")

if __name__ == "__main__":
    train_supervised()