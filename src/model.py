import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniTransformer(nn.Module):
    def __init__(self, tokenizer, embed_dim=128, num_heads=8, num_layers=6, max_len=20, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        # We use TransformerEncoderLayer for GPT-style models (Self-Attn + FF)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, self.vocab_size)
        
        self.to(self.device)

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        # Causal Mask (prevent looking at future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
        
        # Pass through Transformer
        out = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(out)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            
        return logits, loss

    def generate(self, prompt_ids, max_new_tokens=5, temperature=1.0):
        self.eval()
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            with torch.no_grad():
                logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx_next.item() == self.tokenizer.eos_token_id:
                break
        return idx[0].tolist()