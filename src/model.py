"""
ArithTransformer - v2 Architecture

A decoder-only Transformer with:
- RoPE (Rotary Position Embedding) for better length generalization
- Pre-LayerNorm with RMSNorm (more stable training)
- SwiGLU FFN activation (better gradient flow)
- Mixed precision and gradient checkpointing support

Parameter count: ~151M (embed=768, layers=16, heads=12, head_dim=64)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Unlike standard LayerNorm, RMSNorm removes the mean centering
    and only normalizes by RMS, making it more efficient and often
    more stable. No learnable bias.
    
    Formula: output = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS without centering
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute RoPE frequencies for all positions up to max_seq_len.
    
    RoPE encodes position information by rotating query and key vectors.
    The rotation angle depends on position and frequency.
    
    Args:
        head_dim: Dimension of each head (should be even for complex rotation)
        max_seq_len: Maximum sequence length to precompute for
        theta: Base frequency for the geometric progression
        
    Returns:
        cos_freq: Precomputed cosine frequencies [max_seq_len, head_dim/2]
        sin_freq: Precomputed sine frequencies [max_seq_len, head_dim/2]
    """
    # Ensure head_dim is even
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    
    # Compute frequencies using geometric progression
    # freq_i = theta^(-2i/d) where i ranges from 0 to d/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Create position indices [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len)
    
    # Compute angles: angle = position * frequency
    # Shape: [max_seq_len, head_dim/2]
    angles = positions[:, None] * freqs[None, :]
    
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos_freq: torch.Tensor, sin_freq: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE rotation to query/key tensors.
    
    For each position, we rotate the vector by the precomputed angle.
    This is done by decomposing into even/odd dimensions and applying
    a 2D rotation matrix.
    
    Args:
        x: Input tensor [batch, num_heads, seq_len, head_dim]
        cos_freq: Precomputed cosine [seq_len, head_dim/2]
        sin_freq: Precomputed sine [seq_len, head_dim/2]
        
    Returns:
        Rotated tensor with position encoding fused into
    """
    # x: [batch, heads, seq_len, head_dim]
    # cos/sin: [seq_len, head_dim/2] -> [1, 1, seq_len, head_dim/2]
    cos_freq = cos_freq.unsqueeze(0).unsqueeze(0)
    sin_freq = sin_freq.unsqueeze(0).unsqueeze(0)
    
    x_real = x[..., 0::2]  # Even dimensions
    x_imag = x[..., 1::2]  # Odd dimensions
    
    # Apply rotation: (a + ib) * (cos + i*sin) = a*cos - b*sin + i(a*sin + b*cos)
    out_real = x_real * cos_freq - x_imag * sin_freq
    out_imag = x_real * sin_freq + x_imag * cos_freq
    
    # Interleave back: [batch, heads, seq_len, head_dim]
    out = torch.zeros_like(x)
    out[..., 0::2] = out_real
    out[..., 1::2] = out_imag
    
    return out


class Attention(nn.Module):
    """
    Self-attention with RoPE (Rotary Position Embedding).
    
    Implements causal masked self-attention where position information
    is encoded via rotation rather than addition.
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim == num_heads * head_dim, "embed_dim must equal num_heads * head_dim"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_len = max_len
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Precompute RoPE frequencies (head_dim/2 pairs)
        cos_freq, sin_freq = precompute_rope_freqs(head_dim, max_len)
        self.register_buffer("cos_freq", cos_freq, persistent=False)
        self.register_buffer("sin_freq", sin_freq, persistent=False)
        
        # Causal mask (can be precomputed once)
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        # cos_freq and sin_freq are [max_len, head_dim/2], slice to [seq_len, head_dim/2]
        q = apply_rope(q, self.cos_freq[:seq_len], self.sin_freq[:seq_len])
        k = apply_rope(k, self.cos_freq[:seq_len], self.sin_freq[:seq_len])
        
        # Compute attention scores
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        # scores: [batch, heads, seq_len, seq_len]
        # mask: [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
        scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: [batch, heads, seq_len, seq_len]
        # v: [batch, heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.o_proj(attn_output)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU = SiLU(x) * Gate(x) where:
    - SiLU (Sigmoid Linear Unit) = x * sigmoid(x)
    - Gate = W_gate(x)
    
    This uses THREE linear layers (W_up, W_gate, W_down) instead of two.
    Intermediate dimension is typically 2.67x the model dimension.
    
    Formula: FFN(x) = down(W_down(SiLU(W_gate(x)) * W_up(x)))
    
    Reference: SwiGLU from "GLU Variants Improve Transformer" (Shazeer, 2020)
    Used in PaLM, LLaMA, Gemini.
    """
    def __init__(self, embed_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Up projection
        self.w_up = nn.Linear(embed_dim, intermediate_dim, bias=False)
        
        # Gate projection (for SwiGLU gating)
        self.w_gate = nn.Linear(embed_dim, intermediate_dim, bias=False)
        
        # Down projection
        self.w_down = nn.Linear(intermediate_dim, embed_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Up and gate
        up = self.w_up(x)
        gate = self.w_gate(x)
        
        # SwiGLU: SiLU(gate) * up
        # silu(x) = x * sigmoid(x)
        gated = F.silu(gate) * up
        
        # Dropout
        gated = self.dropout(gated)
        
        # Down projection
        return self.w_down(gated)


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block with Pre-LayerNorm.
    
    Architecture:
    1. RMSNorm(input)
    2. Self-Attention with RoPE
    3. Residual connection
    4. RMSNorm(output)
    5. SwiGLU FFN
    6. Residual connection
    
    This is "Pre-LN" style where normalization is applied BEFORE
    the sublayer rather than after.
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, 
                 dim_feedforward: int, max_len: int, dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Pre-LN for attention
        self.norm1 = RMSNorm(embed_dim)
        
        # Self-attention
        self.attention = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        # Pre-LN for FFN
        self.norm2 = RMSNorm(embed_dim)
        
        # SwiGLU FFN
        self.ffn = SwiGLUFFN(
            embed_dim=embed_dim,
            intermediate_dim=dim_feedforward,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention with residual
        residual = x
        x = self.norm1(x)
        
        if self.use_gradient_checkpointing:
            attn_out = checkpoint(self.attention, x, use_reentrant=False)
        else:
            attn_out = self.attention(x)
        
        x = residual + attn_out
        
        # Pre-LN FFN with residual
        residual = x
        x = self.norm2(x)
        
        if self.use_gradient_checkpointing:
            ffn_out = checkpoint(self.ffn, x, use_reentrant=False)
        else:
            ffn_out = self.ffn(x)
        
        return residual + ffn_out


class ArithTransformer(nn.Module):
    """
    Arithmetic Transformer for v2 Architecture.
    
    A decoder-only Transformer with:
    - Token embedding (no positional embedding - uses RoPE)
    - N decoder blocks with Pre-LN, RoPE attention, and SwiGLU
    - Final RMSNorm
    - LM head (tied to token embeddings)
    
    Parameter count: ~151M
    VRAM (bs=96): ~3.5 GB (with AMP + gradient checkpointing)
    """
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 768, 
        num_heads: int = 12, 
        num_layers: int = 16,
        dim_feedforward: int = 3072,
        max_len: int = 64, 
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
        tie_weights: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        
        # Token embedding (no position embedding - uses RoPE)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dim_feedforward=dim_feedforward,
                max_len=max_len,
                dropout=dropout,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm
        self.final_norm = RMSNorm(embed_dim)
        
        # LM head (optionally tied with token embeddings)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with appropriate scaling."""
        if isinstance(module, nn.Linear):
            # Linear layers: normal init with std = sqrt(2 / fan_in)
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, RMSNorm):
            # RMSNorm weight initialized to ones (default)
            pass
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input token IDs [batch, seq_len]
            targets: Target token IDs for loss computation [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided, else None)
        """
        # Token embedding
        x = self.token_embedding(x)
        
        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Final RMSNorm
        x = self.final_norm(x)
        
        # LM head
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt_ids: list, max_new_tokens: int = 10, temperature: float = 1.0, eos_token_id: int = None) -> list:
        """
        Autoregressive generation with sampling.
        
        Args:
            prompt_ids: List of input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            eos_token_id: Optional EOS token ID to stop generation
            
        Returns:
            List of generated token IDs (including prompt)
        """
        self.eval()
        device = next(self.parameters()).device
        
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            # Truncate to max_len if needed
            idx_cond = idx[:, -self.max_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get last token logits
            logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)
            
            # Stop if EOS token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return idx[0].tolist()


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(vocab_size: int = 16, config: dict = None) -> ArithTransformer:
    """
    Create an ArithTransformer model.
    
    Args:
        vocab_size: Size of vocabulary
        config: Optional config dict with model parameters
        
    Returns:
        ArithTransformer model
    """
    if config is None:
        config = {
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 16,
            'dim_feedforward': 3072,
            'max_len': 64,
            'dropout': 0.1,
            'use_gradient_checkpointing': True
        }
    
    return ArithTransformer(
        vocab_size=vocab_size,
        embed_dim=config.get('embed_dim', 768),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 16),
        dim_feedforward=config.get('dim_feedforward', 3072),
        max_len=config.get('max_len', 64),
        dropout=config.get('dropout', 0.1),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', True)
    )


# Keep MiniTransformer for backward compatibility (v1 models)
class MiniTransformer(nn.Module):
    """
    Legacy v1 model for backward compatibility.
    DO NOT use for new projects - use ArithTransformer instead.
    """
    def __init__(self, tokenizer, embed_dim=256, num_heads=8, num_layers=8, max_len=32, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
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
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
        out = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(out)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    def generate(self, prompt_ids, max_new_tokens=10, temperature=1.0):
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


if __name__ == "__main__":
    # Quick test of the model
    print("Testing ArithTransformer v2...")
    
    model = ArithTransformer(
        vocab_size=16,
        embed_dim=768,
        num_heads=12,
        num_layers=16,
        dim_feedforward=3072,
        max_len=64
    )
    
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    print(f"Expected ~151M, got {total_params/1e6:.2f}M")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    x = torch.randint(0, 16, (batch_size, seq_len))
    targets = torch.randint(0, 16, (batch_size, seq_len))
    
    logits, loss = model(x, targets)
    print(f"Output shape: {logits.shape}")  # [batch, seq_len, vocab_size]
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    prompt = [1, 2, 3, 4, 5]  # Simple prompt
    generated = model.generate(prompt, max_new_tokens=5, eos_token_id=14)
    print(f"Generated tokens: {generated[:10]}")
    
    print("\n✅ Model test passed!")