import torch
import random
from torch.utils.data import Dataset

class ArithmeticTokenizer:
    def __init__(self):
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '=', '<EOS>', '<PAD>']
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token_id = self.stoi['<PAD>']
        self.eos_token_id = self.stoi['<EOS>']

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids if i not in [self.pad_token_id, self.eos_token_id]])

# --- IMPROVED DATASET: Balanced Distribution ---
class ArithmeticDataset(Dataset):
    def __init__(self, tokenizer, num_samples=300000, max_digits=4, operations=['+', '-']):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.operations = operations

    def __len__(self):
        return self.num_samples

    def _get_random_number(self):
        # 1. First, pick how many digits we want (Uniformly!)
        # This gives equal chance to 1-digit (e.g., 7) and 4-digits (e.g., 9402)
        n_digits = random.randint(1, self.max_digits)
        
        # 2. Generate a number with that many digits
        if n_digits == 1:
            return random.randint(0, 9)
        else:
            # e.g., for 2 digits: 10 to 99
            return random.randint(10**(n_digits-1), (10**n_digits) - 1)

    def __getitem__(self, idx):
        # Generate balanced operands
        a = self._get_random_number()
        b = self._get_random_number()
        op = random.choice(self.operations)
        
        if op == '+': 
            res = a + b
        else:
            # Swap to ensure positive result
            if a < b: a, b = b, a 
            res = a - b
        
        prompt_str = f"{a}{op}{b}="
        full_str = f"{a}{op}{b}={res}"
        
        encoded_full = self.tokenizer.encode(full_str) + [self.tokenizer.eos_token_id]
        encoded_prompt = self.tokenizer.encode(prompt_str)
        
        return {
            "input_ids": torch.tensor(encoded_full, dtype=torch.long),
            "prompt_ids": torch.tensor(encoded_prompt, dtype=torch.long),
            "prompt_str": prompt_str,
            "target_str": full_str.split('=')[-1]
        }