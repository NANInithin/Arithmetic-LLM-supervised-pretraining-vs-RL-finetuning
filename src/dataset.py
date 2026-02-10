import torch
import random
from torch.utils.data import Dataset

class ArithmeticTokenizer:
    def __init__(self):
        # Added '*' to the vocabulary
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '=', '<EOS>', '<PAD>']
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token_id = self.stoi['<PAD>']
        self.eos_token_id = self.stoi['<EOS>']

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids if i not in [self.pad_token_id, self.eos_token_id]])

class ArithmeticDataset(Dataset):
    def __init__(self, tokenizer, num_samples=300000, max_digits=4, operations=['+', '-', '*'], reverse_target=True):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.operations = operations
        self.reverse_target = reverse_target

    def __len__(self):
        return self.num_samples

    def _get_random_number(self, max_digits=None):
        n_digits = random.randint(1, max_digits or self.max_digits)
        if n_digits == 1:
            return random.randint(0, 9)
        return random.randint(10**(n_digits-1), (10**n_digits) - 1)

    def __getitem__(self, idx):
        op = random.choice(self.operations)
        # Multiplication is harder, so we cap operands at 2 digits initially
        if op == '*':
            a = self._get_random_number(max_digits=2)
            b = self._get_random_number(max_digits=2)
            res = a * b
        else:
            a = self._get_random_number()
            b = self._get_random_number()
            if op == '+': 
                res = a + b
            else:
                if a < b: a, b = b, a 
                res = a - b
        
        res_str = str(res)
        if self.reverse_target:
            res_str = res_str[::-1]
        
        prompt_str = f"{a}{op}{b}="
        full_str = f"{prompt_str}{res_str}"
        
        return {
            "input_ids": torch.tensor(self.tokenizer.encode(full_str) + [self.tokenizer.eos_token_id], dtype=torch.long),
            "prompt_ids": torch.tensor(self.tokenizer.encode(prompt_str), dtype=torch.long),
            "prompt_str": prompt_str,
            "target_str": res_str
        }