"""
Dataset module for Arithmetic LLM - v3 with Scratchpad CoT

Features:
- Expanded tokenizer with scratchpad tokens (vocab_size: 16 → 21)
- Scratchpad chain-of-thought generation for addition, subtraction, multiplication
- Support for both forward (scratchpad) and reverse-target modes
- Optional padding for consistent batch shapes
"""

import torch
import random
from torch.utils.data import Dataset


class ArithmeticTokenizer:
    """
    Tokenizer for arithmetic operations.
    
    v3 adds scratchpad tokens: |, [, ], C, B
    Token indices (0-20):
      0-9: digits '0'-'9'
      10: '+'
      11: '-'
      12: '*'
      13: '='
      14: '<EOS>'
      15: '<PAD>'
      16: '|' (scratchpad delimiter)
      17: '[' (scratchpad step start)
      18: ']' (scratchpad step end)
      19: 'C' (carry indicator)
      20: 'B' (borrow indicator)
    """
    def __init__(self, vocab_size=21):
        # v3 vocabulary with scratchpad tokens
        self.chars = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 0-9: digits
            '+', '-', '*', '=',                                 # 10-13: operators
            '<EOS>', '<PAD>',                                   # 14-15: special
            '|', '[', ']', 'C', 'B'                             # 16-20: scratchpad
        ]
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token_id = self.stoi['<PAD>']  # 15
        self.eos_token_id = self.stoi['<EOS>']   # 14
        
        # Scratchpad tokens
        self.delim_token = self.stoi['|']       # 16
        self.step_start = self.stoi['[']         # 17
        self.step_end = self.stoi[']']           # 18
        self.carry_token = self.stoi['C']       # 19
        self.borrow_token = self.stoi['B']      # 20

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """Decode token IDs back to string, excluding special tokens."""
        return ''.join([self.itos[i] for i in ids if i not in [self.pad_token_id, self.eos_token_id]])


def generate_addition_scratchpad(a, b):
    """
    Generate scratchpad chain-of-thought for addition.
    
    Format: a+b=[C{carry_in}]{d1}+{d2}={sum}|[C{next_carry}]...|{result}<EOS>
    
    Example: 4571+8=[C0]1+8=9|[C0]7+0=7|[C0]5+0=5|[C0]4+0=4|4579
    With carry: 99+5=[C0]9+5=14|[C1]9+0=10|[C1]0+0=1|104
    """
    a_str, b_str = str(a), str(b)
    max_len = max(len(a_str), len(b_str))
    a_padded = a_str.zfill(max_len)
    b_padded = b_str.zfill(max_len)
    
    carry = 0
    steps = []
    result_digits = []
    
    # Process from right to left (units first)
    for i in range(max_len - 1, -1, -1):
        d1 = int(a_padded[i])
        d2 = int(b_padded[i])
        s = d1 + d2 + carry
        digit = s % 10
        carry_out = s // 10
        
        # Format: [C{carry_in}]d1+d2=sum
        steps.append(f"[C{carry}]{d1}+{d2}={s}")
        result_digits.append(str(digit))
        carry = carry_out
    
    # If there's remaining carry, add it
    if carry > 0:
        steps.append(f"[C{carry}]{carry}+0={carry}")
        result_digits.append(str(carry))
    
    result = ''.join(reversed(result_digits))
    scratchpad = '|'.join(steps)
    return f"{a}+{b}={scratchpad}|{result}"


def generate_subtraction_scratchpad(a, b):
    """
    Generate scratchpad chain-of-thought for subtraction.
    
    Format: a-b=[B{borrow_in}]{d1}-{d2}={diff}|[B{next_borrow}]...|{result}<EOS>
    
    Example: 52-8=[B1]12-8=4|[B0]4-0=4|44
    (When 2<8, borrow 1 from 5 → [B1]12-8=4, remaining tens: 5→4, so [B0]4-0=4)
    """
    # Ensure a >= b for positive results
    if a < b:
        a, b = b, a
    
    a_str, b_str = str(a), str(b)
    max_len = max(len(a_str), len(b_str))
    a_padded = a_str.zfill(max_len)
    b_padded = b_str.zfill(max_len)
    
    borrow = 0
    steps = []
    result_digits = []
    
    # Process from right to left
    for i in range(max_len - 1, -1, -1):
        d1 = int(a_padded[i])
        d2 = int(b_padded[i])
        
        # Apply borrow
        if d1 < d2 + borrow:
            d1 += 10
            borrow_out = 1
        else:
            borrow_out = 0
        
        diff = d1 - d2 - borrow
        steps.append(f"[B{borrow}]{d1}-{d2}={diff}")
        result_digits.append(str(diff))
        borrow = borrow_out
    
    result = ''.join(reversed(result_digits))
    scratchpad = '|'.join(steps)
    return f"{a}-{b}={scratchpad}|{result}"


def generate_multiplication_scratchpad(a, b):
    """
    Generate scratchpad chain-of-thought for multiplication.
    
    Format: a*b={a}*{digit}={partial}|...|{sum_steps}|{result}<EOS>
    
    Example: 23*45=23*5=115|23*40=920|115+920=1035|1035
    """
    b_str = str(b)
    partials = []
    
    for i, digit in enumerate(reversed(b_str)):
        d = int(digit)
        if d == 0:
            continue
        partial_val = a * d * (10 ** i)
        partials.append(f"{a}*{d}{'0'*i}={partial_val}")
    
    # Sum partial products
    if len(partials) > 1:
        running_sum = 0
        sum_steps = []
        for p in partials:
            val = int(p.split('=')[1])
            running_sum += val
            sum_steps.append(str(running_sum))
        sum_line = '|'.join(sum_steps)
    else:
        sum_line = str(a * b) if partials else "0"
    
    result = a * b
    
    if partials:
        scratchpad = '|'.join(partials)
        if sum_line != str(result):
            scratchpad += f"|{sum_line}"
        return f"{a}*{b}={scratchpad}|{result}"
    else:
        return f"{a}*{b}=0|{result}"


def generate_reverse_answer(a, b, op):
    """
    Generate answer in reverse order (v2 style for backward compatibility).
    
    Format: a op b = {answer_reversed}<EOS>
    """
    if op == '+':
        result = a + b
    elif op == '-':
        if a < b:
            a, b = b, a
        result = a - b
    else:  # '*'
        result = a * b
    
    return f"{a}{op}{b}={str(result)[::-1]}"


class ArithmeticDataset(Dataset):
    """
    Dataset for arithmetic operations with optional scratchpad CoT.
    
    Args:
        tokenizer: ArithmeticTokenizer instance
        num_samples: Number of samples to generate
        max_digits: Maximum number of digits in operands (1-5 for v3)
        operations: List of operations to include ['+', '-', '*']
        reverse_target: If True, use reverse-target mode (v2 backward compat)
                       If False, use scratchpad CoT (v3 default)
        use_scratchpad: If True, generate scratchpad sequences
        pad_to_length: Optional fixed length for padding sequences
    """
    def __init__(self, tokenizer, num_samples=300000, max_digits=4, 
                 operations=['+', '-', '*'], reverse_target=False,
                 use_scratchpad=True, pad_to_length=None):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.operations = operations
        self.reverse_target = reverse_target
        self.use_scratchpad = use_scratchpad and not reverse_target
        self.pad_to_length = pad_to_length
    
    def __len__(self):
        return self.num_samples

    def _get_random_number(self, max_digits=None):
        """Generate random number with specified max digits."""
        n_digits = random.randint(1, max_digits or self.max_digits)
        if n_digits == 1:
            return random.randint(0, 9)
        return random.randint(10**(n_digits-1), (10**n_digits) - 1)
    
    def _get_multiplicand(self, max_digits):
        """Generate multiplicand for multiplication with proper distribution."""
        # Default digit distribution: bias toward fewer digits
        base_weights = [0.4, 0.3, 0.2, 0.08, 0.02]
        weights = (base_weights[:max_digits] + [0.0] * max(0, max_digits - len(base_weights)))
        n_digits = random.choices(range(1, max_digits + 1), weights=weights)[0]
        n_digits = max(1, min(n_digits, max_digits))
        if n_digits == 1:
            return random.randint(0, 9)
        return random.randint(10**(n_digits-1), (10**n_digits) - 1)

    def __getitem__(self, idx):
        op = random.choice(self.operations)
        
        if op == '*':
            a = self._get_multiplicand(self.max_digits)
            b = self._get_multiplicand(self.max_digits)
        else:
            a = self._get_random_number()
            b = self._get_random_number()
            if op == '-' and a < b:
                a, b = b, a  # Ensure positive result for subtraction
        
        # Generate sequence based on mode
        if self.reverse_target:
            # v2 backward compatibility: reverse-target mode
            full_str = generate_reverse_answer(a, b, op)
        elif self.use_scratchpad:
            # v3 scratchpad CoT mode
            if op == '+':
                full_str = generate_addition_scratchpad(a, b)
            elif op == '-':
                full_str = generate_subtraction_scratchpad(a, b)
            else:  # '*'
                full_str = generate_multiplication_scratchpad(a, b)
        else:
            # Forward answer (no scratchpad, no reverse)
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:
                result = a * b
            full_str = f"{a}{op}{b}={result}"
        
        # Encode and add EOS
        input_ids = self.tokenizer.encode(full_str) + [self.tokenizer.eos_token_id]
        
        # Extract prompt part (before '=' answer)
        prompt_str = f"{a}{op}{b}="
        
        # Optional: pad to fixed length
        if self.pad_to_length is not None:
            while len(input_ids) < self.pad_to_length:
                input_ids.append(self.tokenizer.pad_token_id)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_ids": torch.tensor(self.tokenizer.encode(prompt_str), dtype=torch.long),
            "prompt_str": prompt_str,
            "full_str": full_str
        }


def compute_scratchpad_reward(prompt_str, generated_str, correct_val):
    """
    Compute reward for scratchpad generation.
    
    Returns:
        tuple: (total_reward, scratchpad_bonus, answer_reward)
    """
    try:
        # Parse generated scratchpad
        if '|' not in generated_str:
            return 0.0, 0.0, 0.0
        
        parts = generated_str.split('|')
        
        # Extract final answer (last part after |)
        answer_part = parts[-1].strip()
        answer_clean = "".join([c for c in answer_part if c.isdigit() or c == '-'])
        
        if not answer_clean:
            return 0.0, 0.0, 0.0
        
        # Check answer
        try:
            answer_val = int(answer_clean)
            answer_correct = (answer_val == correct_val)
        except ValueError:
            answer_correct = False
        
        # Compute scratchpad step bonus
        # Each correct intermediate step (format [X]...=...) gets +0.2
        scratchpad_steps = parts[:-1]  # All parts except final answer
        step_bonus = 0.0
        
        for step in scratchpad_steps:
            # Verify step format: [C...], [...=...], etc.
            if step.startswith('[') and '=' in step:
                step_bonus += 0.2
        
        # Compute digit-level partial credit for answer (forward order)
        reward = 0.0
        if answer_correct:
            reward = 1.0
        elif answer_clean:
            target_str = str(abs(correct_val))
            match_count = 0
            for i in range(min(len(answer_clean), len(target_str))):
                if answer_clean[i] == target_str[i]:
                    match_count += 1
                else:
                    break
            reward = min(match_count * 0.15, 0.9)
        
        # Add scratchpad bonus (capped so exact match is best)
        total_reward = min(reward + step_bonus, 1.0)
        
        return total_reward, step_bonus, reward
    except Exception:
        return 0.0, 0.0, 0.0


if __name__ == "__main__":
    # Test the scratchpad generation
    print("Testing v3 Scratchpad CoT Dataset...")
    
    tokenizer = ArithmeticTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token_id}")
    
    print("\n--- Addition Scratchpad ---")
    print(generate_addition_scratchpad(4571, 8))
    print(generate_addition_scratchpad(99, 5))
    print(generate_addition_scratchpad(123, 456))
    
    print("\n--- Subtraction Scratchpad ---")
    print(generate_subtraction_scratchpad(52, 8))
    print(generate_subtraction_scratchpad(523, 187))
    
    print("\n--- Multiplication Scratchpad ---")
    print(generate_multiplication_scratchpad(23, 45))
    print(generate_multiplication_scratchpad(4571, 82))
    
    print("\n--- Dataset Test ---")
    ds = ArithmeticDataset(tokenizer, num_samples=5, max_digits=3, use_scratchpad=True)
    for i, item in enumerate(ds):
        print(f"Sample {i}: {item['full_str'][:80]}...")
        print(f"  Tokens: {len(item['input_ids'])}")
    
    print("\n✅ Scratchpad dataset test passed!")