# src/__init__.py

"""RL Arithmetic Fine-Tuning Package

A package for training and evaluating small Transformer models on arithmetic
tasks using reinforcement learning with curriculum learning and prioritized replay.
"""

__version__ = "0.1.0"
__author__ = "Nithin Nani"
__license__ = "MIT"

from .dataset import ArithmeticTokenizer, ArithmeticDataset
from .model import MiniTransformer

__all__ = [
    "ArithmeticTokenizer",
    "ArithmeticDataset", 
    "MiniTransformer",
]
