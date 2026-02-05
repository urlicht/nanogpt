import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import torch
import tiktoken

@dataclass
class DatasetBundle:
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    n_vocab: int

@dataclass
class Tokenizer:
    kind: str
    n_vocab: int
    encode: Callable[[str], list[int]]
    decode: Callable[[Iterable[int]], str]
    stoi: dict[str, int] | None = None
    itos: dict[int, str] | None = None
    encoding: tiktoken.Encoding | None = None

def load_text(path: str | Path) -> str:
    """Load raw text from disk."""
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    return text

def get_unique_chars(data, chunk_size=1024*1024):
    """
    Efficient unique character extractor
    input: strings, lists, or file paths
    """
    chars = set()
    
    # path, file streaming
    if isinstance(data, str) and os.path.isfile(data):
        with open(data, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chars.update(chunk) # C-speed update
                
    # string
    elif isinstance(data, str):
        chars.update(data)
        
    # iterable/list
    else:
        for item in data:
            chars.update(item)

    return sorted(chars)

def build_vocab(data) -> tuple[dict[str, int], dict[int, str]]:
    """Build vocab mappings"""
    chars = get_unique_chars(data)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}

    return stoi, itos

def encode(text: str, stoi: dict[str, int]):
    """Encode text into token IDs"""
    return [stoi[c] for c in text]

def decode(ids, itos: dict[int, str]):
    """Decode token IDs back into text"""
    return ''.join([itos[i] for i in ids])

def build_char_tokenizer(data) -> Tokenizer:
    """Build a character-level tokenizer from data."""
    stoi, itos = build_vocab(data)
    return Tokenizer(
        kind="char",
        n_vocab=len(stoi),
        encode=lambda text: encode(text, stoi),
        decode=lambda ids: decode(ids, itos),
        stoi=stoi,
        itos=itos,
    )

def build_tiktoken_tokenizer(name: str = "gpt2") -> Tokenizer:
    """Build a tiktoken tokenizer (e.g., GPT-2)."""
    encoding = tiktoken.get_encoding(name)
    return Tokenizer(
        kind=f"tiktoken:{name}",
        n_vocab=encoding.n_vocab,
        encode=encoding.encode,
        decode=encoding.decode,
        encoding=encoding,
    )

def build_tokenizer(method: str, data=None, *, encoding_name: str = "gpt2") -> Tokenizer:
    """Build a tokenizer for either char or tiktoken encoding."""
    if method == "char":
        if data is None:
            raise ValueError("data is required for char tokenizer")
        return build_char_tokenizer(data)
    if method == "tiktoken":
        return build_tiktoken_tokenizer(encoding_name)
    raise ValueError(f"unknown tokenizer method: {method}")
