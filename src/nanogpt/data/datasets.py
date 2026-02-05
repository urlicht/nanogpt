import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

@dataclass
class DatasetMetadata:
    source: str
    is_huggingface: bool
    streaming: bool
    text_field: str
    val_fraction: float | None
    tokenizer_kind: str
    n_vocab: int
    dtype: str
    train_tokens: int
    val_tokens: int
    train_file: str
    val_file: str
    created_at: str

def prepare_dataset(
    input_source,
    tokenizer,
    output_dir:str="data_out",
    val_fraction:float|None=None,
    is_huggingface:bool=False,
    text_field: str = "text",
    streaming: bool = True,
    seed: int = 3141592,
    dtype: np.dtype = np.uint16,
    metadata_format: str = "json",
    metadata_filename: str | None = None,
):
    """
    Standardizes data into train.bin and val.bin files.
    
    Args:
        input_source: HF path (e.g. 'TinyStories'), file path, or raw string.
        tokenizer: an object with an .encode() method (tiktoken or custom).
        output_dir: folder to save .bin files.
        test_size: ratio for validation split (0.0 to 1.0) or None.
        is_huggingface: boolean, set True if loading from Hugging Face hub.
        text_field: name of the field containing text in HF datasets.
        streaming: stream data instead of loading into memory.
        seed: RNG seed for streaming split when no validation split exists.
        dtype: dtype for .bin output. Must fit vocab size.
        metadata_format: "json" or "yaml".
        metadata_filename: override metadata filename (defaults to metadata.json/yaml).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_vocab = _get_n_vocab(tokenizer)
    _check_vocab_fits_dtype(n_vocab, dtype)
    val_fraction_meta = val_fraction
    
    # 1. load data
    if is_huggingface:
        dataset = load_dataset(input_source, streaming=streaming)
        if val_fraction is None and "train" in dataset and "validation" in dataset:
            train_iter = dataset["train"]
            val_iter = dataset["validation"]
            val_fraction = 0.0
        else:
            split_name = "train" if "train" in dataset else next(iter(dataset))
            train_iter = dataset[split_name]
            val_iter = None
            if val_fraction is None:
                val_fraction = 0.0
                val_fraction_meta = None
    
    elif os.path.isfile(input_source):
        train_iter = _iter_file_lines(input_source)
        val_iter = None
        if val_fraction is None:
            val_fraction = 0.0
            val_fraction_meta = None
        
    else: # assume input_source is a raw string
        if val_fraction is None:
            val_fraction = 0.0
            val_fraction_meta = None
        raw_train, raw_val = _split_text(input_source, val_fraction)
        train_iter = [raw_train]
        val_iter = [raw_val] if raw_val else None

    # 2. tokenize and save
    train_path = output_path / "train.bin"
    val_path = output_path / "val.bin"
    train_tokens = 0
    val_tokens = 0

    with open(train_path, "wb") as train_f, open(val_path, "wb") as val_f:
        if is_huggingface:
            if val_iter is not None:
                train_tokens += _stream_hf_split(
                    train_iter,
                    tokenizer,
                    train_f,
                    dtype,
                    n_vocab,
                    text_field,
                    desc="train",
                )
                val_tokens += _stream_hf_split(
                    val_iter,
                    tokenizer,
                    val_f,
                    dtype,
                    n_vocab,
                    text_field,
                    desc="val",
                )
            elif val_fraction > 0:
                train_tokens, val_tokens = _stream_hf_with_split(
                    train_iter,
                    tokenizer,
                    train_f,
                    val_f,
                    dtype,
                    n_vocab,
                    text_field,
                    val_fraction,
                    rng,
                )
            else:
                train_tokens += _stream_hf_split(
                    train_iter,
                    tokenizer,
                    train_f,
                    dtype,
                    n_vocab,
                    text_field,
                    desc="train",
                )
        elif os.path.isfile(input_source):
            train_tokens, val_tokens = _stream_text_with_split(
                train_iter,
                tokenizer,
                train_f,
                val_f,
                dtype,
                n_vocab,
                val_fraction,
                rng,
                desc="file",
            )
        else:
            if raw_train:
                train_tokens += _write_text_chunk(
                    raw_train,
                    tokenizer,
                    train_f,
                    dtype,
                    n_vocab,
                )
            if raw_val:
                val_tokens += _write_text_chunk(
                    raw_val,
                    tokenizer,
                    val_f,
                    dtype,
                    n_vocab,
                )

    metadata = DatasetMetadata(
        source=str(input_source),
        is_huggingface=is_huggingface,
        streaming=streaming,
        text_field=text_field,
        val_fraction=val_fraction_meta,
        tokenizer_kind=_tokenizer_kind(tokenizer),
        n_vocab=n_vocab,
        dtype=str(np.dtype(dtype)),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        train_file=str(train_path),
        val_file=str(val_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_metadata(metadata, output_path, metadata_format, metadata_filename)

    print(f"train saved to {train_path} ({train_tokens} tokens)")
    if val_tokens > 0:
        print(f"val saved to {val_path} ({val_tokens} tokens)")

def _split_text(text, test_fraction):
    if test_fraction <= 0:
        return text, ""
    n = len(text)
    split_idx = int(n * (1 - test_fraction))
    return text[:split_idx], text[split_idx:]

def _get_n_vocab(tokenizer) -> int:
    if hasattr(tokenizer, "n_vocab"):
        return int(tokenizer.n_vocab)
    if hasattr(tokenizer, "vocab_size"):
        return int(tokenizer.vocab_size)
    raise ValueError("tokenizer must expose n_vocab or vocab_size")

def _check_vocab_fits_dtype(n_vocab: int, dtype: np.dtype) -> None:
    max_value = np.iinfo(dtype).max
    if n_vocab - 1 > max_value:
        raise ValueError(
            f"n_vocab {n_vocab} does not fit in dtype {dtype} "
            f"(max {max_value})"
        )

def _encode_text(tokenizer, text: str) -> list[int]:
    encode_ordinary = getattr(tokenizer, "encode_ordinary", None)
    if callable(encode_ordinary):
        return encode_ordinary(text)
    return tokenizer.encode(text)

def _write_ids(
    ids: list[int],
    file_obj,
    dtype: np.dtype,
    n_vocab: int,
) -> int:
    if not ids:
        return 0
    max_id = max(ids)
    if max_id >= n_vocab:
        raise ValueError(
            f"found token id {max_id} but n_vocab is {n_vocab}"
        )
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(file_obj)
    return len(arr)

def _write_text_chunk(
    text: str,
    tokenizer,
    file_obj,
    dtype: np.dtype,
    n_vocab: int,
) -> int:
    ids = _encode_text(tokenizer, text)
    return _write_ids(ids, file_obj, dtype, n_vocab)

def _iter_file_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line

def _stream_text_with_split(
    iterator: Iterable[str],
    tokenizer,
    train_f,
    val_f,
    dtype: np.dtype,
    n_vocab: int,
    val_fraction: float,
    rng: np.random.Generator,
    desc: str,
) -> tuple[int, int]:
    train_tokens = 0
    val_tokens = 0
    for text in tqdm(iterator, desc=f"tokenizing {desc}", unit="lines"):
        target = "val" if rng.random() < val_fraction else "train"
        if target == "train":
            train_tokens += _write_text_chunk(text, tokenizer, train_f, dtype, n_vocab)
        else:
            val_tokens += _write_text_chunk(text, tokenizer, val_f, dtype, n_vocab)
    return train_tokens, val_tokens

def _stream_hf_split(
    iterator,
    tokenizer,
    file_obj,
    dtype: np.dtype,
    n_vocab: int,
    text_field: str,
    desc: str,
) -> int:
    tokens = 0
    for example in tqdm(iterator, desc=f"tokenizing {desc}", unit="examples"):
        text = example.get(text_field)
        if text is None:
            raise KeyError(f"missing text field '{text_field}' in dataset example")
        if isinstance(text, list):
            text = "".join(text)
        tokens += _write_text_chunk(text, tokenizer, file_obj, dtype, n_vocab)
    return tokens

def _stream_hf_with_split(
    iterator,
    tokenizer,
    train_f,
    val_f,
    dtype: np.dtype,
    n_vocab: int,
    text_field: str,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    train_tokens = 0
    val_tokens = 0
    for example in tqdm(iterator, desc="tokenizing", unit="examples"):
        text = example.get(text_field)
        if text is None:
            raise KeyError(f"missing text field '{text_field}' in dataset example")
        if isinstance(text, list):
            text = "".join(text)
        if rng.random() < val_fraction:
            val_tokens += _write_text_chunk(text, tokenizer, val_f, dtype, n_vocab)
        else:
            train_tokens += _write_text_chunk(text, tokenizer, train_f, dtype, n_vocab)
    return train_tokens, val_tokens

def _tokenizer_kind(tokenizer) -> str:
    if hasattr(tokenizer, "kind"):
        return str(tokenizer.kind)
    return tokenizer.__class__.__name__

def _save_metadata(
    metadata: DatasetMetadata,
    output_path: Path,
    metadata_format: str,
    metadata_filename: str | None,
) -> None:
    payload = asdict(metadata)
    fmt = metadata_format.lower()
    if metadata_filename is None:
        metadata_filename = f"metadata.{fmt}"
    meta_path = output_path / metadata_filename

    if fmt == "json":
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    elif fmt == "yaml":
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required for metadata_format='yaml'"
            ) from exc
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    else:
        raise ValueError("metadata_format must be 'json' or 'yaml'")