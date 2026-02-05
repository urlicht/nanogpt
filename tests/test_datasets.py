import pytest
import json
import numpy as np
from nanogpt.data.datasets import prepare_dataset

class DummyTokenizer:
    def __init__(self, n_vocab: int = 256, use_encode_ordinary: bool = False):
        self.n_vocab = n_vocab
        self.kind = "dummy"
        if use_encode_ordinary:
            self.encode_ordinary = self.encode

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.n_vocab for c in text]

class BadTokenizer(DummyTokenizer):
    def encode(self, text: str) -> list[int]:
        return [self.n_vocab]

def _read_bin(path, dtype=np.uint16) -> np.ndarray:
    return np.fromfile(str(path), dtype=dtype)

class TestDataset:
    def test_prepare_dataset_raw_string(self, tmp_path):
        text = "hello!"
        tokenizer = DummyTokenizer(n_vocab=256, use_encode_ordinary=True)
        out_dir = tmp_path / "out"

        prepare_dataset(
            text,
            tokenizer,
            output_dir=str(out_dir),
            val_fraction=0.5,
            metadata_format="json",
        )

        train = _read_bin(out_dir / "train.bin")
        val = _read_bin(out_dir / "val.bin")

        split_idx = int(len(text) * (1 - 0.5))
        assert len(train) == split_idx
        assert len(val) == len(text) - split_idx

        meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["train_tokens"] == len(train)
        assert meta["val_tokens"] == len(val)
        assert meta["val_fraction"] == 0.5
        assert meta["n_vocab"] == tokenizer.n_vocab
        assert meta["dtype"] == "uint16"

    def test_prepare_dataset_file_stream_all_val(self, tmp_path):
        path = tmp_path / "input.txt"
        path.write_text("ab\ncd\n", encoding="utf-8")

        tokenizer = DummyTokenizer(n_vocab=256)
        out_dir = tmp_path / "out"

        prepare_dataset(
            str(path),
            tokenizer,
            output_dir=str(out_dir),
            val_fraction=1.0,
            metadata_format="json",
        )

        train = _read_bin(out_dir / "train.bin")
        val = _read_bin(out_dir / "val.bin")

        assert len(train) == 0
        assert len(val) == len(path.read_text(encoding="utf-8"))

    def test_prepare_dataset_vocab_overflow_guard(self, tmp_path):
        tokenizer = DummyTokenizer(n_vocab=70000)
        out_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="does not fit in dtype"):
            prepare_dataset(
                "hi",
                tokenizer,
                output_dir=str(out_dir),
                dtype=np.uint16,
                metadata_format="json",
            )

    def test_prepare_dataset_token_id_overflow(self, tmp_path):
        tokenizer = BadTokenizer(n_vocab=10)
        out_dir = tmp_path / "out"

        with pytest.raises(ValueError, match="found token id"):
            prepare_dataset(
                "hi",
                tokenizer,
                output_dir=str(out_dir),
                dtype=np.uint16,
                metadata_format="json",
            )