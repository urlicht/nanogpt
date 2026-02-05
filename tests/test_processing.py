import pytest

from nanogpt.data import get_unique_chars
from nanogpt.data.processing import (
    build_vocab,
    encode,
    decode,
    build_char_tokenizer,
    build_tiktoken_tokenizer,
    build_tokenizer,
)

class TestCharVocab:
    def test_get_unique_chars(self):
        test_str = 'aaabbbcccddd123'
        assert get_unique_chars(test_str) == ['1','2','3','a','b','c','d']

    def test_get_unique_chars_from_file(self, tmp_path):
        path = tmp_path / "sample.txt"
        path.write_text("bcaab", encoding="utf-8")

        assert get_unique_chars(str(path)) == ["a", "b", "c"]

    def test_build_vocab_sorted_and_consistent(self):
        test_str = "baba"
        stoi, itos = build_vocab(test_str)

        assert list(stoi.keys()) == ["a", "b"]
        assert list(itos.values()) == ["a", "b"]
        assert stoi["a"] == 0
        assert stoi["b"] == 1
        assert itos[0] == "a"
        assert itos[1] == "b"

        # consistency check
        for ch, idx in stoi.items():
            assert itos[idx] == ch

    def test_encode_decode_roundtrip(self):
        text = "abcab"
        stoi, itos = build_vocab(text)

        ids = encode(text, stoi)
        assert ids == [stoi[c] for c in text]

        decoded = decode(ids, itos)
        assert decoded == text

    def test_build_char_tokenizer_roundtrip(self):
        text = "hello"
        tokenizer = build_char_tokenizer(text)

        assert tokenizer.kind == "char"
        assert tokenizer.n_vocab == len(set(text))

        ids = tokenizer.encode(text)
        assert ids == encode(text, tokenizer.stoi)
        assert tokenizer.decode(ids) == text

    def test_build_tokenizer_char_requires_data(self):
        with pytest.raises(ValueError, match="data is required"):
            build_tokenizer("char")

    def test_build_tiktoken_tokenizer_roundtrip(self):
        text = "hello world"
        tokenizer = build_tiktoken_tokenizer("gpt2")

        assert tokenizer.kind == "tiktoken:gpt2"
        assert tokenizer.n_vocab > 0

        ids = tokenizer.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert tokenizer.decode(ids) == text

    def test_build_tokenizer_tiktoken(self):
        tokenizer = build_tokenizer("tiktoken", encoding_name="gpt2")
        assert tokenizer.kind == "tiktoken:gpt2"

    def test_build_tokenizer_unknown_method(self):
        with pytest.raises(ValueError, match="unknown tokenizer method"):
            build_tokenizer("bogus")