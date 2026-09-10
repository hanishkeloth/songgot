"""One tokenizer for training, evaluation and deployment: llama.cpp's SentencePiece-compatible
tokenizer over Songgot's vocabulary, driven through the Python bindings with a vocab-only GGUF.

Why: SentencePiece and llama.cpp disagree on two things for this vocabulary. SentencePiece folds
newlines into spaces and needs a leading "▁" piece before each special token; llama.cpp keeps a
newline as the byte token <0x0A> and matches the specials as single ids when they are typed
USER_DEFINED. The app and every GGUF user run llama.cpp, so the model is post-trained and scored on
llama.cpp's ids. Pretraining used SentencePiece on plain text; the vocabulary is identical, only the
newline and special handling differ, and post-training adapts to that.

    from songgot_tok import Tok
    tok = Tok(); ids = tok.encode(prompt, bos=True); text = tok.decode(ids)
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOCAB_GGUF = ROOT / "data" / "vocab" / "songgot-vocab.gguf"
SPECIAL = ["<|system|>", "<|user|>", "<|call|>", "<|end|>", "<|pad|>"]


class Tok:
    def __init__(self, path: pathlib.Path | str = VOCAB_GGUF):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(path), vocab_only=True, verbose=False)
        self.bos_id, self.eos_id = self.llm.token_bos(), self.llm.token_eos()
        self.special = {s: 3 + i for i, s in enumerate(SPECIAL)}
        self.end_id, self.pad_id = self.special["<|end|>"], self.special["<|pad|>"]

    def encode(self, text: str, bos: bool = False) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=True)

    def decode(self, ids, keep_special: bool = False) -> str:
        return self.llm.detokenize(list(int(i) for i in ids), special=keep_special).decode("utf-8", "replace")

    def id_to_piece(self, i: int) -> str:
        return self.llm.detokenize([int(i)], special=True).decode("utf-8", "replace")
