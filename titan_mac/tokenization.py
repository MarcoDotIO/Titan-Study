from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_ref: str):
    if not tokenizer_ref:
        raise ValueError("A tokenizer path or model id is required.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
