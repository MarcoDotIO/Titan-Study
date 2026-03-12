from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DolmaRecord:
    doc_id: str
    text: str
    source: str
    metadata: dict


class PackedSequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]]):
        self._sequences = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences]

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._sequences[index]


def resolve_dataset_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_file():
        return [path]

    preferred_patterns = ("*.json.gz", "*.jsonl.gz", "*.ndjson.gz")
    files: list[Path] = []
    for pattern in preferred_patterns:
        files.extend(path.rglob(pattern))

    if not files:
        files = list(path.rglob("*.gz"))

    files = sorted(file_path for file_path in files if file_path.is_file())
    if not files:
        raise FileNotFoundError(
            f"Dataset directory does not contain any gzip files: {path}"
        )
    return files


def iter_dolma_records(dataset_path: str | Path, max_docs: int | None = None) -> Iterator[DolmaRecord]:
    dataset_files = resolve_dataset_files(dataset_path)
    docs_seen = 0
    for path in dataset_files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                text = raw.get("text", "")
                doc_id = raw.get("id")
                if not doc_id or not text:
                    continue
                yield DolmaRecord(
                    doc_id=doc_id,
                    text=text,
                    source=raw.get("source", ""),
                    metadata=raw.get("metadata", {}),
                )
                docs_seen += 1
                if max_docs is not None and docs_seen >= max_docs:
                    return


def document_split(doc_id: str, validation_percent: int = 1) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < validation_percent else "train"


def pack_tokenized_documents(
    tokenized_documents: Iterable[list[int]],
    *,
    seq_len: int,
    eos_token_id: int,
    max_sequences: int | None = None,
) -> list[list[int]]:
    sequences: list[list[int]] = []
    buffer: list[int] = []
    for document_tokens in tokenized_documents:
        if not document_tokens:
            continue
        buffer.extend(document_tokens)
        buffer.append(eos_token_id)
        while len(buffer) >= seq_len:
            sequences.append(buffer[:seq_len])
            del buffer[:seq_len]
            if max_sequences is not None and len(sequences) >= max_sequences:
                return sequences
    return sequences


def build_packed_sequences(
    dataset_path: str | Path,
    tokenizer,
    *,
    split: str,
    seq_len: int,
    max_docs: int | None = None,
    max_sequences: int | None = None,
) -> list[list[int]]:
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token id for sequence packing.")

    def tokenized_documents() -> Iterator[list[int]]:
        for record in iter_dolma_records(dataset_path, max_docs=max_docs):
            if document_split(record.doc_id) != split:
                continue
            encoded = tokenizer(record.text, add_special_tokens=False, truncation=False)
            token_ids = encoded["input_ids"]
            if token_ids:
                yield token_ids

    return pack_tokenized_documents(
        tokenized_documents(),
        seq_len=seq_len,
        eos_token_id=tokenizer.eos_token_id,
        max_sequences=max_sequences,
    )


def load_datasets(
    dataset_path: str | Path,
    tokenizer,
    *,
    seq_len: int,
    max_docs: int | None = None,
    max_sequences: int | None = None,
) -> tuple[PackedSequenceDataset, PackedSequenceDataset]:
    train_sequences = build_packed_sequences(
        dataset_path,
        tokenizer,
        split="train",
        seq_len=seq_len,
        max_docs=max_docs,
        max_sequences=max_sequences,
    )
    val_sequences = build_packed_sequences(
        dataset_path,
        tokenizer,
        split="val",
        seq_len=seq_len,
        max_docs=max_docs,
        max_sequences=max_sequences,
    )
    return PackedSequenceDataset(train_sequences), PackedSequenceDataset(val_sequences)
