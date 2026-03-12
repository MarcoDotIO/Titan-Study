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
    offset = 0  # track start of unconsumed buffer to avoid O(n) front-deletions
    for document_tokens in tokenized_documents:
        if not document_tokens:
            continue
        buffer.extend(document_tokens)
        buffer.append(eos_token_id)
        while len(buffer) - offset >= seq_len:
            sequences.append(buffer[offset : offset + seq_len])
            offset += seq_len
            if max_sequences is not None and len(sequences) >= max_sequences:
                return sequences
    # Compact the buffer only once at the end to avoid repeated O(n) deletions
    del buffer[:offset]
    return sequences


def build_split_sequences(
    dataset_path: str | Path,
    tokenizer,
    *,
    seq_len: int,
    max_docs: int | None = None,
    max_sequences_train: int | None = None,
    max_sequences_val: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Single-pass over the dataset, splitting records into train/val as we go."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token id for sequence packing.")

    eos = tokenizer.eos_token_id
    train_seqs: list[list[int]] = []
    val_seqs: list[list[int]] = []
    train_buf: list[int] = []
    val_buf: list[int] = []
    train_offset = 0
    val_offset = 0

    train_done = False
    val_done = False

    for record in iter_dolma_records(dataset_path, max_docs=max_docs):
        split = document_split(record.doc_id)
        if split == "train" and train_done:
            continue
        if split == "val" and val_done:
            continue

        encoded = tokenizer(record.text, add_special_tokens=False, truncation=False)
        token_ids = encoded["input_ids"]
        if not token_ids:
            continue

        if split == "train":
            train_buf.extend(token_ids)
            train_buf.append(eos)
            while len(train_buf) - train_offset >= seq_len:
                train_seqs.append(train_buf[train_offset : train_offset + seq_len])
                train_offset += seq_len
                if max_sequences_train is not None and len(train_seqs) >= max_sequences_train:
                    train_done = True
                    break
        else:
            val_buf.extend(token_ids)
            val_buf.append(eos)
            while len(val_buf) - val_offset >= seq_len:
                val_seqs.append(val_buf[val_offset : val_offset + seq_len])
                val_offset += seq_len
                if max_sequences_val is not None and len(val_seqs) >= max_sequences_val:
                    val_done = True
                    break

        if train_done and val_done:
            break

    return train_seqs, val_seqs


def load_datasets(
    dataset_path: str | Path,
    tokenizer,
    *,
    seq_len: int,
    max_docs: int | None = None,
    max_sequences: int | None = None,
) -> tuple[PackedSequenceDataset, PackedSequenceDataset]:
    train_sequences, val_sequences = build_split_sequences(
        dataset_path,
        tokenizer,
        seq_len=seq_len,
        max_docs=max_docs,
        max_sequences_train=max_sequences,
        max_sequences_val=max_sequences,
    )
    return PackedSequenceDataset(train_sequences), PackedSequenceDataset(val_sequences)
