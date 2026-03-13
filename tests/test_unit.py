from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from infer import sample_next_token
from titan_mac.checkpoint import load_checkpoint, save_checkpoint
from titan_mac.config import build_model_config
from titan_mac.data import (
    document_split,
    iter_fineweb_records,
    iter_dolma_records,
    pack_tokenized_documents,
    resolve_fineweb_files,
    resolve_dataset_files,
)
from titan_mac.device import resolve_device
from titan_mac.model import TitansMACLM


def test_iter_dolma_records_reads_expected_fields(dataset_path: Path) -> None:
    record = next(iter(iter_dolma_records(dataset_path, max_docs=1)))
    assert record.doc_id
    assert record.text
    assert record.source == "gutenberg"
    assert isinstance(record.metadata, dict)


def test_iter_dolma_records_reads_bulk_gzip_directory(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "bulk"
    dataset_dir.mkdir()

    shard_a = dataset_dir / "books-0001.json.gz"
    shard_b = dataset_dir / "books-0002.json.gz"
    records = [
        {
            "id": "doc-a",
            "text": "Alpha document",
            "source": "gutenberg",
            "metadata": {"shard": "a"},
        },
        {
            "id": "doc-b",
            "text": "Beta document",
            "source": "gutenberg",
            "metadata": {"shard": "b"},
        },
    ]

    with gzip.open(shard_b, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(records[1]) + "\n")
    with gzip.open(shard_a, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(records[0]) + "\n")

    dataset_files = resolve_dataset_files(dataset_dir)
    assert dataset_files == [shard_a, shard_b]

    loaded = list(iter_dolma_records(dataset_dir))
    assert [record.doc_id for record in loaded] == ["doc-a", "doc-b"]


def test_iter_fineweb_records_reads_parquet_directory(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "fineweb"
    dataset_dir.mkdir()

    shard_a = dataset_dir / "part-0001.parquet"
    shard_b = dataset_dir / "part-0002.parquet"
    table_a = pa.table(
        {
            "text": ["Alpha educational text."],
            "id": ["row-a"],
            "dump": ["CC-MAIN-2024-10"],
            "url": ["https://example.com/a"],
        }
    )
    table_b = pa.table(
        {
            "text": ["Beta educational text."],
            "id": ["row-b"],
            "dump": ["CC-MAIN-2024-18"],
            "url": ["https://example.com/b"],
        }
    )
    pq.write_table(table_b, shard_b)
    pq.write_table(table_a, shard_a)

    dataset_files = resolve_fineweb_files(dataset_dir)
    assert dataset_files == [shard_a, shard_b]

    loaded = list(iter_fineweb_records(dataset_dir))
    assert [record.doc_id for record in loaded] == ["row-a", "row-b"]
    assert [record.source for record in loaded] == ["CC-MAIN-2024-10", "CC-MAIN-2024-18"]


def test_document_split_is_deterministic() -> None:
    doc_id = "600f7d0e70e779b5c95464411000c5998ea252ba"
    assert document_split(doc_id) == document_split(doc_id)
    assert document_split(doc_id) in {"train", "val"}


def test_pack_tokenized_documents_produces_fixed_windows() -> None:
    sequences = pack_tokenized_documents(
        [[1, 2, 3], [4, 5, 6], [7, 8]],
        seq_len=4,
        eos_token_id=99,
    )
    assert sequences == [[1, 2, 3, 99], [4, 5, 6, 99]]


def test_device_resolution_supports_explicit_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_mac_model_forward_backward_stays_finite() -> None:
    config = build_model_config("tiny_test", vocab_size=128)
    model = TitansMACLM(config)
    model.train()
    batch = torch.randint(0, config.vocab_size, (2, 64), dtype=torch.long)
    output = model(batch, labels=batch, reset_memory=True)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    total_grad = sum(parameter.grad.abs().sum().item() for parameter in model.parameters() if parameter.grad is not None)
    assert total_grad > 0.0


def test_mac_model_initial_loss_is_reasonably_scaled() -> None:
    torch.manual_seed(0)
    config = build_model_config("tiny_test", vocab_size=128)
    model = TitansMACLM(config)
    batch = torch.randint(0, config.vocab_size, (2, 64), dtype=torch.long)
    output = model(batch, labels=batch, reset_memory=True)
    assert output.loss is not None
    assert output.loss.item() < 15.0


def test_mac_init_state_fast_params_require_grad() -> None:
    config = build_model_config("tiny_test", vocab_size=128)
    model = TitansMACLM(config)
    state = model.init_memory_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    for layer_state in state.layer_states:
        assert all(tensor.requires_grad for tensor in layer_state.weights)
        assert all(tensor.requires_grad for tensor in layer_state.biases)
        assert all(not tensor.requires_grad for tensor in layer_state.momentum_weights)
        assert all(not tensor.requires_grad for tensor in layer_state.momentum_biases)


def test_checkpoint_round_trip_preserves_step(tmp_path: Path) -> None:
    config = build_model_config("tiny_test", vocab_size=64)
    model = TitansMACLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=4)
    checkpoint_path = tmp_path / "round_trip.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        model_config=config,
        train_args={"preset": "tiny_test"},
        tokenizer_ref="tests-tokenizer",
        step=7,
    )
    payload = load_checkpoint(checkpoint_path)
    assert payload["step"] == 7
    assert payload["tokenizer_ref"] == "tests-tokenizer"
    assert payload["model_config"]["max_seq_len"] == config.max_seq_len
    assert not (tmp_path / "round_trip.pt.tmp").exists()


def test_sample_next_token_handles_non_finite_logits() -> None:
    logits = torch.tensor([[float("nan"), float("inf"), -1.0, 0.5]])
    next_token = sample_next_token(logits, temperature=0.8, top_k=2, top_p=0.9)
    assert next_token.shape == (1, 1)
    assert int(next_token.item()) in {1, 3}
