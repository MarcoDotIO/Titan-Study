from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def dataset_path(repo_root: Path) -> Path:
    return repo_root / ".dataset" / "books-0000.json.gz"


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("tiny_tokenizer")
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    corpus = [
        "Titans learn to memorize books and documents.",
        "Dolma contains long documents from Project Gutenberg.",
        "Memory as Context keeps a persistent prefix and retrieved memory tokens.",
        "The quick brown fox jumps over the lazy dog.",
        "PyTorch training on MPS and CUDA should share one code path.",
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tokenizer.save_pretrained(output_dir)
    return output_dir


@pytest.fixture(scope="session")
def preferred_test_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
