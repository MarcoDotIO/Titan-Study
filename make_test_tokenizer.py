from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny local tokenizer for smoke tests.")
    parser.add_argument(
        "--out-dir",
        default="local_test_tokenizer",
        help="Directory where the tokenizer files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    corpus = [
        "Titans learn to memorize books and documents.",
        "Dolma contains long documents from Project Gutenberg.",
        "Memory as Context keeps a persistent prefix and retrieved memory tokens.",
        "PyTorch training on MPS and CUDA should share one code path.",
        "This tokenizer is only for local smoke tests.",
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
    print(f"Wrote tiny smoke-test tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
