# Titans MAC Training Scaffold

This repository is a PyTorch scaffold for training a paper-shaped Titans Memory-as-Context (MAC) language model on a local Dolma `v1.6-sample` shard. It is designed for two workflows:

1. local smoke tests on Apple Silicon with `mps`, and
2. larger single-device runs on a Linux H100 box with CUDA 12.8.

The implementation follows the paper at the architecture level, but it uses a simpler sequential fast-weight update backend instead of the paper's tensorized scan-based training path. That keeps the code understandable and testable while preserving the online associative-memory update rule.

## Layout

- `train.py`: training CLI
- `infer.py`: checkpoint loading and text generation CLI
- `titan_mac/`: shared package code for configs, data loading, device selection, model code, and checkpointing
- `tests/test_unit.py`: unit coverage for parsing, packing, devices, model stability, and checkpoints
- `tests/test_e2e.py`: end-to-end smoke test using the real local Dolma shard and a tiny local tokenizer fixture

## Step 0: Local Conda Environment

On this Mac, Conda is initialized from `~/.bash_profile`, not `~/.bashrc`.

```bash
source ~/.bash_profile
conda create -n titan-study python=3.11 -y
conda activate titan-study
pip install -r requirements.txt
```

## Install on an H100 Server

```bash
conda create -n titan-study python=3.11 -y
conda activate titan-study
pip install -r requirements.txt
```

`requirements.txt` includes the official PyTorch CUDA 12.8 wheel index, so Linux installs resolve to `torch==2.10.0+cu128`.

## Dataset Expectations

The training loader expects a Dolma gzip NDJSON shard where each record contains at least:

- `id`
- `text`

The local sample shard used by tests is:

```text
.dataset/books-0000.json.gz
```

Records are split by stable hash of `id` into a `99/1` train/validation partition. Documents are tokenized without truncation, joined with EOS separators, and packed into fixed-length windows.

## Tokenizer Requirements

Real training should use a paper-aligned Llama 2 tokenizer, passed either as a local path or a Hugging Face model id:

```bash
python train.py --tokenizer meta-llama/Llama-2-7b-hf
```

If the tokenizer is gated on Hugging Face, authenticate first or point `--tokenizer` at a local tokenizer directory.

Tests do not depend on gated assets. They create a tiny local tokenizer fixture at runtime.

## Training

### Local smoke run on Apple Silicon

```bash
python train.py \
  --preset tiny_test \
  --device mps \
  --dataset-path .dataset/books-0000.json.gz \
  --tokenizer /path/to/local/tokenizer \
  --max-docs 512 \
  --max-sequences 8 \
  --max-steps 2 \
  --eval-every 1 \
  --save-every 1 \
  --out-dir outputs/tiny
```

### Approximate 170M paper-shaped preset

```bash
python train.py \
  --preset paper_170m \
  --device cuda \
  --dataset-path .dataset/books-0000.json.gz \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 4e-4 \
  --weight-decay 0.1 \
  --max-steps 1000 \
  --eval-every 50 \
  --save-every 50 \
  --out-dir outputs/paper_170m
```

### Resume from a checkpoint

```bash
python train.py \
  --resume outputs/paper_170m/latest.pt \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --device cuda \
  --max-steps 2000 \
  --out-dir outputs/paper_170m
```

Checkpoints store:

- model weights
- optimizer state
- scheduler state
- grad scaler state
- model config
- train CLI args
- tokenizer reference
- global step

## Inference

### Single prompt

```bash
python infer.py \
  --checkpoint outputs/paper_170m/latest.pt \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --device cuda \
  --prompt "Summarize the memory architecture." \
  --max-new-tokens 64
```

### Interactive mode

```bash
python infer.py \
  --checkpoint outputs/paper_170m/latest.pt \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --device mps \
  --interactive
```

Interactive mode preserves the model's fast memory across turns. Use `/reset` to clear memory, or pass `--reset-memory` to clear it before each prompt automatically.

## Model Notes

- `paper_170m` preset:
  - `12` layers
  - `d_model=768`
  - `n_heads=12`
  - `ffn_hidden=3072`
  - `memory_hidden=1536`
  - `memory_depth=2`
  - `segment_len=256`
  - `memory_tokens=64`
  - `persistent_tokens=8`
  - `max_seq_len=4096`
- Each MAC block uses:
  - residual connections
  - SiLU activations
  - L2-normalized queries and keys
  - depthwise-separable convolutions after Q/K/V projections
  - associative memory updates with learned `alpha`, `eta`, and `theta`
- Fast weights reset at the start of each training sample and persist across inference turns unless reset.

## Running Tests

Activate the local Conda environment first:

```bash
source ~/.bash_profile
conda activate titan-study
```

Run the full suite:

```bash
pytest -q
```

Run only the unit tests:

```bash
pytest tests/test_unit.py -q
```

Run only the end-to-end smoke test:

```bash
pytest tests/test_e2e.py -q
```

### What the tests cover

`tests/test_unit.py` checks:

- Dolma record parsing from the real local shard shape
- deterministic document splitting
- fixed-length packing behavior
- device resolution
- finite forward/backward passes through the MAC model
- checkpoint save/load round-trips

`tests/test_e2e.py` checks:

- uses the real local Dolma shard
- builds a tiny local tokenizer fixture
- trains a `tiny_test` checkpoint for 2 steps
- resumes training from that checkpoint
- runs `infer.py` against the saved checkpoint

On Apple Silicon, the smoke test prefers `mps`; otherwise it falls back to `cpu`.
