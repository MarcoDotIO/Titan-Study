from __future__ import annotations

import argparse
import itertools
import math
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from titan_mac.checkpoint import (
    load_checkpoint,
    model_config_from_checkpoint,
    save_checkpoint,
    unwrap_model,
)
from titan_mac.config import build_model_config
from titan_mac.data import load_datasets
from titan_mac.device import autocast_context, resolve_device, resolve_dtype, use_grad_scaler
from titan_mac.model import TitansMACLM
from titan_mac.tokenization import load_tokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Titans MAC language model on Dolma.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float32", "float16"],
    )
    parser.add_argument(
        "--dataset-path",
        default=".dataset/books-0000.json.gz",
        help="Path to a Dolma gzip NDJSON shard or a directory of gzip shards.",
    )
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path or model id.")
    parser.add_argument("--preset", default="paper_170m", choices=["tiny_test", "paper_170m"])
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--segment-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    return parser.parse_args()


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def evaluate(model: TitansMACLM, loader: DataLoader | None, device: torch.device) -> float | None:
    if loader is None:
        return None

    was_training = model.training
    model.eval()
    losses = []
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, labels=batch, reset_memory=True)
            if output.loss is not None:
                losses.append(output.loss.item())
    if was_training:
        model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def maybe_compile(model: TitansMACLM, enabled: bool, device: torch.device) -> TitansMACLM:
    if not enabled:
        return model
    if device.type == "mps":
        print("Skipping torch.compile on MPS; using eager mode.")
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile is unavailable in this PyTorch build; using eager mode.")
        return model
    try:
        return torch.compile(model)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"torch.compile failed ({exc}); using eager mode.")
        return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint = load_checkpoint(args.resume, device="cpu") if args.resume else None
    tokenizer_ref = args.tokenizer or (checkpoint["tokenizer_ref"] if checkpoint else None)
    if tokenizer_ref is None:
        raise ValueError("A tokenizer must be provided unless resuming from a checkpoint.")
    tokenizer = load_tokenizer(tokenizer_ref)

    if checkpoint is not None:
        model_config = model_config_from_checkpoint(checkpoint)
    else:
        model_config = build_model_config(
            args.preset,
            tokenizer.vocab_size,
            seq_len_override=args.seq_len,
            segment_len_override=args.segment_len,
        )

    train_dataset, val_dataset = load_datasets(
        args.dataset_path,
        tokenizer,
        seq_len=model_config.max_seq_len,
        max_docs=args.max_docs,
        max_sequences=args.max_sequences,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty. Increase --max-docs or check the tokenizer.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if len(val_dataset) > 0
        else None
    )

    model = TitansMACLM(model_config).to(device)
    model = maybe_compile(model, args.compile, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.max_steps, 1))
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if use_grad_scaler(device, dtype)
        else None
    )

    start_step = 0
    if checkpoint is not None:
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_step = checkpoint.get("step", 0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    train_batches = cycle_loader(train_loader)
    started = time.time()

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum_steps):
            batch = next(train_batches).to(device)
            with autocast_context(device, dtype):
                output = model(batch, labels=batch, reset_memory=True)
                if output.loss is None:
                    raise RuntimeError("Model did not return a loss during training.")
                loss = output.loss / args.grad_accum_steps
            micro_losses.append(loss.detach().item() * args.grad_accum_steps)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        finished_step = step + 1
        avg_loss = sum(micro_losses) / len(micro_losses)
        elapsed = time.time() - started
        tokens_per_step = args.batch_size * model_config.max_seq_len * args.grad_accum_steps
        print(
            f"step={finished_step} "
            f"loss={avg_loss:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f} "
            f"tokens_per_sec={tokens_per_step / max(elapsed, 1e-6):.2f}"
        )
        started = time.time()

        if args.eval_every > 0 and finished_step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            if val_loss is None:
                print("eval=skipped reason=no_validation_sequences")
            else:
                perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
                print(f"eval_loss={val_loss:.4f} perplexity={perplexity:.4f}")

        if args.save_every > 0 and finished_step % args.save_every == 0:
            step_checkpoint = out_dir / f"step_{finished_step:06d}.pt"
            latest_checkpoint = out_dir / "latest.pt"
            save_checkpoint(
                step_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=model_config,
                train_args=vars(args),
                tokenizer_ref=tokenizer_ref,
                step=finished_step,
            )
            save_checkpoint(
                latest_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=model_config,
                train_args=vars(args),
                tokenizer_ref=tokenizer_ref,
                step=finished_step,
            )
            print(f"checkpoint={latest_checkpoint}")

    final_checkpoint = out_dir / "latest.pt"
    save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        model_config=model_config,
        train_args=vars(args),
        tokenizer_ref=tokenizer_ref,
        step=args.max_steps,
    )
    print(f"training_complete checkpoint={final_checkpoint}")


if __name__ == "__main__":
    main()
