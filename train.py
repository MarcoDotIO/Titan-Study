from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
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
from titan_mac.data import StreamingPackedDataset, load_datasets
from titan_mac.device import autocast_context, resolve_device, resolve_dtype, use_grad_scaler
from titan_mac.model import TitansMACLM
from titan_mac.tokenization import load_tokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float


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
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm (0 to disable).")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=None, help="Override WANDB_PROJECT for this run.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional Weights & Biases run name.")
    return parser.parse_args()


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def token_accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    if logits.size(1) < 2:
        return 0, 0
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    predictions = shifted_logits.argmax(dim=-1)
    correct = (predictions == shifted_labels).sum().item()
    total = shifted_labels.numel()
    return int(correct), int(total)


def gradient_l2_norm(model: TitansMACLM) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += grad.pow(2).sum().item()
    return total ** 0.5


def maybe_init_wandb(
    args: argparse.Namespace,
    *,
    model: TitansMACLM,
    out_dir: Path,
):
    if not args.wandb:
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "--wandb was requested but WANDB_API_KEY is not set. Add it to .env or the shell environment."
        )

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - dependency is exercised through integration
        raise RuntimeError("wandb is not installed. Install the project requirements first.") from exc

    project = args.wandb_project or os.getenv("WANDB_PROJECT", "titan-mac")
    entity = os.getenv("WANDB_ENTITY") or None
    run = wandb.init(
        project=project,
        entity=entity,
        name=args.wandb_run_name,
        dir=str(out_dir),
        resume="allow",
    )
    wandb.watch(unwrap_model(model), log="gradients", log_freq=10)
    return run


def evaluate(model: TitansMACLM, loader: DataLoader | None, device: torch.device) -> EvaluationMetrics | None:
    if loader is None:
        return None

    was_training = model.training
    model.eval()
    losses = []
    total_correct = 0
    total_tokens = 0
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, labels=batch, reset_memory=True)
            if output.loss is not None:
                losses.append(output.loss.item())
            correct, total = token_accuracy_counts(output.logits.detach(), batch)
            total_correct += correct
            total_tokens += total
    if was_training:
        model.train()
    if not losses:
        return None
    accuracy = total_correct / max(total_tokens, 1)
    return EvaluationMetrics(loss=sum(losses) / len(losses), accuracy=accuracy)


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
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))
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
    # StreamingPackedDataset has no __len__; treat it as always non-empty.
    # For eager PackedSequenceDataset we can still check.
    if not isinstance(train_dataset, StreamingPackedDataset) and len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty. Increase --max-docs or check the tokenizer.")

    # shuffle=True is incompatible with IterableDataset; streaming datasets handle order via file iteration.
    train_shuffle = not isinstance(train_dataset, StreamingPackedDataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, drop_last=False)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if isinstance(val_dataset, StreamingPackedDataset) or len(val_dataset) > 0
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
    batches_seen = start_step * args.grad_accum_steps
    wandb_run = maybe_init_wandb(args, model=model, out_dir=out_dir)

    try:
        for step in range(start_step, args.max_steps):
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            train_correct = 0
            train_tokens = 0
            for _ in range(args.grad_accum_steps):
                batch = next(train_batches).to(device)
                batches_seen += 1
                with autocast_context(device, dtype):
                    output = model(batch, labels=batch, reset_memory=True)
                    if output.loss is None:
                        raise RuntimeError("Model did not return a loss during training.")
                    loss = output.loss / args.grad_accum_steps
                micro_losses.append(loss.detach().item() * args.grad_accum_steps)
                correct, total = token_accuracy_counts(output.logits.detach(), batch)
                train_correct += correct
                train_tokens += total
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            grad_norm = gradient_l2_norm(unwrap_model(model))

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            finished_step = step + 1
            avg_loss = sum(micro_losses) / len(micro_losses)
            train_accuracy = train_correct / max(train_tokens, 1)
            elapsed = time.time() - started
            tokens_per_step = args.batch_size * model_config.max_seq_len * args.grad_accum_steps
            tokens_per_sec = tokens_per_step / max(elapsed, 1e-6)
            try:
                epoch_size = max(len(train_loader), 1)
                epoch = ((batches_seen - 1) // epoch_size) + 1
                epoch_iteration = ((batches_seen - 1) % epoch_size) + 1
            except TypeError:
                # IterableDataset has no len(); epoch tracking is not meaningful
                epoch_size = None
                epoch = None
                epoch_iteration = None
            learning_rate = scheduler.get_last_lr()[0]
            print(
                f"step={finished_step} "
                f"loss={avg_loss:.4f} "
                f"accuracy={train_accuracy:.4f} "
                f"lr={learning_rate:.6f} "
                f"grad_norm={grad_norm:.4f} "
                f"tokens_per_sec={tokens_per_sec:.2f}"
            )
            started = time.time()

            wandb_metrics = {
                "train/loss": avg_loss,
                "train/accuracy": train_accuracy,
                "train/learning_rate": learning_rate,
                "train/grad_norm": grad_norm,
                "train/tokens_per_sec": tokens_per_sec,
            }
            if epoch is not None:
                wandb_metrics["train/epoch"] = epoch
            if epoch_iteration is not None:
                wandb_metrics["train/epoch_iteration"] = epoch_iteration

            if args.eval_every > 0 and finished_step % args.eval_every == 0:
                eval_metrics = evaluate(model, val_loader, device)
                if eval_metrics is None:
                    print("eval=skipped reason=no_validation_sequences")
                else:
                    perplexity = math.exp(eval_metrics.loss) if eval_metrics.loss < 20 else float("inf")
                    print(
                        f"eval_loss={eval_metrics.loss:.4f} "
                        f"eval_accuracy={eval_metrics.accuracy:.4f} "
                        f"perplexity={perplexity:.4f}"
                    )
                    wandb_metrics.update(
                        {
                            "eval/loss": eval_metrics.loss,
                            "eval/score": eval_metrics.loss,
                            "eval/accuracy": eval_metrics.accuracy,
                        }
                    )
                    if math.isfinite(perplexity):
                        wandb_metrics["eval/perplexity"] = perplexity

            if wandb_run is not None:
                wandb_run.log(wandb_metrics, step=finished_step)

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
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
