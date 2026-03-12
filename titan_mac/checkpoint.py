from __future__ import annotations

from pathlib import Path

import torch

from .config import ModelConfig


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    model_config: ModelConfig,
    train_args: dict,
    tokenizer_ref: str,
    step: int,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    base_model = unwrap_model(model)
    payload = {
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "model_config": model_config.to_dict(),
        "train_args": train_args,
        "tokenizer_ref": tokenizer_ref,
        "step": step,
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    return torch.load(Path(path), map_location=device)


def model_config_from_checkpoint(checkpoint: dict) -> ModelConfig:
    return ModelConfig.from_dict(checkpoint["model_config"])
