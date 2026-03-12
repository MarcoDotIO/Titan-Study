from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_train_resume_and_infer(
    repo_root: Path,
    dataset_path: Path,
    tiny_tokenizer_dir: Path,
    preferred_test_device: str,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "outputs"
    train_command = [
        sys.executable,
        str(repo_root / "train.py"),
        "--dataset-path",
        str(dataset_path),
        "--tokenizer",
        str(tiny_tokenizer_dir),
        "--preset",
        "tiny_test",
        "--device",
        preferred_test_device,
        "--batch-size",
        "1",
        "--grad-accum-steps",
        "1",
        "--max-steps",
        "2",
        "--eval-every",
        "1",
        "--save-every",
        "1",
        "--max-docs",
        "512",
        "--max-sequences",
        "8",
        "--out-dir",
        str(out_dir),
    ]
    first_run = run_command(train_command, repo_root)
    assert "training_complete" in first_run.stdout
    latest_checkpoint = out_dir / "latest.pt"
    assert latest_checkpoint.exists()

    resume_command = train_command + ["--resume", str(latest_checkpoint), "--max-steps", "3"]
    second_run = run_command(resume_command, repo_root)
    assert "training_complete" in second_run.stdout

    infer_command = [
        sys.executable,
        str(repo_root / "infer.py"),
        "--checkpoint",
        str(latest_checkpoint),
        "--tokenizer",
        str(tiny_tokenizer_dir),
        "--device",
        preferred_test_device,
        "--prompt",
        "Titans remember",
        "--max-new-tokens",
        "8",
        "--temperature",
        "0.0",
    ]
    infer_run = run_command(infer_command, repo_root)
    assert "Response:" in infer_run.stdout
