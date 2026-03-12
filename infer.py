from __future__ import annotations

import argparse
import os

import torch

from titan_mac.checkpoint import load_checkpoint, model_config_from_checkpoint, unwrap_model
from titan_mac.device import resolve_device
from titan_mac.model import ModelState, TitansMACLM
from titan_mac.tokenization import load_tokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a trained Titans MAC checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--reset-memory", action="store_true")
    return parser.parse_args()


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0:
        top_values, _ = torch.topk(filtered, min(top_k, filtered.size(-1)))
        threshold = top_values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(filtered, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        filtered = filtered.masked_fill(mask, float("-inf"))
    return filtered


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    logits = filter_logits(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model: TitansMACLM,
    tokenizer,
    prompt: str,
    *,
    device: torch.device,
    state: ModelState | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    reset_memory: bool,
) -> tuple[str, ModelState]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if prompt_ids.numel() == 0:
        start_token = tokenizer.bos_token_id or tokenizer.eos_token_id
        prompt_ids = torch.tensor([[start_token]], dtype=torch.long)
    prompt_ids = prompt_ids.to(device)

    with torch.enable_grad():
        output = model(
            prompt_ids,
            memory_state=state,
            reset_memory=reset_memory or state is None,
        )
    state = output.state
    logits = output.logits[:, -1, :]

    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_token = next_token.to(device)
        token_id = int(next_token.detach().cpu().item())
        generated_ids.append(token_id)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        with torch.enable_grad():
            output = model(next_token.to(device), memory_state=state, reset_memory=False)
        state = output.state
        logits = output.logits[:, -1, :]

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not decoded:
        decoded = "<empty>"
    return decoded, state


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    tokenizer_ref = args.tokenizer or checkpoint.get("tokenizer_ref")
    if tokenizer_ref is None:
        raise ValueError("Tokenizer must be passed explicitly or present in the checkpoint.")

    tokenizer = load_tokenizer(tokenizer_ref)
    model_config = model_config_from_checkpoint(checkpoint)
    model = TitansMACLM(model_config)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    state: ModelState | None = None

    def run_prompt(prompt_text: str) -> None:
        nonlocal state
        if args.reset_memory:
            state = None
        response, state = generate(
            model,
            tokenizer,
            prompt_text,
            device=device,
            state=state,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            reset_memory=args.reset_memory,
        )
        print(f"Response: {response}")

    if args.interactive:
        print("Entering interactive mode. Type /reset to clear memory or /exit to quit.")
        while True:
            prompt = input("> ").strip()
            if not prompt:
                continue
            if prompt == "/exit":
                break
            if prompt == "/reset":
                state = None
                print("Memory reset.")
                continue
            run_prompt(prompt)
        return

    if args.prompt is None:
        raise ValueError("--prompt is required unless --interactive is set.")
    run_prompt(args.prompt)


if __name__ == "__main__":
    main()
