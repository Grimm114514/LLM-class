from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = PROJECT_DIR / "checkpoints"
DEFAULT_PROMPT_PATH = PROJECT_DIR / "prompt.txt"
DEFAULT_TOKENIZER_PATH = PROJECT_DIR / "data" / "bpe_tokenizer.json"


class MiniLlama(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, n_heads: int, n_layers: int, seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        return self.lm_head(x)


def pick_latest_checkpoint(ckpt_dir: Path) -> Path:
    checkpoints = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")
    return checkpoints[-1]


def top_k_sample(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        values, indices = torch.topk(logits, k=k)
        probs = torch.softmax(values, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        return int(indices[next_idx].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_text(
    model: MiniLlama,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seq_len: int,
    eos_id: int | None,
    device: str,
) -> str:
    input_ids = tokenizer.encode(prompt).ids
    if not input_ids:
        # fallback to BOS if prompt is empty
        bos_id = tokenizer.token_to_id("[BOS]")
        if bos_id is None:
            raise ValueError("Prompt is empty and tokenizer has no [BOS] token.")
        input_ids = [bos_id]

    generated = list(input_ids)

    for _ in range(max_new_tokens):
        context = generated[-seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_token = top_k_sample(logits, top_k=top_k, temperature=temperature)
        generated.append(next_token)

        if eos_id is not None and next_token == eos_id:
            break

    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate continuation from prompt.txt using trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path (.pt). Default: latest under checkpoints/")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CKPT_DIR), help="Directory containing checkpoints")
    parser.add_argument("--prompt", type=str, default=str(DEFAULT_PROMPT_PATH), help="Prompt text file path")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling; 0 means full vocab")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature; <=0 uses greedy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else pick_latest_checkpoint(Path(args.checkpoint_dir))
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not prompt_text:
        print("[Warn] prompt.txt is empty, generation will start from [BOS].")

    checkpoint = torch.load(ckpt_path, map_location=args.device)

    vocab_size = int(checkpoint["vocab_size"])
    seq_len = int(checkpoint["seq_len"])
    hidden_dim = int(checkpoint["hidden_dim"])
    n_heads = int(checkpoint["n_heads"])
    n_layers = int(checkpoint["n_layers"])
    eos_id = checkpoint.get("eos_id")

    tokenizer_path = Path(checkpoint.get("tokenizer_path", str(DEFAULT_TOKENIZER_PATH)))
    if not tokenizer_path.exists():
        tokenizer_path = DEFAULT_TOKENIZER_PATH
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    model = MiniLlama(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    full_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        seq_len=seq_len,
        eos_id=eos_id,
        device=args.device,
    )

    continuation = full_text[len(prompt_text):] if prompt_text and full_text.startswith(prompt_text) else full_text

    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Prompt file: {prompt_path}")
    print(f"Device: {args.device}")
    print("=" * 60)
    print("[Prompt]")
    print(prompt_text)
    print("\n[Continuation]")
    print(continuation)
    print("=" * 60)


if __name__ == "__main__":
    main()
