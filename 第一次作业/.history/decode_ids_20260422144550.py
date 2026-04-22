from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer


def find_default_tokenizer(base_dir: Path) -> Path:
    """Prefer mixed 16000 tokenizer, then fall back to available ones."""
    candidates = [
        base_dir / "mix16000.json",
        base_dir / "mix6400.json",
        base_dir / "en16000.json",
        base_dir / "en6400.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("未找到 tokenizer 文件，请通过 --tokenizer 指定路径。")


def parse_id_list(raw: str) -> list[int]:
    """Accept formats like '1,2,3' or '1 2 3'."""
    text = raw.strip().replace(",", " ")
    if not text:
        raise ValueError("输入为空，请提供至少一个 ID。")

    parts = text.split()
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError("ID 序列必须是整数，可用空格或逗号分隔。") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="输入一段 Token ID 序列，解码还原文本。"
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="ID 序列，如 '101 202 303' 或 '101,202,303'；不传时交互输入。",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="tokenizer json 路径，如 mix16000.json",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    tokenizer_path = args.tokenizer or find_default_tokenizer(base_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    raw_ids = args.ids
    if raw_ids is None:
        raw_ids = input("请输入要解码的 ID 序列: ").strip()

    id_list = parse_id_list(raw_ids)
    decoded = tokenizer.decode(id_list)

    print("=" * 80)
    print(f"Tokenizer: {tokenizer_path.name}")
    print("Input IDs:")
    print(id_list)
    print("Decoded text:")
    print(decoded)


if __name__ == "__main__":
    main()
