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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="输入一句文本，输出对应的 Token ID 序列。"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="要编码的文本；不传时将进入交互输入。",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="tokenizer json 路径，如 mix16000.json",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="同时打印分词后的 token 字符串。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    tokenizer_path = args.tokenizer or find_default_tokenizer(base_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    text = args.text
    if text is None:
        text = input("请输入要编码的句子: ").strip()

    if not text:
        raise ValueError("输入文本为空，无法编码。")

    encoding = tokenizer.encode(text)

    print("=" * 80)
    print(f"Tokenizer: {tokenizer_path.name}")
    print(f"Input: {text}")
    print("Token IDs:")
    print(encoding.ids)
    print(f"Token count: {len(encoding.ids)}")

    if args.show_tokens:
        print("Tokens:")
        print(encoding.tokens)


if __name__ == "__main__":
    main()
