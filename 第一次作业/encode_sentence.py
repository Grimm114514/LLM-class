from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer


MODE_TO_FILE = {
    "1": "en6400.json",
    "2": "en16000.json",
    "3": "mix6400.json",
    "4": "mix16000.json",
}

MODE_DESCRIPTION = {
    "1": "纯英文词表 6400",
    "2": "纯英文词表 16000",
    "3": "中英混合词表 6400",
    "4": "中英混合词表 16000",
}


def choose_tokenizer_by_mode(base_dir: Path, mode: str | None) -> Path:
    print("可选模式如下:")
    for key in ("1", "2", "3", "4"):
        print(f"  {key}: {MODE_DESCRIPTION[key]} ({MODE_TO_FILE[key]})")

    selected = mode
    if selected is None:
        selected = input("请输入模式编号(1/2/3/4): ").strip()

    if selected not in MODE_TO_FILE:
        raise ValueError("模式输入无效，请输入 1、2、3 或 4。")

    tokenizer_path = base_dir / MODE_TO_FILE[selected]
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"未找到词表文件: {tokenizer_path.name}")

    print(f"已选择模式 {selected}: {MODE_DESCRIPTION[selected]}")
    return tokenizer_path


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
        "--mode",
        type=str,
        default=None,
        help="词表模式编号：1=en6400, 2=en16000, 3=mix6400, 4=mix16000",
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
    tokenizer_path = args.tokenizer or choose_tokenizer_by_mode(base_dir, args.mode)
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
