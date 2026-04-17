from pathlib import Path

from tokenizers import Tokenizer


TEST_TEXTS = [
    "这个新词qwerty_zzzz和生僻字𪚥在语料里很少见。",
    "罕见组合abc_xyz_123通常不会完整出现在词表里。",
]


def find_tokenizer_files(base_dir: Path) -> list[Path]:
    # 先用你目录里已有命名，再回退到训练脚本的命名。
    candidates = [
        base_dir / "mix6400.json",
        base_dir / "mix16000.json",
        base_dir / "minimind_tokenizer_6400.json",
        base_dir / "minimind_tokenizer_16384.json",
    ]
    files = [p for p in candidates if p.exists()]
    if not files:
        raise FileNotFoundError(
            "未找到可用 tokenizer json。请先运行 train_bpe_compare.py 训练，"
            "或确认 mix6400.json/mix16000.json 存在。"
        )
    return files


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    tokenizer_files = find_tokenizer_files(base_dir)

    print("=" * 80)
    print("生僻字 / OOV 测试")
    print("=" * 80)

    for tokenizer_file in tokenizer_files:
        print(f"\n########## 当前分词器: {tokenizer_file.name} ##########")
        tk = Tokenizer.from_file(str(tokenizer_file))

        for idx, text in enumerate(TEST_TEXTS, start=1):
            enc = tk.encode(text)
            unk_count = enc.tokens.count("<|endoftext|>")

            print("\n" + "-" * 80)
            print(f"样例 {idx}: {text}")
            print(f"Token 数: {len(enc.ids)}")
            print(f"疑似 OOV 回退次数(<|endoftext|>): {unk_count}")
            print("Tokens:")
            print(enc.tokens)


if __name__ == "__main__":
    main()
