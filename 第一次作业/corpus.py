import argparse
import random
from pathlib import Path


def read_non_empty_lines(file_path: Path) -> list[str]:
    """Read non-empty lines and normalize to one trailing newline per line."""
    lines: list[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            # Normalize common noisy chars from crawled corpora.
            text = line.replace("\u00A0", " ").strip()
            if text.startswith(("·", "•", "●", "▪")):
                text = text[1:].lstrip()
            if text:
                lines.append(text + "\n")
    return lines


def sample_lines(lines: list[str], target_size: int, rng: random.Random) -> list[str]:
    """Randomly sample without replacement; if data is small, return all."""
    if target_size <= 0:
        return []
    if len(lines) <= target_size:
        return lines
    return rng.sample(lines, target_size)


def build_mixed_corpus(
    en_path: Path,
    zh_path: Path,
    output_path: Path,
    en_size: int,
    zh_size: int,
    seed: int,
) -> None:
    if not en_path.exists():
        raise FileNotFoundError(f"英文语料不存在: {en_path}")
    if not zh_path.exists():
        raise FileNotFoundError(f"中文语料不存在: {zh_path}")

    rng = random.Random(seed)

    en_lines = read_non_empty_lines(en_path)
    zh_lines = read_non_empty_lines(zh_path)

    en_sample = sample_lines(en_lines, en_size, rng)
    zh_sample = sample_lines(zh_lines, zh_size, rng)

    mixed_data = en_sample + zh_sample
    rng.shuffle(mixed_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(mixed_data)

    print(
        "混合完成: "
        f"英文 {len(en_sample)} 行, 中文 {len(zh_sample)} 行, "
        f"总计 {len(mixed_data)} 行 -> {output_path}"
    )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    edata_dir = base_dir / "edata"

    parser = argparse.ArgumentParser(description="从 edata 中英文本地语料构建混合训练集")
    parser.add_argument(
        "--en-path",
        type=Path,
        default=edata_dir / "news-commentary.en",
        help="英文语料路径",
    )
    parser.add_argument(
        "--zh-path",
        type=Path,
        default=edata_dir / "corpus_cleaned_keep_punc.txt",
        help="中文语料路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "mixed_train.txt",
        help="输出文件路径",
    )
    parser.add_argument("--en-size", type=int, default=50_000, help="英文采样行数")
    parser.add_argument("--zh-size", type=int, default=10_000, help="中文采样行数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_mixed_corpus(
        en_path=args.en_path,
        zh_path=args.zh_path,
        output_path=args.output,
        en_size=args.en_size,
        zh_size=args.zh_size,
        seed=args.seed,
    )