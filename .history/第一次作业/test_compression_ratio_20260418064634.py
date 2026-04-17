from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer


TEST_CASES = [
    "人工智能正在改变世界，The development of AI is fast.",
    "今天北京天气不错，我们一起学习Tokenizer与BPE算法。",
    "这是一个普通句子，用于观察不同词表大小下的压缩率差异。",
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
    return files[:2]


def compute_ratios(tokenizer: Tokenizer) -> tuple[list[float], float]:
    ratios: list[float] = []
    total_chars = 0
    total_tokens = 0

    for text in TEST_CASES:
        enc = tokenizer.encode(text)
        char_count = len(text)
        token_count = len(enc.ids)

        total_chars += char_count
        total_tokens += token_count

        ratio = char_count / token_count if token_count > 0 else 0.0
        ratios.append(ratio)

    avg_ratio = total_chars / total_tokens if total_tokens > 0 else 0.0
    return ratios, avg_ratio


def save_ratio_figure(
    tokenizer_name: str,
    ratios: list[float],
    avg_ratio: float,
    out_path: Path,
) -> None:
    x_labels = [f"S{i}" for i in range(1, len(ratios) + 1)]

    plt.figure(figsize=(8, 5), dpi=140)
    bars = plt.bar(x_labels, ratios)
    plt.axhline(avg_ratio, linestyle="--", linewidth=1.5, label=f"avg={avg_ratio:.3f}")

    for bar, value in zip(bars, ratios):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(f"Compression Ratio (char/token) - {tokenizer_name}")
    plt.xlabel("Test Case")
    plt.ylabel("char/token")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    tokenizer_files = find_tokenizer_files(base_dir)

    print("=" * 80)
    print("压缩率对比测试（输出图片）")
    print("=" * 80)

    image_names = ["00.png", "01.png"]

    for idx, tokenizer_file in enumerate(tokenizer_files):
        tk = Tokenizer.from_file(str(tokenizer_file))
        ratios, avg_ratio = compute_ratios(tk)

        print(f"\n{tokenizer_file.name}")
        for i, value in enumerate(ratios, start=1):
            print(f"  S{i} char/token: {value:.4f}")
        print(f"  AVG char/token: {avg_ratio:.4f}")

        out_path = base_dir / image_names[idx]
        save_ratio_figure(tokenizer_file.name, ratios, avg_ratio, out_path)
        print(f"  已保存图片: {out_path.name}")


if __name__ == "__main__":
    main()
