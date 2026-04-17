"""
课程作业一：数据预处理与 Tokenizer 分词器训练

本脚本按作业要求实现：
1. 基于 tokenizers 库，从 mixed_train.txt 训练 BPE 分词器。
2. 使用基础分词预处理（Whitespace + Punctuation）。
3. 增加特殊 Token：<|endoftext|>、<|im_start|>、<|im_end|>。
4. 一次性训练两套词表：6400 与 16384，并保存 json 文件。
5. 给出 3 个测试样例，展示：Token IDs、Tokens、Decode 还原文本。
6. 打印 Token 数量和简单压缩率指标，便于写报告中的结果分析。
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.trainers import BpeTrainer


# ------------------------
# 0) 基础配置
# ------------------------
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
VOCAB_SIZES = [6400, 16384]

# 3 个测试样例：中英混合、偏中文、包含 OOV 场景
TEST_CASES = [
    "人工智能正在改变世界，The development of AI is fast.",
    "今天北京天气不错，我们一起学习Tokenizer与BPE算法。",
    "这个新词qwerty_zzzz和生僻字𪚥在语料里很少见。",
]


# Windows 终端下尽量保证 UTF-8 输出，减少中文显示异常。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


base_dir = Path(__file__).resolve().parent
corpus_path = base_dir /"mixed_train.txt"

if not corpus_path.exists():
    raise FileNotFoundError(
        f"未找到语料文件：{corpus_path}\n"
        "请确认 mixed_train.txt 路径正确。"
    )

print("=" * 80)
print("开始训练 BPE Tokenizer（分词预处理 + 双词表设置）")
print(f"语料文件: {corpus_path}")
print("=" * 80)


saved_tokenizer_files: list[Path] = []


# ------------------------
# 1) 训练 6400 / 16384 两套词表
# ------------------------
for vocab_size in VOCAB_SIZES:
    print(f"\n[训练中] vocab_size = {vocab_size}")

    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    # 分词版本：先按空白切分，再把标点拆开，输出更接近“可读词片”。
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    save_file = base_dir / f"minimind_tokenizer_{vocab_size}.json"
    tokenizer.save(str(save_file))
    saved_tokenizer_files.append(save_file)

    print(f"[完成] 已保存: {save_file.name}")


print("\n" + "=" * 80)
print("训练完成，开始做压缩率与编码效果对比")
print("=" * 80)


# ------------------------
# 2) 加载并测试两个 tokenizer
# ------------------------
for tokenizer_file in saved_tokenizer_files:
    print(f"\n\n########## 当前分词器: {tokenizer_file.name} ##########")
    tk = Tokenizer.from_file(str(tokenizer_file))

    total_chars = 0
    total_tokens = 0

    for idx, text in enumerate(TEST_CASES, start=1):
        enc = tk.encode(text)
        decoded_text = tk.decode(enc.ids, skip_special_tokens=False)

        token_count = len(enc.ids)
        char_count = len(text)
        total_tokens += token_count
        total_chars += char_count

        compression_ratio = char_count / token_count if token_count > 0 else 0.0
        unk_count = enc.tokens.count("<|endoftext|>")

        print("\n" + "-" * 80)
        print(f"样例 {idx}: {text}")
        print(f"字符数: {char_count}")
        print(f"Token 数: {token_count}")
        print(f"字符/Token 压缩率: {compression_ratio:.4f}")
        print(f"疑似 OOV 回退次数(<|endoftext|>): {unk_count}")
        print("Token IDs:")
        print(enc.ids)
        print("Tokens:")
        print(enc.tokens)
        print("Decode 还原文本:")
        print(decoded_text)

    avg_ratio = total_chars / total_tokens if total_tokens > 0 else 0.0
    print("\n" + "*" * 80)
    print(f"{tokenizer_file.name} 的整体统计")
    print(f"总字符数: {total_chars}")
    print(f"总 Token 数: {total_tokens}")
    print(f"平均字符/Token 压缩率: {avg_ratio:.4f}")
    print("*" * 80)


print("\n全部流程结束。你可以将以上输出直接截图用于作业报告的结果模块。")
