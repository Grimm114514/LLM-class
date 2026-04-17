
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
print("训练完成")
print("=" * 80)

print("\n已生成分词器文件:")
for file in saved_tokenizer_files:
    print(f"- {file.name}")

print("\n测试脚本请分别运行: test_rare_chars.py 与 test_compression_ratio.py")
