import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_CORPUS_PATH = PROJECT_DIR / "traincorpus.txt"
TOKENIZER_PATH = DATA_DIR / "bpe_tokenizer.json"
VOCAB_TXT_PATH = DATA_DIR / "bpe_vocab.txt"

SOURCE_FILES = [
"traincorpus.txt"
]



def _extract_text_from_jsonl_line(line):
    stripped = line.strip()
    if not stripped:
        return ""

    try:
        sample = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if isinstance(sample, dict):
        for key in ("text", "completion", "content"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    if isinstance(sample, str):
        return sample.strip()
    return ""



def _extract_text_from_json_array_file(file_path):
    field_candidates = ("completion", "text", "content")

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            for field in field_candidates:
                marker = f'"{field}":'
                if marker in stripped:
                    raw_value = stripped.split(marker, 1)[1].strip().rstrip(",")
                    if raw_value.startswith('"') and raw_value.endswith('"'):
                        try:
                            text = json.loads(raw_value)
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue
                    break



def build_train_corpus(output_path=TRAIN_CORPUS_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for source in SOURCE_FILES:
        if not source.exists():
            raise FileNotFoundError(f"未找到语料文件: {source}")

    total_samples = 0
    total_chars = 0

    with output_path.open("w", encoding="utf-8") as out:
        for source in SOURCE_FILES:
            print(f"处理语料: {source.name}")

            if source.suffix == ".txt":
                text = source.read_text(encoding="utf-8").strip()
                if text:
                    out.write(text + "\n\n")
                    total_samples += 1
                    total_chars += len(text)

            elif source.suffix == ".jsonl":
                with source.open("r", encoding="utf-8") as f:
                    for line in f:
                        text = _extract_text_from_jsonl_line(line)
                        if text:
                            out.write(text + "\n\n")
                            total_samples += 1
                            total_chars += len(text)

            elif source.suffix == ".json":
                for text in _extract_text_from_json_array_file(source):
                    out.write(text + "\n\n")
                    total_samples += 1
                    total_chars += len(text)

            else:
                raise ValueError(f"不支持的语料格式: {source}")

    print(
        f"合并完成: {output_path.name} | 段落数 {total_samples:,} | 字符数 {total_chars:,}"
    )
    return output_path



def train_bpe(
    corpus_path=TRAIN_CORPUS_PATH,
    tokenizer_path=TOKENIZER_PATH,
    vocab_txt_path=VOCAB_TXT_PATH,
    vocab_size=32000,
    min_frequency=2,
):
    if not corpus_path.exists():
        raise FileNotFoundError(f"未找到训练语料: {corpus_path}")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )

    print("开始训练 BPE 词表...")
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))

    vocab = tokenizer.get_vocab()
    id_to_token = sorted(vocab.items(), key=lambda kv: kv[1])
    with vocab_txt_path.open("w", encoding="utf-8") as f:
        for token, idx in id_to_token:
            f.write(f"{idx}\t{token}\n")

    print(
        f"BPE 训练完成: vocab_size={tokenizer.get_vocab_size()} | "
        f"tokenizer={tokenizer_path.name} | vocab={vocab_txt_path.name}"
    )


if __name__ == "__main__":
    train_bpe()
