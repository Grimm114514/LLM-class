import os

from datasets import load_dataset


MB = 1024 * 1024


def pick_text(example, field_candidates=None, min_len=20):
    if field_candidates:
        for key in field_candidates:
            value = example.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    # 兜底: 选择样本中第一个足够长的字符串字段
    for value in example.values():
        if isinstance(value, str):
            text = value.strip()
            if len(text) >= min_len:
                return text
    return ""


def stream_dataset_to_size(
    dataset_name,
    output_path,
    target_mb,
    split="train",
    field_candidates=None,
):
    target_bytes = int(target_mb * MB)
    written_bytes = 0
    written_samples = 0

    hf_token = os.getenv("HF_TOKEN")

    print(f"\n开始下载: {dataset_name} ({target_mb}MB)")
    print(f"输出文件: {output_path}")

    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            token=hf_token,
        )
    except Exception as exc:
        print(f"切分 {split} 不可用，尝试自动回退。原因: {exc}")
        fallback_splits = ["train", "validation", "test"]
        dataset = None
        for candidate in fallback_splits:
            try:
                dataset = load_dataset(
                    dataset_name,
                    split=candidate,
                    streaming=True,
                    token=hf_token,
                )
                print(f"已回退使用切分: {candidate}")
                break
            except Exception:
                continue
        if dataset is None:
            raise RuntimeError(f"数据集 {dataset_name} 没有可用切分")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset, start=1):
            text = pick_text(example, field_candidates=field_candidates)
            if not text:
                continue

            # 双换行分段，兼顾后续按段处理
            chunk = text + "\n\n"
            chunk_bytes = len(chunk.encode("utf-8"))

            if written_bytes + chunk_bytes > target_bytes:
                remain = target_bytes - written_bytes
                if remain > 0:
                    # 按字节截断并保证 utf-8 安全
                    safe_chunk = chunk.encode("utf-8")[:remain].decode("utf-8", errors="ignore")
                    if safe_chunk:
                        f.write(safe_chunk)
                        written_bytes += len(safe_chunk.encode("utf-8"))
                break

            f.write(chunk)
            written_bytes += chunk_bytes
            written_samples += 1

            if idx % 5000 == 0:
                progress = written_bytes / target_bytes * 100
                print(
                    f"已处理 {idx} 条, 已写入 {written_bytes / MB:.2f}MB "
                    f"({progress:.2f}%)"
                )

    print(
        f"完成: {dataset_name}, 已写入 {written_bytes / MB:.2f}MB, "
        f"样本数约 {written_samples}"
    )


if __name__ == "__main__":
    tasks = [
        {
            "dataset_name": "pleisto/wikipedia-cn-20230720-filtered",
            "output_path": "wiki_zh_600mb.txt",
            "target_mb": 600,
            "split": "train",
            "field_candidates": ["completion", "text", "content"],
        },
        {
            "dataset_name": "shibing624/THUCNews",
            "output_path": "thucnews_100mb.txt",
            "target_mb": 100,
            "split": "train",
            "field_candidates": ["text", "content", "title"],
        },
        {
            "dataset_name": "Skywork/SkyPile-150B",
            "output_path": "skypile_300mb.txt",
            "target_mb": 300,
            "split": "train",
            "field_candidates": ["text", "content", "completion"],
        },
    ]

    for task in tasks:
        stream_dataset_to_size(**task)