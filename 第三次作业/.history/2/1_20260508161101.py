import json
from pathlib import Path

def clean_file(path: Path):
    rows = []
    # 关键：utf-8-sig 可自动处理 BOM
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[跳过] {path} 第{i}行 JSON 无效: {e}")
                continue

            ins = obj.get("instruction", "")
            if ins.startswith("【") and "】" in ins:
                ins = ins.split("】", 1)[1].strip()
            obj["instruction"] = ins
            rows.append(obj)

    # 写回普通 utf-8（不带 BOM）
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[完成] {path} 共处理 {len(rows)} 条")

for p in [Path("data/sft_data.jsonl"), Path("data/dpo_source.jsonl")]:
    if p.exists():
        clean_file(p)
    else:
        print(f"[缺失] {p}")
