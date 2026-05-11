
import json
from pathlib import Path

def clean_file(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ins = obj.get("instruction", "")
            # 去掉形如 【SFT-001】 或 【DPO-037】 的前缀
            if "】" in ins and ins.startswith("【"):
                ins = ins.split("】", 1)[1].strip()
            obj["instruction"] = ins
            rows.append(obj)

    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

for p in [Path("data/sft_data.jsonl"), Path("data/dpo_source.jsonl")]:
    if p.exists():
        clean_file(p)
        print(f"cleaned: {p}")
    else:
        print(f"missing: {p}")
