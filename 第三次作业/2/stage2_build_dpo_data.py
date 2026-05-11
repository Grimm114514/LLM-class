import json
from pathlib import Path

DPO_SOURCE_PATH = Path("data/dpo_source.jsonl")
DPO_PREF_PATH = Path("data/dpo_preference.jsonl")
DPO_PROMPT_PATH = Path("prompts/dpo_prompt_template.txt")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not DPO_SOURCE_PATH.exists():
        raise FileNotFoundError("缺少 data/dpo_source.jsonl，请先准备 DPO 原始数据。")

    prompt_template = DPO_PROMPT_PATH.read_text(encoding="utf-8")
    rows = load_jsonl(DPO_SOURCE_PATH)

    out: list[dict] = []
    for row in rows:
        prompt = prompt_template.format(
            instruction=row["instruction"],
            input=row.get("input", ""),
        )
        out.append(
            {
                "prompt": prompt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )

    write_jsonl(DPO_PREF_PATH, out)
    print("[阶段2] DPO 偏好数据已构建：data/dpo_preference.jsonl")


if __name__ == "__main__":
    main()
