import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL_DIR = Path("models/base")
SFT_MODEL_DIR = Path("models/sft")
SFT_DATA_PATH = Path("data/sft_data.jsonl")
SFT_PROMPT_PATH = Path("prompts/sft_prompt_template.txt")
SFT_LOG_SAVE_PATH = Path("outputs/sft/log_history.json")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_sample(row: dict, prompt_template: str) -> str:
    return prompt_template.format(
        instruction=row["instruction"],
        input=row.get("input", ""),
    ) + row["output"]


def main() -> None:
    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError("请先运行 download.py，确保 models/base 存在。")

    if not SFT_DATA_PATH.exists():
        raise FileNotFoundError("缺少 data/sft_data.jsonl，请先准备 SFT 数据。")

    prompt_template = SFT_PROMPT_PATH.read_text(encoding="utf-8")
    rows = load_jsonl(SFT_DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [format_sample(row, prompt_template) for row in rows]
    dataset = Dataset.from_dict({"text": texts})

    def tok(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="outputs/sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()

    SFT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(SFT_MODEL_DIR)
    tokenizer.save_pretrained(SFT_MODEL_DIR)

    SFT_LOG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SFT_LOG_SAVE_PATH.write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[阶段1] SFT 训练完成，模型已保存到 models/sft")
    print("[阶段1] 日志已保存到 outputs/sft/log_history.json")


if __name__ == "__main__":
    main()
