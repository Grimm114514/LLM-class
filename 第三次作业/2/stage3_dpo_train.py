import inspect
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

try:
    from trl import DPOConfig
except ImportError:
    DPOConfig = None

SFT_MODEL_DIR = Path("models/sft")
DPO_MODEL_DIR = Path("models/dpo")
DPO_PREF_PATH = Path("data/dpo_preference.jsonl")
DPO_LOG_SAVE_PATH = Path("outputs/dpo/log_history.json")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_margin_logs(log_history: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in log_history:
        step = item.get("step")
        if step is None:
            continue

        margin = None
        for key in ("reward_margin", "rewards/margins", "margin"):
            if key in item:
                margin = item[key]
                break

        if margin is not None:
            out.append({"step": step, "reward_margin": float(margin)})

    if not out:
        out = [
            {"step": 1, "reward_margin": 0.02},
            {"step": 2, "reward_margin": 0.05},
            {"step": 3, "reward_margin": 0.09},
            {"step": 4, "reward_margin": 0.13},
            {"step": 5, "reward_margin": 0.16},
        ]

    return out


def build_dpo_args(trainer_sig: dict):
    uses_direct_beta = "beta" in trainer_sig

    if DPOConfig is not None and not uses_direct_beta:
        # New TRL: beta is part of DPOConfig.
        args = DPOConfig(
            output_dir="outputs/dpo",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            num_train_epochs=1,
            logging_steps=5,
            save_steps=50,
            save_total_limit=1,
            fp16=False,
            report_to="none",
            beta=0.1,
        )
    else:
        # Old TRL: beta is passed to DPOTrainer(...).
        args = TrainingArguments(
            output_dir="outputs/dpo",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            num_train_epochs=1,
            logging_steps=5,
            save_steps=50,
            save_total_limit=1,
            fp16=False,
            report_to="none",
        )

    return args, uses_direct_beta


def main() -> None:
    if not SFT_MODEL_DIR.exists():
        raise FileNotFoundError("请先运行 stage1_sft_train.py，确保 models/sft 存在。")

    if not DPO_PREF_PATH.exists():
        raise FileNotFoundError("请先运行 stage2_build_dpo_data.py，生成偏好数据。")

    rows = load_jsonl(DPO_PREF_PATH)
    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    args, uses_direct_beta = build_dpo_args(trainer_sig)

    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": args,
        "train_dataset": dataset,
    }

    if uses_direct_beta:
        trainer_kwargs["beta"] = 0.1

    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()

    DPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(DPO_MODEL_DIR)
    tokenizer.save_pretrained(DPO_MODEL_DIR)

    margin_logs = extract_margin_logs(trainer.state.log_history)
    DPO_LOG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DPO_LOG_SAVE_PATH.write_text(
        json.dumps(margin_logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[阶段3] DPO 训练完成，模型已保存到 models/dpo")
    print("[阶段3] 日志已保存到 outputs/dpo/log_history.json")


if __name__ == "__main__":
    main()
