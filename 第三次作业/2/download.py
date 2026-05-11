import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# 你可以替换为自己的基座模型
MODEL_ID = "Qwen/Qwen2.5-0.5B"
SAVE_DIR = Path("models/base")

# 国内网络可选：使用镜像站加速下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[阶段0] 开始下载 Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"[阶段0] 开始下载 Model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
    )

    print(f"[阶段0] 保存到本地目录: {SAVE_DIR}")
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print("[阶段0] 完成：基座模型已下载并保存。")


if __name__ == "__main__":
    main()
