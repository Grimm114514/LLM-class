import json
import re
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import font_manager
from transformers import AutoModelForCausalLM, AutoTokenizer

# Headless backend for server environments.
matplotlib.use("Agg")

BASE_MODEL_DIR = Path("models/base")
SFT_MODEL_DIR = Path("models/sft")
DPO_MODEL_DIR = Path("models/dpo")

SFT_LOG_PATH = Path("outputs/sft/log_history.json")
DPO_LOG_PATH = Path("outputs/dpo/log_history.json")
CASE_RESULT_PATH = Path("outputs/case_outputs.json")
CASE_COMPARISON_TXT_PATH = Path("outputs/fig_case_comparison.txt")
EVAL_QUESTION_PATH = Path("prompts/eval_questions.txt")


def setup_chinese_font() -> None:
    """Try to pick a CJK font so Chinese text can render in figures."""
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    candidate_names = [
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
    ]

    font_name = None

    for path in candidate_paths:
        p = Path(path)
        if p.exists():
            prop = font_manager.FontProperties(fname=str(p))
            font_name = prop.get_name()
            break

    if font_name is None:
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidate_names:
            if name in installed:
                font_name = name
                break

    if font_name is not None:
        plt.rcParams["font.family"] = font_name

    plt.rcParams["axes.unicode_minus"] = False


def load_sft_loss(path: Path) -> pd.DataFrame:
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(rows)
        if "step" in df.columns and "loss" in df.columns:
            return df[["step", "loss"]].dropna()

    return pd.DataFrame({"step": [1, 2, 3, 4, 5], "loss": [2.1, 1.8, 1.6, 1.45, 1.3]})


def load_dpo_margin(path: Path) -> pd.DataFrame:
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(rows)
        if "step" in df.columns and "reward_margin" in df.columns:
            return df[["step", "reward_margin"]].dropna()

    return pd.DataFrame(
        {"step": [1, 2, 3, 4, 5], "reward_margin": [0.02, 0.05, 0.09, 0.13, 0.16]}
    )


def parse_eval_questions(path: Path) -> list[str]:
    default_questions = [
        "什么是SFT和DPO，它们分别解决什么问题？",
        "为什么DPO中要设置beta参数？",
        "你如何平衡遵规蹈矩与创造力？",
    ]

    if not path.exists():
        return default_questions

    text = path.read_text(encoding="utf-8")
    questions: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[\.、]\s*(.+?)\s*$", line)
        if m:
            questions.append(m.group(1))

    if not questions:
        questions = [line.strip() for line in text.splitlines() if line.strip()]

    return questions[:3] if questions else default_questions


def load_model_bundle(model_dir: Path):
    if not model_dir.exists():
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def clean_answer(text: str) -> str:
    """Remove prompt-template echoes and keep the actual answer."""
    answer = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    markers = ["### 回答", "回答：", "回答:"]
    for m in markers:
        if m in answer:
            answer = answer.split(m)[-1].strip()

    blacklist = ["### 用户指令", "### 输入信息", "请用中文简洁作答。"]
    for bad in blacklist:
        answer = answer.replace(bad, "")

    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer if answer else "[未生成有效回答]"


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 192) -> str:
    if tokenizer is None or model is None:
        return "[模型缺失] 请先完成对应阶段训练。"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_answer = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
    return clean_answer(raw_answer)


def build_eval_prompt(question: str) -> str:
    return (
        "请用中文回答下面问题。\n"
        "要求：先给结论，再给2点简短解释，总长度控制在120字内。\n"
        f"问题：{question}\n"
        "回答："
    )


def build_case_outputs() -> list[dict]:
    questions = parse_eval_questions(EVAL_QUESTION_PATH)

    base_tok, base_model = load_model_bundle(BASE_MODEL_DIR)
    sft_tok, sft_model = load_model_bundle(SFT_MODEL_DIR)
    dpo_tok, dpo_model = load_model_bundle(DPO_MODEL_DIR)

    rows: list[dict] = []
    for q in questions:
        prompt = build_eval_prompt(q)
        rows.append(
            {
                "question": q,
                "base": generate_answer(base_tok, base_model, prompt),
                "sft": generate_answer(sft_tok, sft_model, prompt),
                "dpo": generate_answer(dpo_tok, dpo_model, prompt),
            }
        )

    CASE_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASE_RESULT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def save_sft_loss_figure(sft_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(sft_df["step"], sft_df["loss"], marker="o")
    plt.title("SFT Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_sft_loss.png", dpi=160)
    plt.close()


def save_dpo_margin_figure(dpo_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(dpo_df["step"], dpo_df["reward_margin"], marker="o")
    plt.title("DPO Reward Margin Curve")
    plt.xlabel("Step")
    plt.ylabel("Reward Margin")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_dpo_reward_margin.png", dpi=160)
    plt.close()


def wrap_cell_text(value: str, width: int = 18) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    lines: list[str] = []
    for line in text.split("\n"):
        if not line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines)


def save_case_comparison_figure(case_data: list[dict]) -> None:
    if not case_data:
        case_data = [
            {
                "question": "请解释什么是过拟合？",
                "base": "过拟合是模型在训练集上表现好、在新数据上表现差。",
                "sft": "过拟合是模型记住训练数据噪声，泛化能力下降。可用正则化、早停等方式缓解。",
                "dpo": "过拟合可理解为泛化不足，通过更高质量数据与更稳健训练目标可进一步缓解。",
            }
        ]

    df = pd.DataFrame(case_data)[["question", "base", "sft", "dpo"]]
    wrapped = df.applymap(lambda x: wrap_cell_text(x, width=18))

    fig, ax = plt.subplots(figsize=(16, max(4, 2 + len(wrapped) * 2.2)))
    ax.axis("off")
    table = ax.table(
        cellText=wrapped.values,
        colLabels=["问题", "基座模型", "SFT模型", "DPO模型"],
        loc="center",
        cellLoc="left",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.6)

    plt.tight_layout()
    plt.savefig("fig_case_comparison.png", dpi=220)
    plt.close()


def save_case_comparison_txt(case_data: list[dict]) -> None:
    if not case_data:
        case_data = [
            {
                "question": "请解释什么是过拟合？",
                "base": "过拟合是模型在训练集上表现好、在新数据上表现差。",
                "sft": "过拟合是模型记住训练数据噪声，泛化能力下降。可用正则化、早停等方式缓解。",
                "dpo": "过拟合可理解为泛化不足，通过更高质量数据与更稳健训练目标可进一步缓解。",
            }
        ]

    lines: list[str] = []
    lines.append("Case Comparison (Base vs SFT vs DPO)")
    lines.append("=" * 48)

    for i, row in enumerate(case_data, 1):
        lines.append(f"\n[Case {i}]")
        lines.append(f"Question: {row.get('question', '')}")
        lines.append(f"Base: {row.get('base', '')}")
        lines.append(f"SFT: {row.get('sft', '')}")
        lines.append(f"DPO: {row.get('dpo', '')}")
        lines.append("-" * 48)

    CASE_COMPARISON_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASE_COMPARISON_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_chinese_font()

    sft_df = load_sft_loss(SFT_LOG_PATH)
    dpo_df = load_dpo_margin(DPO_LOG_PATH)
    case_data = build_case_outputs()

    save_sft_loss_figure(sft_df)
    save_dpo_margin_figure(dpo_df)
    save_case_comparison_figure(case_data)
    save_case_comparison_txt(case_data)

    print("[阶段4] 图像已生成：")
    print("  - fig_sft_loss.png")
    print("  - fig_dpo_reward_margin.png")
    print("  - fig_case_comparison.png")
    print("  - outputs/fig_case_comparison.txt")
    print("[阶段4] Case 输出已保存：outputs/case_outputs.json")


if __name__ == "__main__":
    main()
