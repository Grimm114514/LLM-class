import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

# ==========================================
# 1. 超参数配置 (完全对齐你的实验设计)
# ==========================================
SEQ_LEN = 512          # 截断长度 (Context Window)
HIDDEN_DIM = 512       # 隐藏层维度
N_HEADS = 8            # 注意力头数
N_LAYERS = 6           # 网络层数
BATCH_SIZE = 32        # 批次大小
LEARNING_RATE = 1e-4   # 峰值学习率
EPOCHS = 3             # 训练轮数 (可根据总数据量调整)
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_CORPUS_PATH = Path(__file__).resolve().parent / "traincorpus.txt"
BPE_TOKENIZER_PATH = DATA_DIR / "bpe_tokenizer.json"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _extract_text_from_json_array_file(file_path):
    """
    流式读取 json 数组文件，提取 completion/text/content 字段。
    """
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


def _extract_text_from_jsonl_line(line):
    """
    解析一行 jsonl；若不是合法 json，则退化为原始文本。
    """
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


class BPETokenizerWrapper:
    """
    使用已训练好的 Hugging Face tokenizers BPE 模型。
    """

    def __init__(self, tokenizer_path):
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"未找到 BPE tokenizer 文件: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")

        if self.pad_id is None:
            self.pad_id = 0
        if self.unk_id is None:
            self.unk_id = 1

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids


def load_token_ids_from_train_corpus(corpus_path, tokenizer):
    """
    从 traincorpus.txt 逐行编码，避免一次性构建超大字符串。
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"未找到训练语料: {corpus_path}")

    token_ids = []
    line_count = 0

    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            ids = tokenizer.encode(text)
            if not ids:
                continue
            token_ids.extend(ids)
            if tokenizer.eos_id is not None:
                token_ids.append(tokenizer.eos_id)
            line_count += 1

    if not token_ids:
        raise RuntimeError("读取完成，但未提取到任何可训练 token")

    print(f"语料加载完成: 样本行数 {line_count:,} | token 总数 {len(token_ids):,}")
    return token_ids


class LanguageModelingDataset(Dataset):
    """
    将连续 token 序列切分为固定长度的自回归训练样本。
    """

    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        if len(self.token_ids) < seq_len + 1:
            raise ValueError("token 数量不足，无法构建训练样本")

    def __len__(self):
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.token_ids[start:end]
        input_ids = chunk[:-1]
        labels = chunk[1:]
        return input_ids, labels


class MiniLlama(nn.Module):
    """
    课程作业可用的简化版 Decoder-Only Transformer。
    """

    def __init__(self, vocab_size, hidden_dim, n_heads, n_layers, seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# ==========================================
# 2. 训练函数定义
# ==========================================
def train_mini_llama(model, train_dataloader, vocab_size, pad_id=0, device="cuda"):
    """
    自回归语言模型标准的预训练主循环
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("未检测到 CUDA，自动切换到 CPU 训练。")
        device = "cpu"

    model = model.to(device)
    model.train()

    # 使用 AdamW 优化器，大模型训练的标配 (加入少量的 weight_decay 防止过拟合)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 计算总的训练步数，用于余弦退火学习率调度
    total_steps = len(train_dataloader) * EPOCHS
    
    # 初始化 Cosine Annealing 调度器
    # T_max 设为总步数，意味着学习率会在整个训练周期内平滑下降到 0 (或 eta_min)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 初始化混合精度 Scaler (如果是 bfloat16，理论上不需要 scaler，但为了兼容性保留)
    use_cuda = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=False) # bfloat16 下通常设为 False 即可安全运行

    # 交叉熵损失函数 (忽略 pad_token 的 loss 计算，假设 pad_token_id = 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    print(f"🚀 开始预训练 | 总 Epoch: {EPOCHS} | 总 Steps: {total_steps}")
    step_losses = []
    epoch_losses = []

    for epoch in range(EPOCHS):
        # 使用 tqdm 包装 dataloader 以显示进度条
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        epoch_loss = 0.0

        for step, batch in enumerate(progress_bar):
            # 假设 dataloader 返回的是 (input_ids, labels)
            # 对于自回归任务，labels 通常是 input_ids 向右平移一位
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 开启自动混合精度上下文 (现代架构极力推荐使用 bfloat16 防止溢出)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                # 前向传播：模型输出的 logits 形状通常为 [batch_size, seq_len, vocab_size]
                logits = model(input_ids)
                
                # 展平预测值和标签以计算 Loss
                # logits: [batch_size * seq_len, vocab_size]
                # labels: [batch_size * seq_len]
                loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

            # 反向传播 (使用 scaler 处理，防止 FP16 下的梯度下溢，BF16下等价于 loss.backward())
            scaler.scale(loss).backward()

            # 梯度裁剪 (Gradient Clipping) - 大模型训练极为关键的步骤，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 优化器步进
            scaler.step(optimizer)
            scaler.update()

            # 学习率调度器步进 (注意：CosineAnnealing 是按 step 更新，不是按 epoch)
            scheduler.step()

            # 记录并打印当前信息
            epoch_loss += loss.item()
            step_losses.append(loss.item())
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})

        # 每个 Epoch 结束打印平均 Loss
        avg_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        print(f"✅ Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.4f} | Perplexity (困惑度): {math.exp(avg_loss):.4f}")

    print("🎉 预训练结束！")
    return model, step_losses, epoch_losses


def plot_training_loss(step_losses, epoch_losses, save_path):
    """
    绘制并保存训练损失曲线。
    """
    if not step_losses:
        print("未记录到 step loss，跳过绘图。")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(step_losses, label="Step Loss", linewidth=1.0, alpha=0.8)

    if epoch_losses:
        steps_per_epoch = len(step_losses) // len(epoch_losses)
        epoch_x = [steps_per_epoch * (i + 1) - 1 for i in range(len(epoch_losses))]
        plt.plot(epoch_x, epoch_losses, marker="o", linewidth=2.0, label="Epoch Avg Loss")

    plt.title("Training Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练损失曲线已保存到: {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前训练设备: {device}")

    tokenizer = BPETokenizerWrapper(BPE_TOKENIZER_PATH)
    token_ids = load_token_ids_from_train_corpus(TRAIN_CORPUS_PATH, tokenizer)
    vocab_size = tokenizer.vocab_size

    dataset = LanguageModelingDataset(token_ids, seq_len=SEQ_LEN)
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    model = MiniLlama(
        vocab_size=vocab_size,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        seq_len=SEQ_LEN,
    )

    trained_model, step_losses, epoch_losses = train_mini_llama(
        model,
        train_dataloader,
        vocab_size=vocab_size,
        pad_id=tokenizer.pad_id,
        device=device,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = CKPT_DIR / f"minillama_char_{timestamp}.pt"
    loss_plot_path = CKPT_DIR / f"loss_curve_{timestamp}.png"

    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "vocab_size": vocab_size,
            "seq_len": SEQ_LEN,
            "hidden_dim": HIDDEN_DIM,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "tokenizer_path": str(BPE_TOKENIZER_PATH),
            "pad_id": tokenizer.pad_id,
            "unk_id": tokenizer.unk_id,
            "eos_id": tokenizer.eos_id,
        },
        ckpt_path,
    )
    plot_training_loss(step_losses, epoch_losses, loss_plot_path)
    print(f"模型权重已保存到: {ckpt_path}")