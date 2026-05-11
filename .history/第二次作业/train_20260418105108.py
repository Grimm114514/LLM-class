import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import math

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

# ==========================================
# 2. 训练函数定义
# ==========================================
def train_mini_llama(model, train_dataloader, vocab_size, device='cuda'):
    """
    自回归语言模型标准的预训练主循环
    """
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
    scaler = torch.cuda.amp.GradScaler(enabled=False) # bfloat16 下通常设为 False 即可安全运行

    # 交叉熵损失函数 (忽略 pad_token 的 loss 计算，假设 pad_token_id = 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"🚀 开始预训练 | 总 Epoch: {EPOCHS} | 总 Steps: {total_steps}")

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
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})

        # 每个 Epoch 结束打印平均 Loss
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"✅ Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.4f} | Perplexity (困惑度): {math.exp(avg_loss):.4f}")

    print("🎉 预训练结束！")
    return model