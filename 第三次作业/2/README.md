# 课程作业三：SFT + DPO（基础版）

## 文件结构（按阶段拆分）
- `download.py`：阶段0，下载并保存基座模型
- `stage1_sft_train.py`：阶段1，SFT监督微调
- `stage2_build_dpo_data.py`：阶段2，构建DPO偏好数据
- `stage3_dpo_train.py`：阶段3，DPO偏好对齐训练
- `stage4_generate_figures.py`：阶段4，生成报告用图片（保存到当前目录）
- `prompts/`：提示词独立存放

## 你需要准备的数据
- `data/sft_data.jsonl`（SFT）
  - 每行字段示例：
  - `{"instruction":"...","input":"...","output":"..."}`
- `data/dpo_source.jsonl`（DPO原始）
  - 每行字段示例：
  - `{"instruction":"...","input":"...","chosen":"...","rejected":"..."}`

## 运行顺序
1. `python download.py`
2. `python stage1_sft_train.py`
3. `python stage2_build_dpo_data.py`
4. `python stage3_dpo_train.py`
5. `python stage4_generate_figures.py`

## 图片输出（自动保存到本目录）
- `fig_sft_loss.png`
- `fig_dpo_reward_margin.png`
- `fig_case_comparison.png`

## 建议依赖
`pip install transformers datasets trl matplotlib pandas`
