from transformers import AutoModelForCausalLM, AutoTokenizer

# 填入你在网页左上角看到的模型名称
model_id = "Qwen/Qwen2.5-0.5B" 

print("正在下载并加载 Tokenizer...")
# 这一步会自动下载所有 tokenizer 相关的文件
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("正在下载并加载 模型权重与配置...")
# 这一步会自动下载 config.json, model.safetensors 等文件
model = AutoModelForCausalLM.from_pretrained(model_id)

print("加载成功！")
