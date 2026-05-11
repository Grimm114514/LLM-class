import os
# 必须放在最开头，让所有 huggingface 请求走国内镜像代理
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B" 

print("正在下载并加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("正在下载并加载 模型权重与配置...")
model = AutoModelForCausalLM.from_pretrained(model_id)

print("加载成功！")

prompt = "中国科学院大学是"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))