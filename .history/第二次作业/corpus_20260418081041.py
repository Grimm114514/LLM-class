from datasets import load_dataset
import os

def download_and_save_wiki(output_path, max_chars=150_000_000):
    # 1.5亿个中文字符大约对应 300-400MB 纯文本文件
    print("正在加载维基百科数据...")
    
    # 开启 streaming 流式下载，避免一次性吃满内存
    dataset = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train", streaming=True)
    
    current_chars = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item['completion'].strip() # 根据具体数据集的 key 调整
            if text:
                f.write(text + "\n\n")
                current_chars += len(text)
                
            if current_chars >= max_chars:
                print(f"已达到目标大小，停止拉取。共写入 {current_chars} 个字符。")
                break
                
    print(f"数据已保存至: {output_path}")

# 执行下载
if __name__ == "__main__":
    download_and_save_wiki("my_pretrain_wiki.txt")