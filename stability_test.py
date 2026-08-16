# -*- coding: utf-8 -*-
"""稳定性对比：同一批"爱飘"的问法，连问 3 遍，看老师傅靠不靠谱（治随机性验收）
用法：
    MODEL_NAME=Qwen/Qwen3-1.7B python stability_test.py           # 本地 1.7B
    LLM=deepseek DEEPSEEK_API_KEY=sk-xxx python stability_test.py  # DeepSeek 云端
"""
import sys
sys.path.insert(0, ".")
from customer_service import serve

# 压测里"最会飘"的 3 道题：措辞随运行变化
Q = [
    "明天想提车，你们有现车没？",
    "你们店里最贵的那台能不能分期？",
    "我这台车骑了半年电池就不行了，咋办？",
]

for q in Q:
    print(f"\n问：{q}")
    for i in range(3):
        ans = serve(q, log=False)
        print(f"  第{i+1}遍 → {ans[:90]}")
