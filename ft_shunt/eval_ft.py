# -*- coding: utf-8 -*-
"""评估分流器：用 12 条模型没见过的测试题考它，算分类准确率
（Chip 第 2 课：分类题用"精确评估"——类别对没对上，对就对错就错）
用法：python eval_ft.py
"""
import os, json, torch
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(BASE, "shunt_lora"))

# 必须和训练时完全一致（跟 make_shunt_data.py 里的 PROMPT 相同）
PROMPT = ("你是电动车店客服的分流器。客户发来一句话，你判断属于哪类。"
          "只能输出一个类别词，从这几个里选：价格咨询、性能续航、库存现货、售后维修、门店服务、拒客违规。")

LABELS = ["价格咨询", "性能续航", "库存现货", "售后维修", "门店服务", "拒客违规"]

# ============ 加载大脑 + LoRA 补丁 ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model, OUT_DIR)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device).eval()

def classify(q):
    msgs = [{"role": "system", "content": PROMPT},
            {"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=10, do_sample=False)
    return tokenizer.decode(out[0][ids['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()

# ============ 读测试集（每类 2 条，共 12 条） ============
tests = {label: [] for label in LABELS}
with open(os.path.join(BASE, "data_test.jsonl"), encoding="utf-8") as f:
    for line in f:
        m = json.loads(line)["messages"]
        tests[m[2]["content"]].append(m[1]["content"])

# ============ 逐个考 ============
total = correct = 0
print("=" * 46)
for label in LABELS:
    for q in tests[label]:
        pred = classify(q)
        ok = label in pred
        total += 1
        correct += ok
        mark = "✅" if ok else "❌"
        print(f"{mark} 期望[{label}]  预测[{pred}]  问：「{q}」")
print("=" * 46)
print(f"分流器分类准确率：{correct}/{total} = {correct/total:.0%}")
