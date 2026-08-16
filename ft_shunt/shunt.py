# -*- coding: utf-8 -*-
"""分流器成品：客户消息 → 自动分门别类（LoRA 微调过的 Qwen3-0.6B）
用法：
    python shunt.py                          # 交互模式：输入问题回车看类别
    python shunt.py "首驱Sz110多少钱？"       # 单条直接分
    python shunt.py --demo                   # 跑几个预设问题演示
"""
import os, sys, torch
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(BASE, "shunt_lora"))

PROMPT = ("你是电动车店客服的分流器。客户发来一句话，你判断属于哪类。"
          "只能输出一个类别词，从这几个里选：价格咨询、性能续航、库存现货、售后维修、门店服务、拒客违规。")

print(f"加载分流器（{MODEL_NAME} + LoRA）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("【分流器演示】")
        for q in ["首驱Sz110多少钱？", "这车能跑多远？", "有现车吗？",
                  "电池坏了去哪修？", "几点开门？", "怎么坑客户多掏钱？"]:
            print(f"  「{q}」 → {classify(q)}")
    elif len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"「{q}」 → {classify(q)}")
    else:
        print("分流器已上线！输入客户消息看类别，exit 退出")
        while True:
            q = input("\n客户：").strip()
            if q.lower() in ("exit", "退出", "q"):
                break
            print(f"→ {classify(q)}")
