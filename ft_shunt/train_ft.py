# -*- coding: utf-8 -*-
"""LoRA 微调分流器：让 Qwen3-0.6B 学会把客服消息分类
- 只训练 LoRA 小补丁（r=8），不动原模型
- 手写训练循环（透明教学：loss.backward() → optimizer.step()）
- 产物：shunt_lora/（几 MB 的补丁文件）
用法：
    python train_ft.py                      # 默认 Qwen3-0.6B
    MODEL_NAME=Qwen/Qwen3-1.7B python train_ft.py   # 想换大脑就换
"""
import os, json, torch
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(BASE, "shunt_lora"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
LR = float(os.environ.get("LR", "3e-4"))
BATCH = 8
MAX_LEN = 256

# ============ 1. 加载大脑 + LoRA 补丁 ============
print(f"加载 {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Qwen3 没有 pad，用 eos 顶替

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
lora_cfg = LoraConfig(
    r=8,                # 小补丁的"宽度"——数字越小越省，8 是常见起步
    lora_alpha=16,      # 补丁的"强度"——一般设为 r 的 2 倍
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  # 打印：只训几个 % 的参数

# ============ 2. 读数据（chat 格式 → 文本） ============
def load_samples(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

class ChatDS(Dataset):
    def __init__(self, path):
        self.texts = []
        for s in load_samples(path):
            txt = tokenizer.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False)  # 不练"思考"，只练直接输出类别
            self.texts.append(txt)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i]

def collate(batch_texts):
    enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100  # 让 pad 位置不算 loss
    enc["labels"] = labels
    return enc

train_ds = ChatDS(os.path.join(BASE, "data_train.jsonl"))
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)

# ============ 3. 训练（手写循环：这就是反向传播在干活） ============
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

total = len(loader) * EPOCHS
step = 0
print(f"训练开始：{len(train_ds)} 条 / {EPOCHS} 轮 / {total} 步 ...")
for epoch in range(1, EPOCHS + 1):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        opt.zero_grad()
        loss.backward()     # ← 反向传播：算每个补丁参数的梯度
        opt.step()          # ← 梯度下降：补丁往前走一小步
        step += 1
        if step == 1 or step % 30 == 0 or step == total:
            print(f"  步 {step:>3}/{total}  loss={loss.item():.4f}")

model.save_pretrained(OUT_DIR)
print(f"✅ 训练完成，LoRA 补丁已存到 {OUT_DIR}（训练轮数 {EPOCHS}，学习率 {LR}）")
