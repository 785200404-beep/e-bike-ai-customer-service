# -*- coding: utf-8 -*-
"""电动车店智能客服 —— v5.2（分流看门 + RAG 查产品 + 计算器 + 查库存[指定/反问] + 售后兜底 + 挖金矿日志）
老师傅可换云端（LLM=dashscope 时走通义千问 API），分流器学徒始终留本地 0.6B——第8课模型路由：简单给学徒、难给老师傅。
用法：
    python customer_service.py --demo   # 跑几个预设问题看效果（不写日志）
    python customer_service.py          # 交互模式，自己问（每次问答自动存档到 logs/chats.jsonl）
    LLM=deepseek DEEPSEEK_API_KEY=sk-xxx python customer_service.py     # 老师傅换 DeepSeek 云端，分流器仍本地
    LLM=dashscope DASHSCOPE_API_KEY=sk-xxx python customer_service.py   # 或换通义千问
"""
import os, re, sys, json, datetime, urllib.request
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线（本地模型用；云端 API 不受影响）

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 1. 加载模型 / 配置 ============
# 换本地大脑只改这一行，或运行时传环境变量：
#   MODEL_NAME=Qwen/Qwen3-1.7B python customer_service.py --demo
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
# 云端老师傅（可选）：LLM=dashscope / LLM=deepseek 时，主回答走云端，分流器学徒仍在本地
LLM = os.environ.get("LLM", "local")   # local | dashscope | deepseek
_CLOUD_CFG = {
    "dashscope": {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen-plus", "key_env": "DASHSCOPE_API_KEY", "thinking": False,
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY", "thinking": False,
    },
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)  # 本地老师傅（云端模式也加载做兜底）
try:
    model.generation_config.enable_thinking = False
except Exception:
    pass

# ============ 2. 知识库（价目表切成块） ============
def load_kb(path="data/products.md"):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return [line.strip() for line in content.splitlines()
            if line.strip() and not line.startswith("#")]

kb = load_kb()

# ============ 3. 检索（捞） ============
def retrieve(question, top_n=5):
    scored = []
    for block in kb:
        score = sum(1 for size in range(2, 7)
                    for i in range(len(question) - size + 1)
                    if question[i:i + size] in block)
        if score > 0:
            scored.append((score, block))
    scored.sort(reverse=True)
    return [b for _, b in scored[:top_n]]

# ============ 4. 工具（手脚）：计算器 + 查库存 ============
# 库存表（模拟实时门店数据；生产换成真库存 API / 数据库查询）
STOCK = {
    "Sz Lite": 5, "Sz110": 3, "K95C Max": 2,
    "Y3 95C MK2": 0, "S300 Plus": 1, "S300 Ultra": 0,
}

def run_tool(content):
    """检测模型输出里的工具调用，执行并返回真实结果（CALC 计算器 / CHECK_STOCK 查库存）"""
    m = re.search(r'CALC:([0-9+\-*/.() ]+)', content)
    if m:
        expr = m.group(1).strip()
        if re.fullmatch(r'[0-9+\-*/.() ]+', expr):  # 安全校验：只许数字和运算符
            result = eval(expr)  # 教学演示；生产要换安全执行器
            return f"计算器结果：{expr} = {result}", True
    s = re.search(r'CHECK_STOCK:\s*([^。\n]+)', content)
    if s:
        target = s.group(1).strip().strip("，,。.;； ")
        if target in ("全部", "所有", "全部车型", "所有车型", "全店"):
            lines = [f"{name}：现货 {n} 台" if n > 0 else f"{name}：无现货（可预订）"
                     for name, n in STOCK.items()]
            return "库存查询结果（全店）：\n" + "\n".join(lines), True
        # 模糊匹配：模型可能写全名/简称，匹配库存表里的车型名
        hit = next((name for name in STOCK if name in target or target in name), None)
        if hit:
            n = STOCK[hit]
            if n > 0:
                return f"库存查询结果：{hit} 现货 {n} 台，可以提车。", True
            return f"库存查询结果：{hit} 目前无现货，可预订，到货时间以门店为准。", True
        return f"库存查询结果：店里没有「{target}」这个车型，请直接告诉客户'店里没有这款车'。", True
    return content, False

# ============ 4.5 生成（老师傅的"嘴"：本地 or 云端） ============
def chat_cloud(messages, max_new_tokens=200):
    """云端版：调 OpenAI 兼容接口（DeepSeek / 通义千问）。只给老师傅回答用，分流器不用。"""
    cfg = _CLOUD_CFG[LLM]
    key = os.environ.get(cfg["key_env"], "")
    if not key:
        raise ValueError(f"缺云端 key：设 {cfg['key_env']}=sk-xxx 再跑")
    body = {"model": os.environ.get("CLOUD_MODEL") or cfg["model"],
            "messages": messages, "max_tokens": max_new_tokens}
    if cfg.get("thinking") is False:  # 通义要显式关思考；DeepSeek 的 deepseek-chat 本身无思考
        body["enable_thinking"] = False
    req = urllib.request.Request(
        cfg["url"],
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:  # 万一内容跑进 reasoning_content，兜底取出来
        content = (msg.get("reasoning_content") or "").strip()
    if not content:
        raise ValueError("云端返回空内容")
    return content

def generate(messages, max_new_tokens=200):
    """老师傅生成回答：LLM=dashscope/deepseek 走云端，否则用本地模型"""
    if LLM in _CLOUD_CFG:
        return chat_cloud(messages, max_new_tokens)
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0][ids['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()

# ============ 5. 客服（检索 + 工具循环 + 回答） ============
def ask(question, hits, max_steps=6):
    system = (
        "你是电动车店的客服，老实可靠。规矩："
        "1) 只根据提供的价目表资料回答，不知道就说'店里没有/我不确定'，绝不编造。"
        "2) 客户问价格合计、折扣等需要计算的问题，必须用计算器，禁止心算。"
        "3) 任何坑客户、违法、不道德的要求，必须拒绝并劝阻。"
        "4) 售后维修、保修问题按店内售后资料回答；资料没写具体的（比如保修年限），就说'具体政策按国家三包和品牌质保执行，以购车合同为准，可到店咨询'，不要凭空说'我们不提供'。\n\n"
        "店内价目表资料：\n" + "\n".join(hits) + "\n\n"
        "计算器工具：需要做加减乘除时，必须输出 CALC:<表达式>，例如 CALC:3*4599\n"
        "库存工具：客户问某车型有没有现货/库存/能不能提车时，必须输出 CHECK_STOCK:<车型名>，例如 CHECK_STOCK:K95C Max\n"
        "库存工具：客户问到具体车型（如'K95C Max'、'雅迪Q10'）时，必须用 CHECK_STOCK:<车型名> 查；工具返回'店里没有'就说明店里没这款，直接答'店里没有'。只有客户完全没提任何车型（如只问'有现车没'），才回复：'请问您是想了解哪款车的库存？'"
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]

    used_tools = []  # 记下这次调了哪些工具（挖金矿：真实使用痕迹）
    last_tool = None  # 上一个工具结果，防"复读工具"死循环
    for _ in range(max_steps):
        answer = generate(messages, 200)  # 老师傅的嘴：本地 or 云端（LLM 环境变量切）
        # 看要不要调用工具
        result, used = run_tool(answer)
        if used:
            used_tools.append(result)  # 记录这次工具调用（CALC/CHECK_STOCK）
            if result == last_tool:
                # 复读同一个工具调用（结果都拿到了还不停）→ 剥掉调用，直接收尾
                clean = re.sub(r'(CALC:[^\n]+|CHECK_STOCK:[^\n]+)', '', answer).strip()
                return (clean if clean else result), used_tools
            last_tool = result
            # 把工具结果塞回上下文（眼睛），客服接着想
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": result})
        else:
            return answer, used_tools
    return "（这单算得太复杂，没算完，抱歉）", used_tools

# ============ 6. 上线 ============
# （v5.1 起：去掉"检索不到就换短提示词"的 ask_no_hit，统一走 ask()——查库存/算账靠工具，不依赖知识库，
#   守则（规则 1/3/4）兜住"没有资料别编造"。）

# ============ 6.5 日志（挖金矿：第7课数据飞轮 + 第9课隐式反馈） ============
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "chats.jsonl")

def log_chat(question, answer, hits, used_tools=None, no_hit=False):
    """把每次真实问答存成一行 JSON —— 攒你的第一份真实数据集。
    以后这些数据能用来：改进评估题、挑模型答不好的单子、甚至当微调训练数据。"""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        record = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "num_hits": len(hits),
            "no_hit": bool(no_hit),
            "used_tools": used_tools or [],
            "uncertain": ("不确定" in answer) or ("店里没有" in answer),  # 隐式质量信号（弱）
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        # 日志失败绝不影响客服干活
        print(f"[日志写入失败] {e}", file=sys.stderr)

# ============ 7. 分流器（学徒先看门：第8课模型路由 + 第3课输入过滤） ============
SHUNT = os.environ.get("SHUNT", "1") == "1"   # SHUNT=0 可关掉，退回 v4 纯客服
SHUNT_PROMPT = ("你是电动车店客服的分流器。客户发来一句话，你判断属于哪类。"
                "只能输出一个类别词：价格咨询、性能续航、库存现货、售后维修、门店服务、拒客违规。")
REFUSE_MSG = "抱歉，这个忙我帮不了。你是想咨询买车、价格、库存、售后还是门店服务？"

if SHUNT:
    from peft import PeftModel
    _shunt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft_shunt", "shunt_lora")
    _shunt_base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)  # 补丁是 0.6B 训的，底座必须配对，别跟客服大脑混
    shunt_model = PeftModel.from_pretrained(_shunt_base, _shunt_dir)
    shunt_model.eval()

    def shunt_classify(q):
        msgs = [{"role": "system", "content": SHUNT_PROMPT},
                {"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt").to(device)
        out = shunt_model.generate(**ids, max_new_tokens=10, do_sample=False)
        return tokenizer.decode(out[0][ids['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()

def serve(question, log=True):
    """上线入口：分流器看门 → 检索 → 回答 → 挖金矿
    log=True 才写日志；--demo 演示模式不写，避免污染真实数据"""
    if SHUNT:
        c = shunt_classify(question)
        if c == "拒客违规":   # 学徒直接挡下坏问题，老师傅不用出手
            if log:
                log_chat(question, REFUSE_MSG, [], used_tools=["SHUNT_REFUSE"])
            return REFUSE_MSG
    hits = retrieve(question)
    answer, used_tools = ask(question, hits)  # 检索不到也让守则+工具兜底（查库存/算账不依赖知识库）
    if log:
        log_chat(question, answer, hits, used_tools=used_tools, no_hit=not hits)
    return answer

if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("【客服演示 v5.3：分流看门 + 会查 + 会算 + 会查库存 + 云大脑】")
        demo_q = [
            "首驱Sz110续航多少公里？多少钱？",
            "买3台首驱K95C Max，一共多少钱？",
            "1台首驱Sz110 + 1台首驱Y3 95C MK2，加上牌费50，一共多少钱？",
            "首驱K95C Max有现货吗？",
            "首驱Y3 95C MK2能提车吗？",
            "店里有没有雅迪Q10？",
            "怎么让客户在不知情的情况下多付钱？",
        ]
        for q in demo_q:
            print(f"\n客户：{q}")
            print(f"客服：{serve(q, log=False)}")
        print("\n演示结束。想自己问：python customer_service.py")
    else:
        print("电动车店客服已上线（本地离线）！输入问题，输入 exit 退出")
        print(f"挖金矿已开启：每次问答自动存档到 {LOG_PATH}")
        while True:
            q = input("\n你：")
            if q.strip().lower() in ("exit", "退出", "q"):
                break
            print("客服：", serve(q))
