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
from urllib.parse import quote
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
def _norm(s):
    """检索归一化：去掉所有空格（「Sz 110」=「Sz110」）+ 转小写，客户怎么写都能捞到"""
    return s.replace(" ", "").replace("　", "").lower()

def _de_markdown(s):
    """剥掉模型爱加的 Markdown 符号——小程序聊天框是纯文本，不渲染 Markdown，
    客户看到的 `**电池**` 就是一堆星号点点。回答出口统一过一遍，模型写不写 markdown 都不影响体验。"""
    s = s.replace("**", "").replace("*", "").replace("`", "")          # 加粗/斜体/行内代码的 * 和 `
    s = re.sub(r'^#{1,6}\s*', '', s, flags=re.M)                        # 标题 # 去掉
    s = re.sub(r'(?m)^\s*[-•]\s+', '', s)                               # 行首 - / • 列表符去掉
    s = re.sub(r'([0-9])\n([A-Za-z]+(?=[，。,.、]))', r'\1\2', s)       # "4600\nW，" 数字断行拼回去
    return s.strip()

def _strip_proactive_stock(answer, question):
    """防客服嘴碎：客户没问库存，模型却爱在答尾追加'需要我帮您查一下库存吗'这类主动推销。
    这类句子整句剥掉；客户本来就在问库存/现货时不动（那是真需求，评估也要考）。"""
    if any(w in question for w in ("库存", "现货", "有货", "提车", "现车", "提走", "货源", "能提")):
        return answer
    parts = re.split(r'(?<=[。！？!?；;])', answer)   # 按句号/问号切句
    while parts:
        tail = parts[-1].strip()
        if not tail:
            parts.pop()
            continue
        # 末尾句子同时含"库存/现货"和"需要/帮/要不要/查一下" → 是主动推销库存，剥掉
        if re.search(r'(库存|现货)', tail) and re.search(r'(需要|帮|要不要|查一下|查查)', tail):
            parts.pop()
        else:
            break
    cleaned = ''.join(parts).strip()
    return cleaned if cleaned else answer

# 系列问法：「K系列 / Sz系列 / 店里有哪些O系」→ 直接把整个系列捞出来
# （不然 n-gram 打分对单字母系列 K/S/Y/O 会捞空——车型名是「K1 95CV」，查询里的「k系」匹配不上）
_SERIES_RE = re.compile(r'(?:首驱|店里|你们|我们|在售|有)?\s*([A-Za-z]{1,2})[0-9]*\s*(?:系列|系)')

def _block_series(block):
    """从知识库一行提取系列前缀：『首驱 K1 95CV MAX：…』→ 'k'；『首驱 Sz5 Ultra』→ 'sz'"""
    m = re.match(r'\s*首驱\s+([A-Za-z]+)', block)
    return m.group(1).lower() if m else ""

def retrieve(question, top_n=8):
    # 1) 系列问法：命中「K系列/Sz系列」→ 把该系列所有车型都捞出来（整系列列全，不许截断）
    m = _SERIES_RE.search(question)
    if m:
        series = m.group(1).lower()
        blocks = [b for b in kb if _block_series(b) == series]
        if blocks:
            return blocks[:12]  # 最多 12 款（当前最大系列 K=10 款），全系列都给
    # 2) 普通问法：n-gram 打分
    q = _norm(question)
    scored = []
    for block in kb:
        b = _norm(block)
        score = sum(1 for size in range(2, 7)
                    for i in range(len(q) - size + 1)
                    if q[i:i + size] in b)
        if score > 0:
            scored.append((score, block))
    scored.sort(reverse=True)
    return [b for _, b in scored[:top_n]]

# ============ 4. 工具（手脚）：计算器 + 查库存 ============
# 库存表（真实门店数据存在 data/stock.json，老板每天开门改一次 / /admin 网页上改；
# 改完不用重启服务，/api/stock 会热更新内存）。文件缺失/损坏时用默认值兜底，绝不让客服崩。
_STOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stock.json")
_DEFAULT_STOCK = {
    "Sz Lite": 5, "Sz110": 3, "K95C Max": 2,
    "Y3 95C MK2": 0, "S300 Ultra": 0,  # S300 Plus 已并入 Ultra（2026-08-16 产品手册同步）
}
STOCK = dict(_DEFAULT_STOCK)

def load_stock():
    """从 data/stock.json 读库存；文件不存在/损坏 → 用默认值兜底（库存别把客服搞崩）"""
    global STOCK
    try:
        with open(_STOCK_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            STOCK = data
    except Exception:
        STOCK = dict(_DEFAULT_STOCK)
    return STOCK

load_stock()

def reload_stock():
    """/api/stock 改完库存后热更新内存（不用重启服务）"""
    return load_stock()

# ============ 3.5 门店网络（data/stores.json，南宁一网 13 家） ============
# 客户问"你们在哪里/就近门店/附近有没有店"时，代码直接算最近门店推送（确定性），不劳烦模型猜。
# 数据存在 data/stores.json（老板可自己改地址/电话/坐标），改了重启服务或 /api/store 重新拉取。
_STORES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stores.json")

def load_stores():
    """读门店清单 + 售后总部；文件缺失/损坏 → 空列表兜底（别让客服崩）。
    stores.json 里 stores 是 13 家销售门店，after_sales 是售后总部（独立条目，不算进"13 家"）。"""
    global STORES, AFTER_SALES
    AFTER_SALES = None
    try:
        with open(_STORES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            raw_as = data.get("after_sales")
            if isinstance(raw_as, dict) and raw_as.get("name"):
                AFTER_SALES = raw_as
            data = data.get("stores", [])
        if isinstance(data, list) and data:
            STORES = [s for s in data if isinstance(s, dict) and s.get("name")]
            return STORES
    except Exception:
        pass
    STORES = []
    return STORES

AFTER_SALES = None  # 售后总部（首驱售后总部，独立于 13 家门店）

STORES = load_stores()

def _haversine_km(lat1, lng1, lat2, lng2):
    """球面距离（公里）——用来找"离客户最近的门店" """
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

def _nearest_store(lat, lng):
    """给定客户坐标 → 返回 (最近门店 dict, 距离公里)；没有门店/坐标非法 → (None, None)"""
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return None, None
    best, best_d = None, None
    for s in STORES:
        try:
            d = _haversine_km(lat, lng, float(s["lat"]), float(s["lng"]))
        except (TypeError, ValueError, KeyError):
            continue
        if best_d is None or d < best_d:
            best, best_d = s, d
    return best, best_d

def run_tool(content):
    """检测模型输出里的工具调用，执行并返回真实结果（CALC 计算器 / CHECK_STOCK 查库存）"""
    m = re.search(r'CALC:([0-9+\-*/.() ]+)', content)
    if m:
        expr = m.group(1).strip()
        if re.fullmatch(r'[0-9+\-*/.() ]+', expr):  # 安全校验：只许数字和运算符
            try:
                result = eval(expr)  # 教学演示；生产要换安全执行器
                return f"计算器结果：{expr} = {result}", True
            except Exception:
                # 表达式算不出来（模型偶发输出被截断的 "CALC:2 *"）→ 剥掉这次调用，当没调过，
                # 别让客服崩（评估/上线兜底：宁可答不全，不能炸）
                clean = re.sub(r'CALC:[0-9+\-*/.() ]+', '', content).strip()
                return (clean if clean else content), False
    s = re.search(r'CHECK_STOCK:\s*([^。\n]+)', content)
    if s:
        target = s.group(1).strip().strip("，,。.;； ")
        if target in ("全部", "所有", "全部车型", "所有车型", "全店"):
            lines = [f"{name}：现货 {n} 台" if n > 0 else f"{name}：无现货（可预订）"
                     for name, n in STOCK.items()]
            return "库存查询结果（全店）：\n" + "\n".join(lines), True
        # 模糊匹配：模型可能写全名/简称，匹配库存表里的车型名
        # 归一化后再比：模型按知识库写「首驱 Sz 110」（带空格），库存 key 是「Sz110」→ 去空格才匹配得上
        nt = _norm(target)
        hit = next((name for name in STOCK
                    if _norm(name) in nt or nt in _norm(name)), None)
        if hit:
            n = STOCK[hit]
            if n > 0:
                return f"库存查询结果：{hit} 现货 {n} 台，可以提车。", True
            return f"库存查询结果：{hit} 目前无现货，可预订，到货时间以门店为准。", True
        # 没匹配到库存表 → 查一下是不是店里在售车型（在售但库存表没收录 → 别冤枉"店里没有"）
        if _kb_has_model(target):
            return f"库存查询结果：「{target}」店里在售，实时库存请到店或来电确认。", True
        return f"库存查询结果：店里没有「{target}」这个车型，请直接告诉客户'店里没有这款车'。", True
    return content, False

def _kb_has_model(target):
    """target 是不是店里在售车型（出现在价目表里）——在售但库存表没收录 → 不该说"店里没有" """
    nt = _norm(target)
    for block in kb:
        m = re.match(r'\s*首驱\s+([^：:]+)', block)
        if m and (nt in _norm(m.group(1)) or _norm(m.group(1)) in nt):
            return True
    return False

# —— 库存确定性保险（v5.7）：模型偶尔"懒得调 CHECK_STOCK 就编库存"，这里替它兜底 ——
# 背景：DeepSeek 云端实测 Y3 95C MK2 / S300 Ultra 偶发不调工具，凭空编"有现货"（实际 0 台）。
# 修法：客户问库存但没用上 CHECK_STOCK → 代码直接代调一次，把真实库存塞回上下文再答；还不听就用模板兜底。
def _is_stock_question(q):
    """判断客户是不是在问库存/现货/能否提车"""
    return any(w in q for w in ("库存", "现货", "有货", "还有", "能提", "提车", "现车", "货源", "有没有货", "有现车"))

def _extract_model(question, hits=None):
    """从问题里找"库存表里的车型"——只有这些有 CHECK_STOCK 真实数据，别拿其他车型去查。
    只认问题本身点名的车型；hits 检索结果不作数（v5.9 修：开列型问题「今天能提哪些」
    会捞到竞品行/参数说明行里的别的车型名，拿去代调 CHECK_STOCK 就答错了）。"""
    nq = _norm(question)
    for name in STOCK:
        if _norm(name) in nq:
            return name
    return None

def _open_stock_answer():
    """开列/反问型库存问题（客户没点车型，只问"有没有现车/能提哪些"）的确定性回答：
    直接把有现货的车型列出来再反问，比干巴巴一句"想了解哪款"更帮忙；数据来自 STOCK，绝不编。"""
    names = "、".join(f"首驱{name}" for name, n in STOCK.items() if n > 0)
    if names:
        return f"目前有现货可以提的车型：{names}。您具体想看哪款车？我帮您查库存和价格。"
    return "目前店里现货不多，多数车型需要预订。您具体想看哪款车？我帮您查库存和价格。"

# ============ 4.7 竞品"有没有"确定性护栏（v5.9） ============
# 背景：客户问"店里有没有雅迪冠能E9 PRO？"这类"店里有没有竞品车"时，
# 检索会把竞品-行整段捞进上下文，模型看到完整参数就顺着编"有在售的"——这是大错。
# 修法：命中"有没有/有卖/在售" + 竞品品牌/竞品车型 → 代码直接答"没有"，不劳烦模型（与库存保险同理）。
_COMP_BRANDS = ("雅迪", "小牛", "九号", "极核")

def _kb_competitor_models():
    """从知识库竞品行提取竞品车型名（'雅迪冠能E9 PRO：…'、竞品行'代表车型：…、…'）"""
    names = set()
    for block in kb:
        if block.startswith("竞品-"):
            m = re.search(r'代表车型：([^\n]+)', block)
            if m:
                for n in re.split(r'[、,，]', m.group(1)):
                    n = n.strip()
                    if n:
                        names.add(n)
        elif re.match(r'^(雅迪|小牛|九号|极核)[^：:]+：', block):
            m = re.match(r'^([^：:]+)：', block)
            if m:
                names.add(m.group(1).strip())
    return names

_COMP_MODELS = _kb_competitor_models()

def _competitor_avail(question):
    """客户问"店里有没有/有卖/在售"竞品车吗？命中返回竞品车型名（如'雅迪冠能E9 PRO'），否则 None。
    只认"问有没有"的，绝不误伤"哪个好/怎么选/比"这类对比题；售后/服务类问法不拦。"""
    if any(w in question for w in ("售后", "维修", "服务", "保修", "以旧换新")):
        return None
    if not re.search(r'有没有|有卖|有没有卖|在售|有现货|有现车|有货|卖.{0,8}(吗|么|？)', question):
        return None
    nq = _norm(question)
    for brand in _COMP_BRANDS:
        if brand in question:
            i = question.find(brand)
            rest = question[i + len(brand):]
            m = re.match(r'[^有的？?。,.，！!吗么]+', rest)  # 停在"有/的/标点/语气词"前
            if m and m.group(0).strip():
                return brand + m.group(0).strip()
            return brand
    # 没带品牌，但问的是资料里明确的竞品车型（如"冠能E9 PRO"）
    for name in _COMP_MODELS:
        if _norm(name) in nq:
            return name
    return None

# ============ 4.75 门店位置/就近门店确定性护栏（v5.11） ============
# 背景：客户问"你们在哪里 / 就近门店 / 附近有没有店"时，n-gram 检索捞不到门店地址行
# （"你们在哪里"和地址文本没有任何公共子串），模型没资料就编。而且"就近门店"必须靠客户位置，
# 模型根本不知道客户在哪。修法：命中位置问法 → 代码直接答。
#   带客户坐标（lat/lng，小程序 wx.getLocation 传来）→ 算最近门店推送；
#   没坐标 → 报总店地址 + 说明 13 家分店 + 反问客户在哪个区/地标，别瞎猜。
# 用正则收紧命中，避免误伤非位置问法：
#   - 不匹配裸"导航"（"这车有导航吗"是车机导航，不是门店位置）
#   - "最近/附近"后面必须跟"店/门店/家"（"最近有优惠吗/附近有充电桩吗"不拦）
_LOC_RE = re.compile(
    r'在哪里|在哪|在哪儿|什么位置|位置在|'
    r'门店地址|店址|门店位置|店的地址|你们地址|地址是|地址在|地址发我|'
    r'就近|最近.{0,4}(店|门店|家)|离我|附近.{0,4}(店|门店)|'
    r'怎么走|怎么去|怎么过来|怎么过去|在哪条|'
    r'导航到|导航过去|导航去|导航过来|导航一下'
)

def _is_location_question(q):
    """客户是不是在问"你们店在哪 / 就近门店 / 附近有没有店"？"""
    return bool(_LOC_RE.search(q))

def _loc_store_line(s):
    """门店一句话简介：店长 + 地址 + 电话 + 营业时间"""
    c = f"店长 {s['contact']}，" if s.get("contact") else ""
    return f"【{s.get('name')}】{s.get('address')}，{c}电话 {s.get('phone')}，营业 {s.get('hours') or '每天 9:00-21:00'}"

_AFTER_SALES_RE = re.compile(r'售后|维修|修车|保养')
def _is_after_sales_question(q):
    """客户是不是在问售后/维修/保养？命中 → 报售后总部（确定性），别让模型拿"13 家门店"糊弄"""
    return bool(_AFTER_SALES_RE.search(q))

def _after_sales_store():
    return AFTER_SALES if isinstance(AFTER_SALES, dict) and AFTER_SALES.get("name") else None

def _after_sales_answer():
    s = _after_sales_store()
    if not s:
        return "售后信息还没填，老板去 data/stores.json 的 after_sales 填一下。"
    c = f"店长 {s['contact']}，" if s.get("contact") else ""
    return (f"首驱售后总部：{s.get('address')}，{c}电话 {s.get('phone')}。"
            f"点下面的「导航到店」可直达，或先打售后电话 {s.get('phone')}。")

# ============ 4.8 跑外卖/跑单选车确定性护栏（v5.14） ============
# 背景：顾客问"拿来跑外卖"时，知识库里没有外卖词条，模型拿价目表车型编参数
# （编出"首驱A7"、把 K95C Max 说成 72V38Ah/150km/快充，实际 72V32Ah/82km 无快充）。
# 外卖骑手是门店大客群，这类问答必须保真。修法：命中"外卖/跑单/跑腿/送餐"→ 代码直接给选车建议。
_DELIVERY_RE = re.compile(r'外卖|跑单|跑腿|送餐')
def _is_delivery_question(q):
    return bool(_DELIVERY_RE.search(q))

def _kb_spec_line(name):
    """价目表原文：找 '首驱 {name}：…' 那一行（别的行/竞品行不算），找不到返回 ''"""
    nn = _norm(name)
    for block in kb:
        b = block.split("\n")[0].strip()
        if b.startswith("首驱") and "：" in b:
            bname = b.split("：", 1)[0].replace("首驱", "", 1).strip()
            if _norm(bname) == nn:
                return b
    return ""

def _delivery_reco():
    """没点名车型 → 按日里程分档推荐（车型/参数照价目表原文，绝不编）"""
    return (
        "能跑，外卖完全够用——关键是按一天跑的里程选对车（外卖天天高强度骑，别只看纸面续航，要留 20% 余量）：\n"
        "· 市区短途（一天 60 公里内）：新国标电自免驾照、上绿牌——首驱Sz 110（48V30Ah锂电，续航110km，5199元）\n"
        "· 跑得远（一天 60-100 公里）：选电摩，但要考驾照、上黄牌——首驱K1 95CV MAX（72V32Ah，等速续航90km，极速73）"
        "、首驱Y3 95C MK2（72V32Ah，等速续航100km，6999元）\n"
        "· 要极速、接长途大单：首驱K95C MAX（72V32Ah，全速续航82km，极速70，6699元）\n"
        "另外：满载爬坡、手机充电都会掉电；天天骑磨损快，售后近很关键——南宁 13 家店都能修。"
        "以旧换新还能抵 300-800 元。您一天大概跑多少公里？我帮您锁一款，或留个电话让店里把实车视频发您。"
    )

def _delivery_for_model(m):
    """客户点名具体车型 → 按这款车实际情况答（照价目表，不编）"""
    spec = _kb_spec_line(m)
    if spec and "：" in spec:  # 剥掉 "首驱 Sz 110：" 前缀，只留参数部分
        spec = spec.split("：", 1)[1].strip()
    line = f"首驱{m}（{spec}）" if spec else f"首驱{m}"
    avail = STOCK.get(m, 0)
    if "新国标" in spec:
        cls, verdict = "新国标电自，免驾照、上绿牌", "适合市区短途配送（一天 60 公里内）；跑得更远建议看续航更长的款"
    elif "轻便摩托车" in spec or "电轻摩" in spec:
        cls, verdict = "电动轻便摩托车，需驾照（F照）、上蓝牌", "短途配送够用，但极速/续航一般，长途不推荐"
    else:
        cls, verdict = "电摩，需驾照（E/D照）、上黄牌", "跑得快、续航长，适合全天跑单；注意市区限行路段"
    stock_txt = f"店里目前现货 {avail} 台" if avail else "店里目前无现货，可预订"
    return (f"您问的{line}。这是{cls}，{verdict}。{stock_txt}。"
            f"拿不准的话，告诉我您一天跑多少公里，我帮您对比着选；或留个电话，让店里把实车视频发您。")

def _delivery_answer(question, history=None):
    """跑外卖选车（确定性）：点名车型→按车型答；没点名→按里程分档推荐"""
    m = _extract_model(question)
    if not m and history:
        for q0, _ in reversed(history):
            m = _extract_model(q0)
            if m:
                break
    return _delivery_for_model(m) if m else _delivery_reco()

def _location_answer(question, lat=None, lng=None):
    """位置问法的确定性回答（数据来自 data/stores.json，绝不编）"""
    nearest, dist = _nearest_store(lat, lng) if (lat is not None and lng is not None) else (None, None)
    if nearest:
        dist_txt = f"（约 {dist:.1f} 公里）" if dist is not None else ""
        return (f"离您最近的是{dist_txt}：{_loc_store_line(nearest)}。"
                f"点下面的「导航到店」直接过去，或留个电话，我让店里同事把实车视频和路线发您。")
    # 没定位 → 报总店 + 反问区，别瞎猜客户在哪
    primary = next((s for s in STORES if s.get("primary")), STORES[0] if STORES else None)
    if not primary:
        return "门店信息还没填，老板去 data/stores.json 填一下。"
    n = len(STORES)
    return (f"本店是首驱南宁一网连锁，南宁市区共 {n} 家门店。总店在：{_loc_store_line(primary)}。"
            f"点下面的「导航到店」可以直接导航过去。其他区（江南/青秀/兴宁/西乡塘）也都有店，"
            f"您靠近哪个区或哪个地标？告诉我，我帮您推最近的一家。")

def _map_link_of(s):
    """给单个门店/售后条目生成 {name, address, lat, lng, amap, qq} 链接对象；没坐标 → None"""
    if not s or not s.get("lat") or not s.get("lng"):
        return None
    name = s.get("name", "首驱门店")
    addr = s.get("address", "")
    latv, lngv = float(s["lat"]), float(s["lng"])
    amap = (f"https://uri.amap.com/marker?position={lngv},{latv}"
            f"&name={quote(name)}&src=yadi_cs")
    qq = (f"https://apis.map.qq.com/uri/v1/marker"
          f"?marker=coord:{latv},{lngv};title:{quote(name)};addr:{quote(addr)}"
          f"&type=0&src=yadi_cs")
    return {"name": name, "address": addr, "lat": latv, "lng": lngv, "amap": amap, "qq": qq}

def map_link_for(question, lat=None, lng=None):
    """客户问位置/就近门店/售后在哪时，返回可直接点跳地图的链接对象（高德 + 腾讯双链接）：
    {name, address, lat, lng, amap, qq}；非位置问法 / 没坐标 → 返回 None。
    网页版渲染成 <a> 链接，小程序渲染成「导航到店」按钮（wx.openLocation）。
    问售后/维修 → 指向售后总部；问位置/就近 → 有定位指向最近门店、没定位指向总店（答案里也是这么说的）。"""
    if not _is_location_question(question):
        return None
    if _is_after_sales_question(question):  # 问"售后/维修在哪" → 售后总部（独立于 13 家门店）
        return _map_link_of(_after_sales_store())
    nearest, _ = _nearest_store(lat, lng) if (lat is not None and lng is not None) else (None, None)
    s = nearest or next((x for x in STORES if x.get("primary")), STORES[0] if STORES else None)
    return _map_link_of(s)

def _stock_answer_from_result(result, model_name):
    """把 CHECK_STOCK 工具结果转成给客户看的话术（确定性兜底，不再依赖模型自觉）"""
    if "无现货" in result or "可预订" in result:
        return f"{model_name} 目前无现货，可预订，到货时间以门店为准。"
    m = re.search(r'现货\s*(\d+)', result)
    if m:
        return f"{model_name} 目前有现货 {m.group(1)} 台，可以提车。"
    return f"{model_name} 目前缺货，具体到货时间请到店或来电咨询。"

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
def ask(question, hits, history=None, max_steps=6):
    """history: 多轮记忆 [(客户问题, 客服回答), ...]，按时间顺序。模型本身没记忆，
    记忆 = 把前几轮对话塞回上下文（短期记忆）。只存最终的问答对，工具内部往返不进历史。"""
    system = (
        "你是电动车店的客服，老实可靠。规矩："
        "1) 只根据提供的价目表资料回答，不知道就说'店里没有/我不确定'，绝不编造。"
        "2) 客户问价格合计、折扣等需要计算的问题，必须用计算器，禁止心算。"
        "3) 任何坑客户、违法、不道德的要求，必须拒绝并劝阻。"
        "4) 售后维修、保修问题按店内售后资料回答；资料没写具体的（比如保修年限），就说'具体政策按国家三包和品牌质保执行，以购车合同为准，可到店咨询'，不要凭空说'我们不提供'。"
        "5) 客户问整个系列/品类（如'K系列有哪些'、'K1的产品信息'、'都有什么车'）时，把资料里该系列所有车型都列出来（每款一行：型号+电池+续航+核心配置），不要只反问客户想要哪一款；客户明确点名某一款时，才单独详细介绍那一款。"
        "6) 回答用纯文本，禁止用任何 Markdown 符号——不要用 ** 星号加粗（别写'**电池**：'，直接写'电池：'）、不要用 # 标题、不要用 - 项目符号。"
        "7) 车型名必须严格照价目表原文，禁止自己拼造型号名（价目表只有'K1 95CV MAX'，不能说成'K1 95C MAX'）。客户问的型号价目表里没有，就答'店里没有这个型号，相近的有 XX、XX，您指的是哪款？'，绝不能用相近车型的参数冒名顶替。"
        "8) 客户没问库存/现货/提车时，绝对不要主动提库存——不要追加'需要我帮您查一下库存吗'、'要不要帮您查现货'这类话，也不要反问'您想了解哪款车的库存'。只答客户问的，客户没问库存，这个话题就到此为止。"
        "9) 本店只卖首驱，不卖雅迪/小牛/九号/极核等竞品。客户主动提到竞品品牌或要求对比（'哪个好/怎么选/和XX比/对比'）时，才用资料里的'竞品-'和'对比-'行做客观对比：先一句'本店只卖首驱，但可以帮您对比'，再点出首驱优势（智能配置/动力/性价比），有需要可引用'首驱卖点-'行；客户没提竞品、只问首驱自家车时，绝不主动扯竞品。"
        "10) 客户问资料里没写的具体参数（大灯类型/亮度、整备重量、车身尺寸、防水等级等），就说'这个参数资料没标注，以实车和购车合同为准，可到店看实车'，绝不许编一个参数名——例如资料没写某车型的大灯，就别说'LED大灯/双透镜大灯'；资料里写了的（电池/续航/极速/功率/电机/制动/屏幕/解锁/安全）照实答。"
        "11) 客户问完首驱具体车型的价格或库存后，可以在回答末尾自然加一句：'方便留个电话或微信吗？我让店里同事把这款车的实车视频和门店定位发给您。'——这是邀约留资，帮门店抓潜在客户；客户正在对比竞品、或只问售后/门店服务时不要加，别每句都加。\n\n"
        "店内价目表资料：\n" + "\n".join(hits) + "\n\n"
        "资料说明：'首驱...'行是店内价目表（在售车型）；'竞品-...'行是竞品参考（店里不卖，仅对比用）；'首驱卖点-...'行是首驱品牌卖点；'对比-...'行是横向对比话术。只有客户提到竞品或要求对比时才引用后三类，平时只答首驱自家车。\n\n"
        "计算器工具：需要做加减乘除时，必须输出 CALC:<表达式>，例如 CALC:3*4599\n"
        "库存工具：客户问某车型有没有现货/库存/还有没有/还能提吗/能不能提车时，必须输出 CHECK_STOCK:<车型名>，例如 CHECK_STOCK:K95C Max\n"
        "库存工具：客户问到具体车型（如'K95C Max'、'雅迪Q10'）时，必须用 CHECK_STOCK:<车型名> 查；工具返回'店里没有'就说明店里没这款，直接答'店里没有'。"
        "库存工具：客户明确问库存但完全没提任何车型（如只问'有现车没'）时，才回复：'请问您是想了解哪款车的库存？'"
        "\n\n对话开始前已附上之前的对话记录，客户说的'这台/那台/刚才那款'要结合上文理解，不要反问客户指的哪台。"
    )
    messages = [{"role": "system", "content": system}]
    for h_q, h_a in (history or []):
        messages.append({"role": "user", "content": h_q})
        messages.append({"role": "assistant", "content": h_a})
    messages.append({"role": "user", "content": question})

    used_tools = []  # 记下这次调了哪些工具（挖金矿：真实使用痕迹）
    last_tool = None  # 上一个工具结果，防"复读工具"死循环
    forced_stock = False  # 是否代调过 CHECK_STOCK（确定性库存保险）
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
            # —— 库存确定性保险：客户问库存/能否提车，但老师傅没调 CHECK_STOCK 就敢答 → 代调一次再答 ——
            if _is_stock_question(question) and not any("库存查询" in t for t in used_tools):
                target = _extract_model(question, hits)
                if target:
                    # 客户问题里点明了车型 → 代调真实库存，塞回上下文再让老师傅答
                    stock_result, _ = run_tool(f"CHECK_STOCK:{target}")
                    used_tools.append(stock_result)
                    forced_stock = True
                    messages.append({"role": "assistant", "content": f"CHECK_STOCK:{target}"})
                    messages.append({"role": "user", "content": stock_result})
                    continue  # 带真实库存再让老师傅答一轮
                # 客户没点任何车型（只问"有没有现车/今天能提哪些"）→ 别拿检索结果里捞到的车型去代调
                # （v5.9 修：捞错车型会带偏，老师傅顺着"无现货"就编"店里没现货"）。
                # 直接确定性回答：列出现货车型 + 反问想看哪款。
                return _open_stock_answer(), used_tools
            # 兜底：代调过库存，但老师傅最终还是没按真实库存说 → 模板答案顶上
            if forced_stock:
                real = next((t for t in reversed(used_tools) if "库存查询" in t), None)
                if real and not any(w in answer for w in ("现货", "无现货", "可预订", "没货", "缺货", "没有库存")):
                    answer = _stock_answer_from_result(real, _extract_model(question, hits))
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

def serve(question, log=True, return_tools=False, history=None, lat=None, lng=None):
    """上线入口：分流器看门 → 检索 → 回答 → 挖金矿
    log=True 才写日志；--demo 演示模式不写，避免污染真实数据
    return_tools=True 时返回 (answer, used_tools)，给 API 层用（小程序要展示用了啥工具）
    history: 多轮记忆 [(客户问题, 客服回答), ...]，让客服"记得"前面聊了什么
    lat/lng: 客户坐标（小程序 wx.getLocation 传来），问"就近门店"时用来算最近门店"""
    if SHUNT:
        c = shunt_classify(question)
        if c == "拒客违规":   # 学徒直接挡下坏问题，老师傅不用出手
            if log:
                log_chat(question, REFUSE_MSG, [], used_tools=["SHUNT_REFUSE"])
            return (REFUSE_MSG, ["SHUNT_REFUSE"]) if return_tools else REFUSE_MSG
    comp = _competitor_avail(question)  # 问"店里有没有竞品车"→ 确定性答没有（v5.9 护栏，不劳烦模型）
    if comp:
        answer = f"本店只卖首驱，{comp} 店里没有。想看首驱的哪款车？我帮您介绍。"
        if log:
            log_chat(question, answer, [], used_tools=["COMPETITOR_AVAIL"])
        return (answer, ["COMPETITOR_AVAIL"]) if return_tools else answer
    if _is_delivery_question(question):  # 问"拿来跑外卖/跑单"→ 确定性选车建议（v5.14 护栏）
        answer = _delivery_answer(question, history=history)
        if log:
            log_chat(question, answer, [], used_tools=["DELIVERY_ANSWER"])
        return (answer, ["DELIVERY_ANSWER"]) if return_tools else answer
    if _is_location_question(question):  # 问"你们在哪/就近门店"→ 确定性推门店（v5.11 护栏）
        if _is_after_sales_question(question):  # 问"售后/维修在哪"→ 报售后总部（v5.13）
            answer = _after_sales_answer()
            used = "AFTER_SALES_ANSWER"
        else:
            answer = _location_answer(question, lat, lng)
            used = "LOCATION_ANSWER"
        if log:
            log_chat(question, answer, [], used_tools=[used])
        return (answer, [used]) if return_tools else answer
    hits = retrieve(question)
    answer, used_tools = ask(question, hits, history=history)  # 检索不到也让守则+工具兜底（查库存/算账不依赖知识库）
    answer = _de_markdown(answer)  # 剥掉 Markdown 符号：聊天框是纯文本，不能给客户看 **点点**
    answer = _strip_proactive_stock(answer, question)  # 客户没问库存，剥掉答尾'要不要查库存'的嘴碎
    if log:
        log_chat(question, answer, hits, used_tools=used_tools, no_hit=not hits)
    return (answer, used_tools) if return_tools else answer

if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("【客服演示 v5.3：分流看门 + 会查 + 会算 + 会查库存 + 云大脑】")
        demo_q = [
            "首驱Sz110续航多少公里？多少钱？",
            "买3台首驱K95C Max，一共多少钱？",
            "1台首驱Sz110 + 1台首驱Y3 95C MK2，加牌照邮递费15，一共多少钱？",
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
