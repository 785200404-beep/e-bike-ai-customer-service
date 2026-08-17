# -*- coding: utf-8 -*-
"""客服 API 服务 —— 把小程序的请求转给客服（部署/API 化：客服从"命令行"变"接口"）
用法：
    pip install flask
    python server.py                                  # 老师傅用本地模型
    LLM=deepseek DEEPSEEK_API_KEY=sk-xxx python server.py   # 老师傅用 DeepSeek 云端
    curl -X POST http://127.0.0.1:8000/api/chat \
         -H 'Content-Type: application/json' \
         -d '{"question": "首驱Sz110多少钱？"}'
"""
import os, re, json, datetime

from flask import Flask, request, jsonify, send_from_directory

import customer_service as cs   # 启动时一次性加载模型（分流器 0.6B + 客服大脑），之后每个请求复用

app = Flask(__name__)

# ============ 数据文件（留资 / 门店 / 库存） ============
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LEADS_PATH = os.path.join(_DATA_DIR, "leads.json")
STORE_PATH = os.path.join(_DATA_DIR, "store.json")

def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _get_store():
    """读门店信息；文件不存在就建一个占位（地址/电话老板自己改），admin_key 也在这里"""
    store = _read_json(STORE_PATH, None)
    if not isinstance(store, dict) or not store:
        store = {
            "name": "首驱电动车 · 南宁一网门店",
            "address": "（门店地址，请编辑 data/store.json 填入）",
            "phone": "0771-0000000",
            "hours": "每天 9:00-21:00，周末不休",
            "lat": 22.817, "lng": 108.366,
            "admin_key": "yadi2026",
        }
        _write_json(STORE_PATH, store)
    return store

def _admin_ok():
    """/admin 和写接口都带 ?key=admin_key 才能进（店里内部工具，防公网被人乱点）"""
    return request.args.get("key") == _get_store().get("admin_key")

# ============ 会话记忆（短期记忆：多轮对话历史） ============
# 模型本身没记忆，"记忆"= 把前几轮问答塞回上下文。
# 这里用内存 dict 存：session_id → [(问题, 回答), ...]。
# 重启即失；员工体验版够用，生产要换 Redis/数据库（长期记忆）。
SESSIONS = {}
MAX_HISTORY_ROUNDS = 6   # 记住最近 6 轮（上下文会越塞越满，先短后长，够用再调）
MAX_SESSIONS = 200       # 防内存泄漏：会话数超了丢最老的


@app.post("/api/chat")
def chat():
    """小程序入口：POST {"question": "...", "session_id": "..."}
    → {"ok": true, "answer": "...", "used_tools": [...], "session_id": "..."}
    session_id 同一客户每次带同一个，客服就能"记得"前面聊了什么。"""
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "question 不能为空"}), 400
    sid = (data.get("session_id") or "default").strip() or "default"

    history = SESSIONS.get(sid, [])[-MAX_HISTORY_ROUNDS:]
    # 小程序可能带客户坐标（wx.getLocation）→ 问"就近门店"时算最近门店
    try:
        lat = float(data.get("lat")) if data.get("lat") is not None else None
        lng = float(data.get("lng")) if data.get("lng") is not None else None
    except (TypeError, ValueError):
        lat = lng = None
    answer, used_tools = cs.serve(q, return_tools=True, history=history, lat=lat, lng=lng)
    # 存历史：只存"最终问答对"，工具内部往返（CALC/CHECK_STOCK）不进历史
    SESSIONS[sid] = (SESSIONS.get(sid, []) + [(q, answer)])[-MAX_HISTORY_ROUNDS:]
    if len(SESSIONS) > MAX_SESSIONS:  # 简单防内存泄漏：丢最早写入的会话
        for k in list(SESSIONS)[:len(SESSIONS) - MAX_SESSIONS]:
            SESSIONS.pop(k, None)
    return jsonify({"ok": True, "answer": answer, "used_tools": used_tools, "session_id": sid})


@app.get("/health")
def health():
    return jsonify({"ok": True, "llm": cs.LLM, "shunt": cs.SHUNT, "sessions": len(SESSIONS)})


# ============ 留资：客户留电话/微信 → 存 data/leads.json ============
@app.post("/api/lead")
def lead():
    """POST {phone, model?, note?} → 存留资表（P0 留资：把问价的潜在客户抓下来）"""
    data = request.get_json(force=True, silent=True) or {}
    phone = (data.get("phone") or "").strip()
    if not re.fullmatch(r"1\d{10}", phone):
        return jsonify({"ok": False, "error": "手机号格式不对（要 11 位 1 开头）"}), 400
    leads = _read_json(LEADS_PATH, [])
    if not isinstance(leads, list):
        leads = []
    leads.append({
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "phone": phone,
        "model": (data.get("model") or "").strip(),
        "note": (data.get("note") or "").strip(),
    })
    _write_json(LEADS_PATH, leads)
    return jsonify({"ok": True})


# ============ 门店信息（小程序/网页版取来展示，不含 admin_key） ============
@app.get("/api/store")
def store():
    """返回：store=总店（老板看板可编辑）、stores=南宁一网 13 家全量（data/stores.json）。
    小程序"到店"弹层按 stores 列表展示；admin_key 一律不外泄。"""
    s = dict(_get_store())
    s.pop("admin_key", None)
    # 全量门店清单（13 家）——只给前端展示用的字段
    branches = []
    for st in cs.STORES:
        item = {k: st.get(k) for k in ("name", "area", "address", "phone", "hours", "lat", "lng", "primary")}
        branches.append(item)
    return jsonify({"ok": True, "store": s, "stores": branches})


@app.post("/api/store")
def store_set():
    """老板看板上改门店信息（地址/电话/营业时间），admin_key 才能改"""
    if not _admin_ok():
        return jsonify({"ok": False, "error": "key 不对"}), 403
    data = request.get_json(force=True, silent=True) or {}
    s = _get_store()
    for k in ("name", "address", "phone", "hours"):
        if isinstance(data.get(k), str) and data[k].strip():
            s[k] = data[k].strip()
    if isinstance(data.get("lat"), (int, float)) and isinstance(data.get("lng"), (int, float)):
        s["lat"] = float(data["lat"]); s["lng"] = float(data["lng"])
    _write_json(STORE_PATH, s)
    out = dict(s)
    out.pop("admin_key", None)
    return jsonify({"ok": True, "store": out})


# ============ 库存：GET 看 / POST 改（改完热更新，不用重启） ============
@app.get("/api/stock")
def stock_get():
    return jsonify({"ok": True, "stock": cs.STOCK})


@app.post("/api/stock")
def stock_set():
    if not _admin_ok():
        return jsonify({"ok": False, "error": "key 不对"}), 403
    data = request.get_json(force=True, silent=True) or {}
    new = data.get("stock")
    if not isinstance(new, dict):
        return jsonify({"ok": False, "error": "stock 要是对象"}), 400
    # 合并式更新：只改传上来的车型，保留其他车型不动（老板看板每次会传全量，
    # 但合并式能避免部分更新/并发时把别的车型冲掉）
    current = dict(cs.STOCK)
    for k, v in new.items():
        if k in current and isinstance(v, (int, float)):
            current[k] = max(0, int(v))
    _write_json(os.path.join(_DATA_DIR, "stock.json"), current)
    cs.reload_stock()
    return jsonify({"ok": True, "stock": cs.STOCK})


# ============ 老板看板数据 ============
def _kb_model_names():
    """从知识库提所有在售车型名（首驱 X：...），供热门车型统计"""
    names = set()
    for block in cs.kb:
        m = re.match(r'\s*首驱\s+([^：:]+)', block)
        if m:
            names.add(m.group(1).strip())
    return names


@app.get("/api/admin")
def admin():
    if not _admin_ok():
        return jsonify({"ok": False, "error": "key 不对"}), 403
    leads = _read_json(LEADS_PATH, [])
    if not isinstance(leads, list):
        leads = []
    now = datetime.datetime.now()

    def age_days(t):
        try:
            dt = datetime.datetime.fromisoformat(t)
            return max(0, (now - dt).days)
        except Exception:
            return 0

    today = now.strftime("%Y-%m-%d")
    today_leads = sum(1 for l in leads if str(l.get("time", "")).startswith(today))
    leads_sorted = sorted(leads, key=lambda x: str(x.get("time", "")), reverse=True)

    # 今日对话数 + 热门车型（从问答日志粗统计）
    today_chats = 0
    mentions = {}
    model_names = _kb_model_names()
    try:
        with open(cs.LOG_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("time", "")).startswith(today):
                    today_chats += 1
                qa = str(rec.get("question", "")) + str(rec.get("answer", ""))
                nqa = cs._norm(qa)
                for name in model_names:
                    if name and cs._norm(name) in nqa:
                        mentions[name] = mentions.get(name, 0) + 1
    except Exception:
        pass
    hot = sorted(mentions.items(), key=lambda kv: -kv[1])[:8]

    return jsonify({
        "ok": True,
        "today_leads": today_leads,
        "total_leads": len(leads),
        "today_chats": today_chats,
        "leads": [{"time": l.get("time"), "phone": l.get("phone"), "model": l.get("model"),
                   "note": l.get("note"), "age_days": age_days(str(l.get("time", "")))}
                  for l in leads_sorted[:20]],
        "hot_models": hot,
        "stock": cs.STOCK,
    })


@app.get("/admin")
def admin_page():
    """老板看板：/admin?key=xxx 进；没 key 就让页面弹密码框"""
    return send_from_directory("web", "admin.html")


@app.get("/")
def web():
    """手机网页版聊天页：同事/顾客用手机浏览器打开即可，无需装小程序（配合内网穿透可公网访问）"""
    return send_from_directory("web", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"客服 API 已上线：http://127.0.0.1:{port}/api/chat（LLM={cs.LLM}，分流器={'开' if cs.SHUNT else '关'}）")
    # debug=False 避免 Flask 双进程重载导致模型加载两遍
    app.run(host="0.0.0.0", port=port, debug=False)
