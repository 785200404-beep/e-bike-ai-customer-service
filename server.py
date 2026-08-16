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
import os

from flask import Flask, request, jsonify, send_from_directory

import customer_service as cs   # 启动时一次性加载模型（分流器 0.6B + 客服大脑），之后每个请求复用

app = Flask(__name__)

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
    answer, used_tools = cs.serve(q, return_tools=True, history=history)
    # 存历史：只存"最终问答对"，工具内部往返（CALC/CHECK_STOCK）不进历史
    SESSIONS[sid] = (SESSIONS.get(sid, []) + [(q, answer)])[-MAX_HISTORY_ROUNDS:]
    if len(SESSIONS) > MAX_SESSIONS:  # 简单防内存泄漏：丢最早写入的会话
        for k in list(SESSIONS)[:len(SESSIONS) - MAX_SESSIONS]:
            SESSIONS.pop(k, None)
    return jsonify({"ok": True, "answer": answer, "used_tools": used_tools, "session_id": sid})


@app.get("/health")
def health():
    return jsonify({"ok": True, "llm": cs.LLM, "shunt": cs.SHUNT, "sessions": len(SESSIONS)})


@app.get("/")
def web():
    """手机网页版聊天页：同事/顾客用手机浏览器打开即可，无需装小程序（配合内网穿透可公网访问）"""
    return send_from_directory("web", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"客服 API 已上线：http://127.0.0.1:{port}/api/chat（LLM={cs.LLM}，分流器={'开' if cs.SHUNT else '关'}）")
    # debug=False 避免 Flask 双进程重载导致模型加载两遍
    app.run(host="0.0.0.0", port=port, debug=False)
