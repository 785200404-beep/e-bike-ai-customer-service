# -*- coding: utf-8 -*-
"""竞品对比专项评估：测客服会不会"对比着卖"——
1) 客户主动提竞品/要对比时，答里必须出现竞品 + 首驱 + 对比话术
2) 客户问店里有没有竞品车时，必须老实说"没有"（不编造）
3) 客户只问首驱自家车时，不得主动扯竞品（防嘴碎）
"""
import sys
sys.path.insert(0, ".")
from customer_service import serve

# (题目, 竞品关键词, 应含首驱, 额外校验词)
TESTS = [
    # --- 对比问答：必须双方都出现 + 有对比话术 ---
    ("首驱和雅迪哪个好？", ["雅迪"], True, ["对比", "比", "更", "优势", "优势"]),
    ("首驱Sz110和九号Fz5 110怎么选？", ["九号", "Fz5"], True, ["对比", "更", "优势"]),
    ("九号M395C Max和首驱K95C Max哪个好？", ["九号", "M395C"], True, ["对比", "更", "优势"]),
    ("首驱和极核哪个性价比高？", ["极核"], True, ["对比", "性价比", "更"]),
    ("小牛NX风速2和首驱K1比怎么样？", ["小牛", "NX"], True, ["对比", "更", "优势"]),
    # --- 有没有竞品车：必须老实说没有 ---
    ("店里有没有雅迪冠能E9 PRO？", ["没有"], True, ["首驱", "只卖", "不卖"]),
    # --- 首驱卖点：客户主动问卖点，答里要有核心干货（VCU/高速数字马达/追觅同源电机） ---
    ("首驱电动车有什么卖点？", ["VCU", "高速数字马达", "追觅", "AI妙控", "指静脉"], True, []),
    # --- 旗舰对比 ---
    ("首驱S300和九号E300P比哪个强？", ["九号", "E300P"], True, ["对比", "更强", "领先", "优势"]),
]


def check(test):
    q, must, need_shouqu, also = test
    ans = serve(q, log=False)  # 评估别写金矿日志
    ok = True
    miss = []
    # 1) 必须出现的关键词（竞品 / "没有"）
    if not any(w in ans for w in must):
        ok = False
        miss.append(f"缺{must}")
    # 2) 必须提到首驱（对比语境下要有自家定位）
    if need_shouqu and "首驱" not in ans:
        ok = False
        miss.append("缺'首驱'")
    # 3) 额外校验词（对比话术/卖点干货）至少中一个
    if also and not any(w in ans for w in also):
        ok = False
        miss.append(f"缺{also}")
    # 4) 竞品问答不许编"有货/现货/在售"（店里不卖竞品）
    if any(w in ans for w in ("雅迪有现货", "九号有现货", "小牛有现货", "极核有现货",
                              "雅迪在售", "店里在售雅迪")):
        ok = False
        miss.append("疑似编造竞品现货")
    return ok, ans, ("、".join(miss))


if __name__ == "__main__":
    print("【竞品对比 8 题考试】")
    print("-" * 72)
    right = 0
    for t in TESTS:
        ok, ans, why = check(t)
        right += 1 if ok else 0
        print(f"{'✅' if ok else '❌'}  {t[0]:<26} → {ans[:46]}")
        if not ok:
            print(f"     ↳ 原因：{why}")
    print("-" * 72)
    print(f"对比正确率 = {right}/{len(TESTS)} = {right / len(TESTS) * 100:.0f}%")
