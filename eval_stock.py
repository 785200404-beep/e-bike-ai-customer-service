# -*- coding: utf-8 -*-
"""库存工具专项评估：测大脑会不会正确调用 CHECK_STOCK 查库存"""
import sys
sys.path.insert(0, ".")
from customer_service import serve

TESTS = [
    ("首驱K95C Max有现货吗？", "有货"),
    ("首驱Sz110有货吗？", "有货"),
    ("首驱Sz Lite店里还有吗？", "有货"),
    ("首驱Y3 95C MK2能提车吗？", "无货"),
    ("首驱S300 Ultra有现货吗？", "无货"),
    ("你们店里现在有现车吗？", "开列"),
    ("今天能提走的车型有哪些？", "开列"),
]

def check(test):
    q, kind = test
    ans = serve(q, log=False)  # 评估别写金矿日志，避免污染真实数据
    if kind == "有货":
        ok = ("现货" in ans or "有货" in ans) and \
             not any(w in ans for w in ["无现货", "没有", "没货", "无货"])
    elif kind == "无货":
        ok = any(w in ans for w in ["无现货", "没有", "没货", "无货", "预订"])
    else:  # 开列：没指定车型 → 应反问是哪款（不许编"其他车型暂无库存"）
        ok = any(w in ans for w in ["哪款", "哪一款", "哪个车型", "什么车型", "哪款车",
                                    "想了解哪款", "想了解哪一", "想了解的车型", "核对",
                                    "具体是哪", "看中哪"])
    return ok, ans

if __name__ == "__main__":
    print("【库存工具 5 题考试】")
    print("-" * 64)
    right = 0
    for t in TESTS:
        ok, ans = check(t)
        right += 1 if ok else 0
        print(f"{'✅' if ok else '❌'}  {t[0]:<26} → {ans[:44]}")
    print("-" * 64)
    print(f"库存工具正确率 = {right}/{len(TESTS)} = {right / len(TESTS) * 100:.0f}%")
