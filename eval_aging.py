"""场景一：应收账龄分析（对应共享核算会计“按月出具账龄分析报告”）。

做三件事：
1. 用 pandas 从账龄表算出标准答案；
2. 让大模型回答同样的 10 个问题，逐题对比标准答案；
3. 让大模型写分析摘要，检查摘要里的每个数字能否在原表或汇总结果中找到。
"""
import re
from pathlib import Path

import pandas as pd

from llm import chat, extract_json
from prompts import AGING_QA, AGING_SUMMARY
from utils import close, parse_amount

HERE = Path(__file__).resolve().parent
OVERDUE = ["逾期1_30天", "逾期31_60天", "逾期61_90天", "逾期90天以上"]
BUCKETS = ["未到期"] + OVERDUE


def load():
    """读入账龄表，并新增两列合计。"""
    df = pd.read_csv(HERE / "aging.csv")
    df["应收合计"] = df[BUCKETS].sum(axis=1)  # axis=1：每一行横向求和
    df["逾期合计"] = df[OVERDUE].sum(axis=1)
    return df


def raw_table(df):
    """发给模型的原始表格：不带我们算好的合计列，让模型自己算。"""
    cols = ["客户编码", "客户名称", "责任销售", "信用期天数"] + BUCKETS
    return df[cols].to_csv(index=False)


def build_questions(df):
    """10 道题，标准答案全部由 pandas 计算，不靠手填。"""
    by_sales = df.groupby("责任销售")["逾期合计"].sum()  # 按责任销售汇总逾期金额
    total = df["应收合计"].sum()
    overdue = df["逾期合计"].sum()
    over_60 = df["逾期61_90天"] + df["逾期90天以上"]
    return [
        ("Q01", "应收账款总额是多少元？", total, "number"),
        ("Q02", "逾期应收合计是多少元？", overdue, "number"),
        ("Q03", "逾期金额占应收总额的百分比是多少（保留 1 位小数）？", round(overdue / total * 100, 1), "percent"),
        ("Q04", "逾期 90 天以上的金额合计是多少元？", df["逾期90天以上"].sum(), "number"),
        ("Q05", "逾期金额合计最大的客户是哪一家？", df.loc[df["逾期合计"].idxmax(), "客户名称"], "text"),
        ("Q06", "逾期 90 天以上金额最大的客户是哪一家？", df.loc[df["逾期90天以上"].idxmax(), "客户名称"], "text"),
        ("Q07", "哪位责任销售名下客户的逾期金额合计最大？", by_sales.idxmax(), "text"),
        ("Q08", "有多少个客户存在逾期？", int((df["逾期合计"] > 0).sum()), "count"),
        ("Q09", "有多少个客户存在 61 天以上的逾期？", int((over_60 > 0).sum()), "count"),
        ("Q10", "王强名下客户的逾期金额合计是多少元？", by_sales["王强"], "number"),
    ]


def judge(answer, truth, kind):
    """判断模型回答是否正确。"""
    if kind == "text":
        return str(truth) in str(answer)
    value = parse_amount(answer)
    if kind == "count":
        return value is not None and int(round(value)) == truth
    if kind == "percent":
        return value is not None and abs(value - truth) <= 0.2
    return close(value, float(truth))


def run_qa(df, version):
    table = raw_table(df)
    rows = []
    for qid, question, truth, kind in build_questions(df):
        reply = chat(AGING_QA[version].format(table=table, question=question))
        data = extract_json(reply)
        answer = data.get("answer") if data else reply
        rows.append({
            "版本": version, "题号": qid, "问题": question,
            "标准答案": truth, "模型回答": answer,
            "是否正确": judge(answer, truth, kind),
        })
    return pd.DataFrame(rows)


# ---------- 摘要的数字追溯检查 ----------

# 数字后面跟着“天、个、家”等单位的，是天数或个数，不是金额，跳过不查
NUM_IN_TEXT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(万元|万|亿|%|元)?\s*([天日月年个家名位条次周])?")


def allowed_values(df):
    """所有“能追溯”的数字：每个单元格、各列合计、按销售的汇总、各类逾期占比。"""
    values = set()
    num = df[BUCKETS + ["应收合计", "逾期合计"]]
    values.update(num.to_numpy().ravel().tolist())
    values.update(num.sum().tolist())
    values.update(df.groupby("责任销售")[["应收合计", "逾期合计"] + OVERDUE].sum().to_numpy().ravel().tolist())
    total = df["应收合计"].sum()
    for col in OVERDUE + ["逾期合计"]:
        values.add(round(df[col].sum() / total * 100, 1))
    values.add(round(df["未到期"].sum() / total * 100, 1))  # 未到期占比，v3 的数据清单里会给出
    values.add(df["逾期61_90天"].sum() + df["逾期90天以上"].sum())
    return [float(v) for v in values if v]


def facts_text(df):
    """v3 用：先用 pandas 把摘要要用的数字算好（单位：万元），模型只负责组织文字。"""
    def w(x):
        return f"{x / 1e4:.1f}万元"

    total = df["应收合计"].sum()
    lines = [
        f"应收合计：{w(total)}",
        f"未到期：{w(df['未到期'].sum())}，占 {df['未到期'].sum() / total * 100:.1f}%",
        f"逾期合计：{w(df['逾期合计'].sum())}，占 {df['逾期合计'].sum() / total * 100:.1f}%",
    ]
    for col in OVERDUE:
        lines.append(f"{col.replace('_', '—')}：{w(df[col].sum())}")
    lines.append("逾期金额前 3 的客户：")
    for _, r in df.nlargest(3, "逾期合计").iterrows():  # nlargest：按逾期合计取前 3 行
        lines.append(
            f"- {r['客户名称']}（{r['客户编码']}，责任销售{r['责任销售']}）："
            f"逾期{w(r['逾期合计'])}，其中 90 天以上{w(r['逾期90天以上'])}"
        )
    return "\n".join(lines)


def check_summary(text, allowed):
    """返回（检查过的数字，找不到出处的数字）。"""
    found, bad = [], []
    for match in NUM_IN_TEXT.finditer(text or ""):
        if match.group(3):
            continue
        raw = match.group(0).strip()
        value = parse_amount(raw)
        # 小于 100 且不是百分数的，多半是序号或天数，跳过
        if value is None or (value < 100 and not raw.endswith("%")):
            continue
        found.append(raw)
        if not any(close(value, a) for a in allowed):
            bad.append(raw)
    return found, bad


def run_summary(df, version, times=3):
    """同一提示词跑 3 次（温度 0.7，模拟日常使用的随机性），统计找不到出处的数字。"""
    table = raw_table(df)
    facts = facts_text(df)
    allowed = allowed_values(df)
    rows = []
    for i in range(times):
        # v1、v2 模板里没有 {facts}，多传的参数会被忽略
        text = chat(AGING_SUMMARY[version].format(table=table, facts=facts), temperature=0.7)
        found, bad = check_summary(text, allowed)
        rows.append({
            "版本": version, "第几次": i + 1, "摘要": text,
            "数字个数": len(found), "未追溯数字个数": len(bad), "未追溯数字": "、".join(bad),
        })
    return pd.DataFrame(rows)
