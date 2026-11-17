"""一键运行全部评测，生成明细表和评测报告。

运行：python run_all.py
结果保存在 results 文件夹：
    评测报告.md          汇总指标，可以直接贴到 README
    账龄问答明细.csv      每道题的模型回答和对错
    账龄摘要明细.csv      每次生成的摘要和找不到出处的数字
    返利提取明细.csv      每份合同每个字段的对错
"""
from pathlib import Path

import pandas as pd

import eval_aging
import eval_rebate
from llm import DRY_RUN, MODEL

HERE = Path(__file__).resolve().parent
QA_VERSIONS = ("v1", "v2", "v3")
SUMMARY_VERSIONS = ("v1", "v2", "v3")
REBATE_VERSIONS = ("v1", "v2")  # 返利场景 v2 已基本满足要求，没有做 v3
ALL = ("v1", "v2", "v3")


def pct(series):
    return f"{series.mean() * 100:.0f}%" if len(series) else "—"


def main():
    out = HERE / "results"
    # 部分运行环境（如带文件操作拦截的桌面 AI 工具）在目录已存在时仍会报错，这里兜底
    try:
        out.mkdir(exist_ok=True)
    except FileExistsError:
        pass

    df = eval_aging.load()
    qa = pd.concat([eval_aging.run_qa(df, v) for v in QA_VERSIONS], ignore_index=True)
    summary = pd.concat([eval_aging.run_summary(df, v) for v in SUMMARY_VERSIONS], ignore_index=True)
    rebate = pd.concat([eval_rebate.run(v) for v in REBATE_VERSIONS], ignore_index=True)

    # utf-8-sig：用 Excel 打开时中文不会乱码
    qa.to_csv(out / "账龄问答明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "账龄摘要明细.csv", index=False, encoding="utf-8-sig")
    rebate.to_csv(out / "返利提取明细.csv", index=False, encoding="utf-8-sig")

    def acc(data, v):
        return pct(data[data["版本"] == v]["是否正确"])

    def untraceable(v):
        s = summary[summary["版本"] == v]
        total = s["数字个数"].sum()
        if not total:
            return "—"
        return f"{s['未追溯数字个数'].sum() / total * 100:.0f}%（{s['未追溯数字个数'].sum()}/{total}）"

    lines = [
        "# 财务场景 AI 输出质量评测报告",
        "",
        f"- 模型：{MODEL}",
        f"- 题量：账龄问答 {qa['题号'].nunique()} 题 × {len(QA_VERSIONS)} 版提示词；"
        f"账龄摘要每版 3 次；返利合同 {rebate['合同编号'].nunique()} 份 × 5 个字段 × {len(REBATE_VERSIONS)} 版",
        "",
        "## 总体结果",
        "",
        "| 指标 | v1 | v2 | v3 |",
        "|---|---|---|---|",
        "| 账龄问答准确率 | " + " | ".join(acc(qa, v) for v in ALL) + " |",
        "| 摘要中找不到出处的数字占比 | " + " | ".join(untraceable(v) for v in ALL) + " |",
        "| 返利字段可直接入台账的比例（格式与口径严格一致） | " + " | ".join(acc(rebate, v) for v in ALL) + " |",
        "",
        "## 返利提取：按字段",
        "",
        "| 字段 | v1 | v2 |",
        "|---|---|---|",
    ]
    by_field = rebate.groupby(["字段", "版本"])["是否正确"].mean().unstack("版本")
    for field, row in by_field.iterrows():
        lines.append(f"| {field} | {row.get('v1', 0) * 100:.0f}% | {row.get('v2', 0) * 100:.0f}% |")

    lines += ["", "## 账龄问答：答错的题", "", "| 版本 | 题号 | 问题 | 标准答案 | 模型回答 |", "|---|---|---|---|---|"]
    for _, r in qa[~qa["是否正确"].astype(bool)].iterrows():
        lines.append(f"| {r['版本']} | {r['题号']} | {r['问题']} | {r['标准答案']} | {r['模型回答']} |")

    lines += ["", "## 账龄摘要：找不到出处的数字", "", "| 版本 | 第几次 | 数字个数 | 找不到出处的数字 |", "|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['版本']} | {r['第几次']} | {r['数字个数']} | {r['未追溯数字'] or '无'} |")

    lines += [
        "", "## 返利提取：未通过严格匹配的项（需人工判断意思是否正确）", "",
        "| 版本 | 合同 | 字段 | 标准答案 | 模型输出 |", "|---|---|---|---|---|",
    ]
    for _, r in rebate[~rebate["是否正确"].astype(bool)].iterrows():
        got = str(r["模型输出"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['版本']} | {r['合同编号']} | {r['字段']} | {r['标准答案'].replace('|', '/')} | {got} |")

    report = out / "评测报告.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"完成，报告已保存到 {report}")
    if DRY_RUN:
        print("注意：这是 DRY RUN，没有真正调用模型，报告里的数字没有意义。")


if __name__ == "__main__":
    main()
