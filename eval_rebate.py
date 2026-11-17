"""场景二：供应商返利条款提取（对应采购财务“审核供应商返利条款”）。

6 份模拟合同，每份有 5 个字段的标准答案，按字段逐一打分。
其中 R04、R06 故意设置了陷阱：原文没有写的内容，模型不应该编出来。
"""
import re
from pathlib import Path

import pandas as pd

from llm import chat, extract_json
from prompts import REBATE, REBATE_FIELDS
from utils import close, parse_amount

HERE = Path(__file__).resolve().parent
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DATE = re.compile(r"(\d{4})[-年/.](\d{1,2})[-月/.](\d{1,2})")


def is_none(value):
    return value is None or str(value).strip() in ("", "未约定", "无", "null", "None")


def norm_period(value):
    """把“按季度结算”“每季”等说法统一成 月度/季度/半年度/年度。"""
    if is_none(value):
        return "未约定"
    text = str(value)
    for key, name in (("半年", "半年度"), ("季", "季度"), ("月", "月度"), ("年", "年度")):
        if key in text:
            return name
    return text


def dates(value):
    return ["%s-%02d-%02d" % (y, int(m), int(d)) for y, m, d in DATE.findall(str(value))]


def score(field, got, expected):
    if expected == "未约定":
        return is_none(got)
    if is_none(got):
        return False
    if field == "返利比例":
        return sorted(float(x) for x in PCT.findall(str(got))) == sorted(float(x) for x in PCT.findall(expected))
    if field == "门槛金额":
        return close(parse_amount(got), parse_amount(expected))
    if field == "结算周期":
        return norm_period(got) == expected
    if field == "兑现条件与方式":
        return all(key in str(got) for key in expected.split("|"))
    if field == "有效期":
        return dates(got) == dates(expected)
    return False


def run(version):
    contracts = pd.read_csv(HERE / "rebate_contracts.csv", dtype=str).fillna("")
    rows = []
    for _, contract in contracts.iterrows():
        reply = chat(REBATE[version].format(text=contract["合同条款原文"]))
        data = extract_json(reply) or {}
        for field in REBATE_FIELDS:
            got = data.get(field)
            expected = contract["期望_" + field]
            rows.append({
                "版本": version, "合同编号": contract["合同编号"], "字段": field,
                "标准答案": expected, "模型输出": got,
                "是否正确": score(field, got, expected),
            })
    return pd.DataFrame(rows)
