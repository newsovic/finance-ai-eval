"""通用小工具：把各种写法的金额统一成数字，判断两个数是否相等。"""
import re

# 匹配 “125,000元”“12.5万”“17.3%” 这类写法
NUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*(万元|万|亿元|亿|%|元)?")


def parse_amount(text):
    """把 '12.5万'、'125,000元'、125000 统一换算成数字（金额单位：元）。

    百分数只取数字本身，例如 '17.3%' 返回 17.3。取不到数字时返回 None。
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = NUM_RE.search(str(text))
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    unit = match.group(2) or ""
    if unit.startswith("万"):
        value *= 1e4
    elif unit.startswith("亿"):
        value *= 1e8
    return value


def close(a, b, rel=0.005, abs_tol=0.01):
    """两个数相差不超过 0.5%（或 0.01）就算相等，允许四舍五入带来的误差。"""
    if a is None or b is None:
        return False
    return abs(a - b) <= max(abs_tol, rel * abs(b))
