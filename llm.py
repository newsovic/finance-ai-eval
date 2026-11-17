"""调用大模型接口。

使用兼容 OpenAI 格式的接口（DeepSeek、通义千问兼容模式等都支持），
通过环境变量配置，避免把密钥写进代码：
    LLM_BASE_URL  接口地址，例如 https://api.deepseek.com
    LLM_API_KEY   你的密钥
    LLM_MODEL     模型名称，例如 deepseek-chat
    DRY_RUN=1     只打印提示词，不真正调用（用来检查流程，不花钱）
"""
import json
import os
import re
import time

import requests

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
DRY_RUN = os.getenv("DRY_RUN") == "1"


def chat(prompt, temperature=0.0, system="你是严谨的财务分析助手。"):
    """发送一条提示词，返回模型回复的文字。"""
    if DRY_RUN:
        print("----- DRY RUN：以下提示词不会真正发送 -----")
        print(prompt[:300], "\n")
        return ""
    if not API_KEY:
        raise RuntimeError("请先设置环境变量 LLM_API_KEY")

    url = BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    # 网络偶尔会失败，最多重试 3 次，每次多等一会儿
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as err:
            print(f"第 {attempt + 1} 次调用失败：{err}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("连续 3 次调用失败，请检查网络、密钥和模型名称")


def extract_json(text):
    """从模型回复中取出 JSON 对象；取不到时返回 None。"""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
