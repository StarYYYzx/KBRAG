# -*- coding: utf-8 -*-
"""
generator：调用 OpenAI 兼容 API 生成回答

设计说明：
- base_url / api_key / model 都从 KBRAG/.env 读取（config.get_api_config）。
- prompt 把召回块组装成"参考资料"，要求模型基于资料回答并标注出处。
"""
from openai import OpenAI
import config


def _build_prompt(question: str, retrieved: list[dict]) -> str:
    """组装 prompt：参考资料 + 问题。retrieved 为含 text/source_file/title 的块。"""
    parts = []
    for i, item in enumerate(retrieved, 1):
        source = f"{item['source_file']}（{item['title']}）"
        parts.append(f"[资料{i}] 出处: {source}\n{item['text']}")
    context = "\n\n".join(parts)
    return f"""你是一个个人知识库问答助手。请只根据下面的参考资料回答问题。

{context}

问题：{question}

要求：
1. 如果参考资料足以回答，请直接回答，并在句末用〔出处：文件名（标题）〕标注来源。
2. 如果资料不足以回答，请明确说"资料库中未找到相关信息"，不要编造。
"""


def generate(question: str, retrieved: list[dict], temperature: float = 0.2) -> str:
    cfg = config.get_api_config()
    if not cfg["api_key"]:
        raise RuntimeError("未配置 OPENAI_API_KEY，请在 KBRAG/.env 中填写")

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": "你是一个严谨的知识库问答助手，回答必须基于给定资料。"},
            {"role": "user", "content": _build_prompt(question, retrieved)},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content
