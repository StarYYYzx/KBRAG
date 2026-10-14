# -*- coding: utf-8 -*-
"""
reranker：交叉编码器（CrossEncoder）重排

增量 4 的设计——为什么用 CrossEncoder 而非 BiEncoder：
- BiEncoder（bge 嵌入）把 query 和文档各自编码成向量再算距离，
  query 与文档"彼此看不见"，信息被压缩进向量后丢失一部分。
- CrossEncoder 把 query 和每个候选文档【拼接成一段】喂进模型，
  query 的注意力能直接看到文档内容（词级交互），判断更准，但只能
  在线对每个候选单独打分，所以必须排在召回之后、对少量候选精排。

典型两段式：
    召回（BiEncoder + BM25，从全库捞 top-RERANK_POOL）→
    精排（CrossEncoder，给这批候选打分取 top-RERANK_FINAL_K）
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import config
from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _load_model() -> CrossEncoder:
    """加载本地 bge-reranker-v2-m3（多语言，中英两库通用）。lru_cache 保证只加载一次。"""
    model_path = Path(config.RERANKER_DIR)
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(
            f"重排模型未就绪: {model_path}\n"
            "请先运行 _download_reranker.py 下载 bge-reranker-v2-m3。"
        )
    return CrossEncoder(str(model_path), max_length=config.RERANK_MAX_LENGTH)


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """对候选块按 query 相关性精排，返回新排序（带 rerank_score）。

    candidates: 每项至少含 text 字段（要喂给模型的文本）。
    top_k: 精排后返回条数，默认 config.RERANK_FINAL_K。
    """
    if not candidates:
        return []
    k = top_k or config.RERANK_FINAL_K
    model = _load_model()

    # CrossEncoder：批量打分,输入 [(query, 文档文本), ...]
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # 打回分数并降序重排
    ordered = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    results = []
    for c, s in ordered[:k]:
        item = dict(c)
        item["rerank_score"] = float(s)
        results.append(item)
    return results