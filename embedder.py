# -*- coding: utf-8 -*-
"""
embedder：加载本地 bge-small-zh-v1.5，负责 文本 <-> 向量

设计说明：
- 模型已下载到 KBRAG/models/bge-small-zh-v1.5（离线使用，无需联网）。
- 只加载一次（模块级缓存），避免每次调用都读盘。
- bge 系列模型官方建议：检索时给 query 加指令前缀，以提升效果。
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer
import config

# query 指令前缀（bge-small-zh 官方推荐，用于非对称检索）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(str(config.MODEL_DIR))
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """对一批文本编码，返回向量列表（已归一化）。"""
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """对单个查询编码（带指令前缀），返回向量。"""
    model = get_model()
    vec = model.encode(
        QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.tolist()


if __name__ == "__main__":
    v = embed_query("什么是 ReAct？")
    print("query dim:", len(v))
