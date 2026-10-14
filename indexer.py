# -*- coding: utf-8 -*-
"""
indexer：FAISS 向量索引 + chunk 元数据 的构建与存取

设计说明：
- 两样东西分开存：
    1) FAISS 索引（纯向量，`index.faiss`）—— 负责"谁和 query 最像"
    2) chunks.json（元数据）—— 负责"索引里第 i 条对应哪块文本/哪篇文档"
  检索时 FAISS 返回的是"向量下标"，再用下标查 chunks.json 拿回文本与来源。
- FAISS 只管相似度检索，不管存储与溯源——这就是"自实现 vs FAISS"选型时
  说的那句"落盘与文档映射要自己写"落地的位置。
"""
import json
import numpy as np
import faiss
import config
from loader import Chunk


def build_index(chunks: list[Chunk], vectors: list[list[float]]) -> faiss.Index:
    """用向量列表构建 FAISS 索引（内积相似度，向量已归一化）。"""
    dim = len(vectors[0])
    mat = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(dim)  # 归一化向量用内积 = 余弦相似度
    index.add(mat)
    return index


def save_index(index: faiss.Index, chunks: list[Chunk]) -> None:
    """索引与元数据落盘。"""
    config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.META_PATH, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=1)


def load_index():
    """从磁盘加载索引与元数据。"""
    index = faiss.read_index(str(config.INDEX_PATH))
    with open(config.META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def search(index: faiss.Index, query_vec: list[float], top_k: int):
    """检索 top-k，返回 (scores, indices)。"""
    q = np.asarray([query_vec], dtype="float32")
    scores, indices = index.search(q, top_k)
    return scores[0], indices[0]
