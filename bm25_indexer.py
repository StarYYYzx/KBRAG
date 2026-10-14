# -*- coding: utf-8 -*-
"""
bm25_indexer：jieba 分词 + bm25s 稀疏索引

设计说明：
- bm25s 自带的正则词边界 \b\w\w+\b 对中文无效（中文无空格），
  因此中文必须先用 jieba 切词，再以 List[List[str]] 喂给 bm25s。
- 稀疏索引从 chunks.json 重建（构建快、无二进制落盘的中文路径问题），
  故采用"查询前惰性构建并缓存"策略，不持久化二进制。
- 与 FAISS 向量索引并存，供 retriever 做 RRF 混合。
"""
import json
import threading

import bm25s
import jieba

import config


class BM25Index:
    """中文 BM25 稀疏索引：惰性构建 + 线程安全缓存。"""

    def __init__(self):
        self._retriever = None
        self._meta = None
        self._lock = threading.Lock()

    @staticmethod
    def _tokenize(texts: list[str]) -> list[list[str]]:
        return [list(jieba.cut(t)) for t in texts]

    def _ensure_built(self):
        with self._lock:
            if self._retriever is not None:
                return
            meta = json.load(open(config.META_PATH, encoding="utf-8"))
            texts = [m["text"] for m in meta]
            tokens = self._tokenize(texts)
            retriever = bm25s.BM25()
            retriever.index(tokens)
            self._meta = meta
            self._retriever = retriever

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        """BM25 检索 top-k，返回 [{chunk_id, score, ...meta}]。"""
        self._ensure_built()
        q_tokens = self._tokenize([question])
        results, scores = self._retriever.retrieve(q_tokens, k=top_k, show_progress=False)
        docs = []
        for rank, (doc_idx, score) in enumerate(zip(results[0], scores[0])):
            item = dict(self._meta[int(doc_idx)])
            item["bm25_score"] = float(score)
            docs.append(item)
        return docs


# 模块级单例（供 retriever 复用）
_index = BM25Index()


def retrieve(question: str, top_k: int) -> list[dict]:
    return _index.retrieve(question, top_k)