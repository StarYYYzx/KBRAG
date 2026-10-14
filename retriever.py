# -*- coding: utf-8 -*-
"""
retriever：混合检索（向量 + BM25）→ RRF 融合排序

增量 3 的设计：
- 两条路并行：
    1) 向量路：bge 语义相似度（召回"意思相近"）
    2) BM25 路：jieba 关键词频（召回"字面出现 X"）
- 因为两路分数量纲不同（向量 0~1、BM25 可到几十上百），不能直接相加，
  故用 RRF（Reciprocal Rank Fusion）：只看名次，score = sum 1/(k + rank)，k=60。
- 每个候选同时带两路得分与 rank，便于调试与后续对照。
"""
import config
import embedder
import indexer
import bm25_indexer
import reranker

RRF_K = 60       # RRF 常量（bm25s 官方默认推荐 60）
HYBRID_POOL = 20 # 每路召回候选数（融合池），再截断到 top_k


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """rank 从 0 开始计，按 1/(k + rank + 1) 累加。"""
    return 1.0 / (k + rank + 1)


def _hybrid_candidates(question: str) -> list[dict]:
    """向量 + BM25 两路 → RRF 融合，返回**未精排**的全量候选（按 rrf_score 降序）。

    拆成独立函数，供 retrieve 与 retrieve_multi 复用：Multi-Query 需要在
    融合前拿到"每条 query 的深池"，而不是被 top5/精排提前截断。
    """
    # 1) 向量路
    qvec = embedder.embed_query(question)
    vec_index, meta = indexer.load_index()
    v_scores, v_indices = indexer.search(vec_index, qvec, HYBRID_POOL)

    # 2) BM25 路
    bm_docs = bm25_indexer.retrieve(question, HYBRID_POOL)

    # 3) RRF 融合
    acc: dict[int, dict] = {}  # chunk_id -> 累计分数与信息

    # 向量路
    for rank, idx in enumerate(v_indices):
        if idx < 0:
            continue
        mid = int(idx)
        mid_score = _rrf_score(rank)
        if mid not in acc:
            item = dict(meta[mid])
            for c in ("vec_rank", "vec_score", "bm25_rank", "bm25_score"):
                item[c] = None
            item["rrf_score"] = 0.0
            acc[mid] = item
        acc[mid]["rrf_score"] += mid_score
        acc[mid]["vec_rank"] = rank
        acc[mid]["vec_score"] = float(v_scores[rank])

    # BM25 路
    for rank, doc in enumerate(bm_docs):
        mid = doc["chunk_id"]
        mid_score = _rrf_score(rank)
        if mid not in acc:
            for c in ("vec_rank", "vec_score", "bm25_rank", "bm25_score"):
                doc[c] = None
            doc["rrf_score"] = 0.0
            acc[mid] = doc
        item = acc[mid]
        item["rrf_score"] += mid_score
        item["bm25_rank"] = rank
        item["bm25_score"] = doc["bm25_score"]

    # 4) RRF 总分排序 → 候选池
    return sorted(acc.values(), key=lambda x: x["rrf_score"], reverse=True)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """两段式检索：

    1) 混合检索（向量 + BM25 → RRF 融合）：从全库捞回候选。
    2) 若启用重排：用 CrossEncoder 精排这批候选，取 RERANK_FINAL_K（默认 5）。
       否则直接取 top_k。

    每个返回块含 text/source_file/title，若有重排还含 rerank_score。
    """
    k = (top_k or config.TOP_K) if not config.USE_RERANK else None
    rrf_sorted = _hybrid_candidates(question)

    if config.USE_RERANK:
        return reranker.rerank(question, rrf_sorted, top_k=config.RERANK_FINAL_K)
    return rrf_sorted[:k]


def retrieve_multi(queries: list[str], top_k: int | None = None) -> list[dict]:
    """Multi-Query 检索：对多条 query 各做**深池召回**后跨 query 融合。

    融合排序优先级：出现次数（跨几条 query 命中）→ RRF 名次分（在各自 query
    里的排序前置度）→ 重排分数。

    为什么这么改（增量 7 修复）：
    旧版每条 query 先截成 top5 再按纯出现次数投票：一个"只有专精关键词能命中、
    且在它那条 query 里排 #1"的正确块，票数只有 1，会和无数的 count=1 噪声块
    打平票被挤出 top5（实测案例：块 162 被关键词 query 排在第 1，仍掉出融合结果）。
    修复 = ① 融合前深池召回 MQ_POOL=20，不让正确块在下注前被截掉；
    ② 平票时用 RRF 名次分仲裁——排 #1 的专精块名次分远高于排在 4、5 位的泛化块。
    """
    k = top_k or config.TOP_K
    acc: dict[int, dict] = {}  # chunk_id -> 结果 dict（含 _count / _rrf）

    for q in queries:
        # 深池召回：拿到该 query 的未精排 top-MQ_POOL（重排开启时也走同一池）
        rrf_sorted = _hybrid_candidates(q)[:config.MQ_POOL]
        for rank, item in enumerate(rrf_sorted):
            cid = item["chunk_id"]
            if cid not in acc:
                acc[cid] = dict(item)
                acc[cid]["_count"] = 0
                acc[cid]["_rrf"] = 0.0
            acc[cid]["_count"] += 1
            acc[cid]["_rrf"] += _rrf_score(rank)

    def _key(kv):
        rank = kv[1].get("rerank_score")
        return (kv[1]["_count"], kv[1]["_rrf"], rank if rank is not None else float("-inf"))

    ordered = sorted(acc.items(), key=_key, reverse=True)
    selected = [kv[1] for kv in ordered[:k]]

    # 重排开启时，对并集统一精排一次（只做 1 次，优于每 query 各精排 3 次）
    if config.USE_RERANK and selected and queries:
        return reranker.rerank(queries[0], selected, top_k=config.RERANK_FINAL_K)
    return selected