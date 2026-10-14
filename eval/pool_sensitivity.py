# -*- coding: utf-8 -*-
"""
pool_sensitivity.py —— 池敏感性实验（高效版）
=============================================
关键优化：cross-encoder 给【同一个候选】的精排分数与"池大小"无关（池只决定
可选候选的数量）。因此只需对每条查询用最大池召回并精排一次，记录每个候选
的 rerank 分；之后对任意 pool，只要"取前 pool 个候选并按 rerank 分取 top5"，
即可得到该池的 MRR / 命中率，完全等价却只精排一次。

步骤（对每条中文题）：
  1. 混合检索（向量+BM25→RRF）召回 MAX_POOL 个候选
  2. CrossEncoder 精排全部候选，记录每候选 rerank_score
  3. 对每个 pool：取前 pool 个候选 → 按 rerank_score 取 top5 → 算 MRR/命中
结论：MRR 随 pool 增大先升、后在某点收敛。收敛点 = 该用多大精排池。
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
EVAL_PATH = os.path.join(BASE, "eval", "eval_set.jsonl")
POOLS = [8, 12, 16, 20, 24, 28]
FINAL_K = 5
MAX_POOL = max(POOLS)

import config
config.USE_RERANK = False   # 混合检索直接召回（无内建重排）

import retriever
import reranker


def main():
    records = [r for r in (json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")) if r["dataset"] == "zh_agent"]
    n = len(records)

    # 对每条题：一次召回 + 一次精排，记录 rerank 排序
    per_q = []  # 每条: top 候选列表(含 chunk_id, rerank_score) 按 rerank 降序
    for rec in records:
        correct = set(rec["correct_chunk_ids"])
        cands = retriever.retrieve(rec["question"], top_k=MAX_POOL)
        ordered = reranker.rerank(rec["question"], cands, top_k=MAX_POOL)  # 全排
        per_q.append((correct, ordered))

    print(f"{'池大小':>6}{'MRR@5':>10}{'命中@5':>10}   {'说明'}")
    print("-" * 52)
    prev = None
    for pool in POOLS:
        total_mrr, hit5 = 0.0, 0
        for correct, ordered in per_q:
            top_ids = [str(c["chunk_id"]) for c in ordered[:pool][:FINAL_K]]
            rank = next((i + 1 for i, cid in enumerate(top_ids) if cid in correct), 0)
            if rank:
                total_mrr += 1.0 / rank
                if rank <= 5:
                    hit5 += 1
        mrr, hit = total_mrr / n, hit5 / n * 100
        note = ""
        if prev is not None:
            delta = mrr - prev
            note = "收敛" if abs(delta) < 0.005 else ("仍在上升" if delta > 0 else "下降(噪声)")
        print(f"{pool:>6}{mrr:>10.4f}{hit:>9.1f}%   {note}")
        prev = mrr


if __name__ == "__main__":
    main()