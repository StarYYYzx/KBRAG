# -*- coding: utf-8 -*-
"""
run_eval_multi.py —— 对话链路（Rewrite 单query vs Multi-Query）专项评测
============================================================
对 multi_set 的每条追问题,三种方式各跑一次检索:
  A) 不改写        : 把追问直接丢检索器（baseline）
  B) 单 query 改写 : rewrite() 得一句,再检索
  C) Multi-Query   : rewrite_multi() 得 N 条,retrieve_multi() 融合

指标:
- 块命中@5 / MRR@5（正确块进 top5、排第几）
- 成本: 改写 LLM 触发次数、检索次数、耗时（近似）
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
FINAL_K = 5

import config
import retriever
import rewriter


def _retrieve_ref(queries: list[str]) -> list[str]:
    """Multi-Query 检索，返回 top5 chunk_id 列表。"""
    r = retriever.retrieve_multi(queries, top_k=FINAL_K)
    return [str(it["chunk_id"]) for it in r]


def _eval(rec, mode):
    t0 = time.time()
    if mode == "no_rewrite":
        queries = [rec["question"]]
        llm_calls = 0
    elif mode == "single":
        queries = [rewriter.rewrite(rec["question"], rec["history"])]
        llm_calls = 1
    else:  # multi
        queries = rewriter.rewrite_multi(rec["question"], rec["history"])
        llm_calls = 1 if queries else 0
    ids = _retrieve_ref(queries)
    dt = time.time() - t0
    correct = set(rec["correct_chunk_ids"])
    rank = next((i + 1 for i, c in enumerate(ids) if c in correct), 0)
    return {
        "query": queries,
        "llm_calls": llm_calls,
        "retrieve_n": len(queries),
        "time": dt,
        "hit": bool(rank),
        "mrr": 1.0 / rank if rank else 0.0,
        "rank": rank,
    }


def main():
    sys.path.insert(0, os.path.join(BASE, "eval"))
    from multi_set import MULTI_SET

    modes = [("no_rewrite", "不改写(baseline)"), ("single", "单query改写"), ("multi", "Multi-Query(3条)")]
    n = len(MULTI_SET)
    summary = []  # (name, hit_rate, mrr, llm, retrieve_per_q)
    for mode, name in modes:
        hits = mrrs = 0.0
        llm = retr = 0
        print(f"\n===== zh_multi  {name}  样本: {n} =====")
        for rec in MULTI_SET:
            out = _eval(rec, mode)
            hits += out["hit"]
            mrrs += out["mrr"]
            llm += out["llm_calls"]
            retr += out["retrieve_n"]
            flag = "HIT" if out["hit"] else "MISS"
            print(f"  [{flag}] rank={out['rank']:>1}  retrieve_n={out['retrieve_n']}  {rec['question'][:22]}")
            for q in out["query"]:
                print(f"         query: {q[:56]}")
        hit_rate = hits / n * 100
        mrr = mrrs / n
        print(f"  → 命中@5: {hits:.0f}/{n} = {hit_rate:.1f}%   MRR@5: {mrr:.3f}")
        print(f"  → 成本: LLM改写调用{llm}次 / 本组共检索{retr}次")
        summary.append((name, hit_rate, mrr, llm, retr / n))

    # 成本对比汇总
    print("\n\n########## 成本与准确度汇总 ##########")
    print(f"{'模式':<20}{'命中@5':>9}{'MRR@5':>9}{'改写LLM次':>9}{'检索次/题':>10}")
    for name, hit_rate, mrr, llm, rpq in summary:
        print(f"{name:<20}{hit_rate:>8.1f}%{mrr:>9.3f}{llm:>9}{rpq:>9.1f}")


if __name__ == "__main__":
    main()