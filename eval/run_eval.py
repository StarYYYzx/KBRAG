# -*- coding: utf-8 -*-
"""
run_eval.py —— 评测集打分脚本（增量4 优化版）
=============================================
对 eval_set.jsonl 每条问题跑检索，从【一次 top5 结果】直接推出 @1/@3/@5 与 MRR@5。
对比"无重排 / 有重排"两组，量化重排收益。

指标（块级 + 文档级）：
- 命中率@k = 返回 top-k 中是否含任一 correct_chunk_ids（文档级看 source_file）
- MRR@5    = 正确块在 top-5 首次出现位置倒数，未命中记 0，取均值

两库：
- zh_agent    ：复用 retriever（向量+BM25+可选重排）
- en_knowledge：bge-en 全量索引（纯向量 + 可选重排）
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
EVAL_PATH = os.path.join(BASE, "eval", "eval_set.jsonl")
EN_VEC = os.path.join(BASE, "models", "en_index_vectors.npy")
EN_META = os.path.join(BASE, "eval", "en_meta.jsonl")
TOPK = [1, 3, 5]
FINAL_K = 5

import config

# ---------- 英文库检索 ----------
_en_model = None
_en_vectors = None
_en_meta = None


def _load_en():
    global _en_model, _en_vectors, _en_meta
    from sentence_transformers import SentenceTransformer
    if _en_model is None:
        _en_model = SentenceTransformer(os.path.join(BASE, "models", "bge-small-en-v1.5"))
    if _en_vectors is None:
        _en_vectors = np.load(EN_VEC)
    if _en_meta is None:
        _en_meta = [json.loads(l) for l in open(EN_META, encoding="utf-8")]
    return _en_model, _en_vectors, _en_meta


def _en_candidates(question: str, pool: int):
    model, vecs, meta = _load_en()
    qv = model.encode(question, normalize_embeddings=True).reshape(1, -1)
    scores = vecs @ qv.T
    scores = scores.squeeze()
    order = np.argsort(-scores)[:pool]
    return [{"chunk_id": meta[int(i)]["chunk_id"], "source_file": meta[int(i)]["title"],
             "text": meta[int(i)]["title"]} for i in order]


def en_retrieve(question: str) -> list[dict]:
    if config.USE_RERANK:
        import reranker
        return reranker.rerank(question, _en_candidates(question, config.RERANK_POOL), top_k=FINAL_K)
    return _en_candidates(question, FINAL_K)


# ---------- 中文库检索 ----------
def zh_retrieve(question: str) -> list[dict]:
    import retriever
    r = retriever.retrieve(question, top_k=FINAL_K)
    return [{"chunk_id": str(it["chunk_id"]), "source_file": it["source_file"]} for it in r]


# ---------- 指标计算 ----------
def eval_one(rec, retrieve_fn):
    correct = set(rec["correct_chunk_ids"])
    correct_docs = set(rec.get("_source_files") or [])
    hits = retrieve_fn(rec["question"])

    ids = [h["chunk_id"] for h in hits]
    docs = [h["source_file"] for h in hits]

    mrr_rank = next((i + 1 for i, cid in enumerate(ids) if cid in correct), 0)

    out = {"mrr": 1.0 / mrr_rank if mrr_rank else 0.0}
    for k in TOPK:
        topk_ids, topk_docs = set(ids[:k]), set(docs[:k])
        out[k] = {"recall@k": 1 if (topk_ids & correct) else 0,
                  "doc_recall@k": 1 if ((topk_docs & correct_docs)
                                        or (not correct_docs and topk_ids & correct)) else 0}
    return out


def run(name, records, retrieve_fn):
    n = len(records)
    if n == 0:
        return None
    for r in records:
        r["metrics"] = eval_one(r, retrieve_fn)

    print(f"\n===== {name}  样本数: {n} =====")
    print("指标          @1      @3      @5     MRR@5")
    for metric, label in (("recall@k", "块命中率"), ("doc_recall@k", "文档命中率")):
        row = [f"{sum(1 for r in records if r['metrics'][k][metric])/n*100:5.1f}%" for k in TOPK]
        print(f"{label}   {row[0]}  {row[1]}  {row[2]}")
    mrr = sum(r["metrics"]["mrr"] for r in records) / n
    print(f"MRR@5       {mrr:8.3f}")

    return {"n": n,
            "recall@5": sum(1 for r in records if r["metrics"][5]["recall@k"]) / n * 100,
            "doc_recall@5": sum(1 for r in records if r["metrics"][5]["doc_recall@k"]) / n * 100,
            "mrr": mrr}


def main():
    if not os.path.exists(EVAL_PATH):
        print("找不到", EVAL_PATH); sys.exit(1)

    records = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")]
    zh = [r for r in records if r["dataset"] == "zh_agent"]
    en = [r for r in records if r["dataset"] == "en_knowledge"]
    has_en = os.path.exists(EN_VEC) and os.path.exists(EN_META)

    # 中文库补 source_file 锚点
    zh_meta = json.load(open(os.path.join(BASE, "data", "chunks.json"), encoding="utf-8"))
    id2file = {str(c["chunk_id"]): c["source_file"] for c in zh_meta}
    for r in zh:
        r["_source_files"] = list(dict.fromkeys(id2file[cid] for cid in r["correct_chunk_ids"] if cid in id2file))

    results = {}
    for label, use in (("无重排", False), ("有重排", True)):
        config.USE_RERANK = use
        results["zh_" + label] = run(f"zh_agent {label} (向量+BM25)", zh, zh_retrieve)
        if has_en:
            results["en_" + label] = run(f"en_knowledge {label} (bge-en 向量)", en, en_retrieve)

    # 对比表
    print("\n\n########## 重排收益对比 ##########")
    print(f"{'库/配置':<20}{'块命中@5':>10}{'文档命中@5':>12}{'MRR@5':>10}")
    for key in ("zh_无重排", "zh_有重排", "en_无重排", "en_有重排"):
        if key in results:
            s = results[key]
            name = "英文库" if key.startswith("en") else "中文库"
            conf = key.split("_", 1)[1]
            print(f"{name} {conf:<14}{s['recall@5']:9.1f}%{s['doc_recall@5']:11.1f}%{s['mrr']:10.3f}")


if __name__ == "__main__":
    main()