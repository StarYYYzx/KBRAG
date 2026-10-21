# -*- coding: utf-8 -*-
"""
graph：langgraph 编排两条链路

设计说明：
- langgraph 只负责"流程"（节点顺序 + 状态传递），节点内部算法由各模块负责。
- 两条链路：
    1) build   : scan -> chunk -> embed -> index   （建索引，一次性）
    2) query   : retrieve -> generate              （问答，可多次）
- 每个节点函数签名：fn(state: dict) -> dict（返回要写回 state 的字段）。
"""
from typing import TypedDict
from langgraph.graph import StateGraph

import config
import embedder
import indexer
import loader
import pdf_loader
import retriever
import generator
import rewriter
import cache


class KBState(TypedDict, total=False):
    # build 链路
    kb_dir: str
    chunks: list          # list[Chunk]
    vectors: list         # list[list[float]]
    status: str
    # query 链路（含多轮对话）
    question: str          # 当前用户输入（可能是追问）
    history: list          # list[dict]，形如 [{"role","content"},...] 已完成的历史问答
    query: str             # 改写后的语义自足 query（主句，= queries[0]）
    queries: list          # 增量8：Multi-Query 生成的 query 列表（单 query 时长度为 1）
    retrieved: list        # list[dict]
    cache_hit: bool        # 思路1：是否命中检索缓存
    answer: str


# ---------- build 链路节点 ----------
def node_scan(state: KBState) -> dict:
    """扫描 md 目录 + PDF 源 -> 按各自结构切块 -> 合并、统一编号。"""
    chunks = loader.build_chunks_from_dir(state["kb_dir"], config.CHUNK_MAX_CHARS)

    # 增量2：追加 PDF 数据源块
    for pdf_path in getattr(config, "PDF_SOURCES", []):
        if not pdf_path.exists():
            continue
        pdf_chunks = pdf_loader.load_chunks(pdf_path, first_id=len(chunks))
        chunks.extend(pdf_chunks)

    return {"chunks": chunks, "status": f"切块完成，共 {len(chunks)} 块"}


def node_embed(state: KBState) -> dict:
    """对所有 chunk 向量化。"""
    texts = [c.text for c in state["chunks"]]
    vectors = embedder.embed_texts(texts)
    return {"vectors": vectors}


def node_index(state: KBState) -> dict:
    """构建 FAISS 索引并落盘。"""
    index = indexer.build_index(state["chunks"], state["vectors"])
    indexer.save_index(index, state["chunks"])
    return {"status": f"索引已保存: {config.INDEX_PATH}"}


def build_graph():
    """编译 build 链路：scan -> embed -> index"""
    g = StateGraph(KBState)
    g.add_node("scan", node_scan)
    g.add_node("embed", node_embed)
    g.add_node("index", node_index)
    g.set_entry_point("scan")
    g.add_edge("scan", "embed")
    g.add_edge("embed", "index")
    g.add_edge("index", "__end__")
    return g.compile()


# ---------- query 链路节点 ----------
def node_rewrite(state: KBState) -> dict:
    """多轮改写：把 [近N轮历史, 追问] 改写成语义自足 query。

    增量 8：默认走 Multi-Query（1 主改写句 + 2 关键词组）；关闭开关时退回单 query。
    queries[0] 始终是主改写句，所以 state["query"] 保持"单句"语义不变。
    无历史时 rewrite_multi 自行降级为单条，不产生多余 LLM 调用。"""
    if config.MULTI_QUERY:
        queries = rewriter.rewrite_multi(state["question"], state.get("history"))
        return {"queries": queries, "query": queries[0]}
    query = rewriter.rewrite(state["question"], state.get("history"))
    return {"queries": [query], "query": query}


def node_retrieve(state: KBState) -> dict:
    """混合检索 + 缓存。优先命中缓存，未命中才跑检索。

    增量 8：queries 长度 > 1 时走 retrieve_multi（各 query 深池召回后融合），
    缓存键同步换成"多条 query 排序拼接"，避免多路结果被单路旧缓存顶掉。"""
    queries = state.get("queries") or [state["query"]]
    rc = cache.get_cache()

    if len(queries) > 1:
        cached = rc.get_multi(queries)
        if cached is not None:
            return {"retrieved": cached, "cache_hit": True}
        retrieved = retriever.retrieve_multi(queries)
        rc.put_multi(queries, retrieved)
        return {"retrieved": retrieved, "cache_hit": False}

    cached = rc.get(queries[0])
    if cached is not None:
        return {"retrieved": cached, "cache_hit": True}
    retrieved = retriever.retrieve(queries[0])
    rc.put(queries[0], retrieved)
    return {"retrieved": retrieved, "cache_hit": False}


def node_generate(state: KBState) -> dict:
    """组装 prompt 调 LLM 生成回答。"""
    answer = generator.generate(state["question"], state["retrieved"])
    return {"answer": answer}


def query_graph():
    """编译 query 链路：rewrite（多轮）→ retrieve → generate"""
    g = StateGraph(KBState)
    g.add_node("rewrite", node_rewrite)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "__end__")
    return g.compile()


if __name__ == "__main__":
    # 冒烟测试：验证两条链路可编译、可跑通（不调 LLM 的部分）
    bg = build_graph()
    print("build 链路编译 OK")
    qg = query_graph()
    print("query 链路编译 OK")
