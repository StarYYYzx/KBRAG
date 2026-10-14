# -*- coding: utf-8 -*-
"""
KBRAG 命令行入口

用法：
    python main.py build --kb <目录>        # 扫描 + 切块 + 建索引（默认用 Agent教科书）
    python main.py query "你的问题"          # 问答（需先 build，且 .env 配好 API）
    python main.py query "你的问题" --top-k 3  # 指定召回条数
    python main.py chat                      # 交互式多轮对话（Rewrite → 检索 → 生成）
"""
import argparse
import sys

import config
import graph
import cache


def show_retrieved(retrieved: list, prefix: str = ""):
    """打印召回参考资料（带来源与溯源）。"""
    for i, item in enumerate(retrieved, 1):
        src = f"{item['source_file']}（{item['title']}）"
        loc = item.get("section_path", "")
        loc_txt = f" 溯源:{loc}" if loc else ""
        print(f"{prefix}[{i}] {src}  相似度 {item.get('score', item.get('rrf_score', 0)):.3f}{loc_txt}")


def cmd_build(args):
    kb_dir = args.kb or str(config.KB_DIR_DEFAULT)
    app = graph.build_graph()
    result = app.invoke({"kb_dir": kb_dir})
    print(result.get("status", "build 完成"))


def cmd_query(args):
    # 若未 build 过，提示先建索引
    if not config.INDEX_PATH.exists():
        print("尚未建索引，请先运行: python main.py build --kb <目录>")
        sys.exit(1)
    app = graph.query_graph()
    result = app.invoke({"question": args.question})
    print("\n==== 回答 ====")
    print(result.get("answer", "(无回答)"))
    if result.get("cache_hit"):
        print("（命中检索缓存）")
    print("\n==== 参考资料 ====")
    show_retrieved(result.get("retrieved", []))


def cmd_chat(args):
    """交互式多轮对话：每轮调用 query_graph，把【改写后 query 的原提问】追加进 history。"""
    # 若未 build 过，提示先建索引
    if not config.INDEX_PATH.exists():
        print("尚未建索引，请先运行: python main.py build --kb <目录>")
        sys.exit(1)

    cache.get_cache().clear()  # 新一轮会话清空缓存，避免跨会话污染
    app = graph.query_graph()
    history: list = []  # 已完成的问答对
    print("进入多轮对话（输入 exit / quit 退出）\n")

    while True:
        q = input("你: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("再见")
            break

        # 每次以当前 history 状态跑一次图：内部会做改写 + 检索
        result = app.invoke({"question": q, "history": list(history)})
        answer = result.get("answer", "(无回答)")
        rewritten = result.get("query", q)

        if args.verbose:
            print(f"  [rewrite→] {rewritten}   缓存命中: {result.get('cache_hit')}")
        print("\n助手: " + answer)
        print("\n==== 参考资料 ====")
        show_retrieved(result.get("retrieved", []))

        # 记录一轮问答，供下一轮作为 history
        history = graph_state_collect(history, q, answer)
        print()


def graph_state_collect(history: list, q: str, answer: str) -> list:
    """把一轮问答追加进 history（与 rewriter.collect_history 逻辑一致，避免 import 循环）。"""
    history = list(history or [])
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})
    return history


def main():
    parser = argparse.ArgumentParser(description="个人知识库 RAG 问答（增量 1 MVP）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="扫描目录并建索引")
    p_build.add_argument("--kb", default=None, help="知识库目录（默认 Agent教科书）")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="提问")
    p_query.add_argument("question", help="你的问题")
    p_query.set_defaults(func=cmd_query)

    p_chat = sub.add_parser("chat", help="交互式多轮对话")
    p_chat.add_argument("--verbose", action="store_true", help="显示改写后的 query 与缓存命中")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
