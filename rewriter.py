# -*- coding: utf-8 -*-
"""
rewriter：多轮对话的 Query Reformulation（LLM 改写）

为什么需要：
- 检索器只接受"语义自足"的一句话。而追问往往依赖上文（指代"它/这个/那"、
  省略主语、承接上一话题），孤立地把它丢给 retriever 会检索不到正确块。
- 所以在 retrieve 之前，先用 LLM 把 [最近 N 轮历史, 当前追问] 改写为一句
  语义自足的新 query，再进入检索链路。

改动设计（决策点 1/2/3/4 用户确认）：
- 改写器：LLM（DeepSeek，复用 generator 的 API 配置）
- 历史窗口：近 N 轮滑动（N=HISTORY_WINDOW，默认 2），控制 token 成本
- 不聚合历史检索块：每轮基于改写后的新 query 独立检索（后续可加聚合）
"""
from openai import OpenAI
import config

HISTORY_WINDOW = 2  # 改写时引用最近几轮（含本轮之前的对话）
MULTI_N = 3         # Multi-Query：生成几条 query（1 条主改写句 + 2 组关键词）


def _num_turns(history: list[dict]) -> int:
    return len(history)


def rewrite(question: str, history: list[dict] | None = None) -> str:
    """把 [可选历史, 追问] 改写成语义自足的一句 query。

    history: 形如 [{"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}, ...]
    无需历史时（history 为空/过短）原样返回，避免无谓的 LLM 调用。
    """
    history = history or []
    # 只取最近 HISTORY_WINDOW 个"问答对"：把扁平历史切成 [q,a] 对再取后 N 对
    pairs = []
    i = 0
    turns = list(history)
    while i + 1 <= len(turns):
        q = turns[i]
        a = turns[i + 1] if i + 1 < len(turns) else None
        pairs.append((q, a))
        i += 2
    recent_pairs = pairs[-HISTORY_WINDOW:]

    # 无历史 或 历史不足一轮时，追问本身已自足，不画蛇添足
    if not recent_pairs:
        return question

    history_block = []
    for idx, (q, a) in enumerate(recent_pairs, 1):
        qtext = q.get("content", "") if isinstance(q, dict) else str(q)
        atext = a.get("content", "") if a and isinstance(a, dict) else ("assistant: " + str(a) if a else "")
        history_block.append(f"问{idx}: {qtext}")
        if atext:
            history_block.append(f"答{idx}: {atext}")
    history_text = "\n".join(history_block)

    prompt = f"""下面是用户与知识库助手的最近对话。请把用户的"当前问题"改写成一句
**语义自足、不依赖上下文、可直接用于文档检索**的完整问题：
- 把"它/这个/那个/上面说的"等指代替换成具体对象名；
- 补全省略的主语、术语；
- 保留并精确复用原文中出现的技术名词（如 ReAct、RAG、LoRA）不要随意改写专业词；
- 只输出改写后的一句话，不要加解释、不要加引号。

最近对话：
{history_text}

当前问题：{question}
改写结果："""

    cfg = config.get_api_config()
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": "你是一个多轮对话改写器，输出一行改写后的检索 query。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    rewritten = resp.choices[0].message.content.strip()
    return rewritten


def rewrite_multi(question: str, history: list[dict] | None = None) -> list[str]:
    """Multi-Query：把 [历史, 追问] 生成为 MULTI_N 条 query（列表）。

    结构：
      1 条主改写句（语义自足、含技术名词）
      + (MULTI_N-1) 条关键词组（给 BM25 路命中精确术语）

    LLM 以「Q1 … /K1 …」格式输出 3 行，解析为列表。
    无历史且追问已自足时降级为单条（避免无谓调用）。
    """
    history = history or []
    if not history:
        return [question]

    # 复用 rewrite 的历史格式化逻辑（简化：直接拼最近 N 轮）
    pairs = []
    turns = list(history)
    i = 0
    while i + 1 <= len(turns):
        q = turns[i]
        a = turns[i + 1] if i + 1 < len(turns) else None
        pairs.append((q, a))
        i += 2
    recent_pairs = pairs[-HISTORY_WINDOW:]
    if not recent_pairs:
        return [question]

    hist_lines = []
    for idx, (q, a) in enumerate(recent_pairs, 1):
        qtext = q.get("content", "") if isinstance(q, dict) else str(q)
        atext = a.get("content", "") if a and isinstance(a, dict) else ("assistant: " + str(a) if a else "")
        hist_lines.append(f"问{idx}: {qtext}")
        if atext:
            hist_lines.append(f"答{idx}: {atext}")
    history_text = "\n".join(hist_lines)

    prompt = f"""下面是用户与知识库助手的最近对话。请围绕用户的"当前问题"输出 {MULTI_N} 行，
用来做【文档检索】的多样化查询。目标是让这 {MULTI_N} 条覆盖不同检索角度，提高命中概率：
- 第 1 行必须以 Q 开头：一条语义自足、补全指代与省略、保留技术名词的完整问题（主改写句）。
- 其余行以 K 开头：各一组精确关键词或短语（不含标点，2-5 个词），专让精确检索/BM25 命中技术术语。
- 技术名词（如 ReAct、RAG、LoRA、CoT、LangGraph）必须在关键词组中原样保留。
- 只输出这些行，不要解释；每行一个检索单位。

最近对话：
{history_text}

当前问题：{question}

输出格式：
Q ...(主改写句)
K ...(关键词组1)
K ...(关键词组2)"""

    cfg = config.get_api_config()
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": "你是一个多轮对话改写器，输出多样化检索查询。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()

    # 解析：取 Q/K 开头行
    queries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[:1] in ("Q", "K"):
            rest = line[1:].lstrip(" .：:，,、\t").strip()
            if rest:
                queries.append(rest)
    # 兜底：没解析出就以主改写句 + 原问题
    if not queries:
        queries = [rewrite(question, history)] if history else [question]
    return queries[:MULTI_N]



def collect_history(history: list[dict], user_msg: str, assistant_msg: str) -> list[dict]:
    """把一轮新问答追加进 history（langgraph state 用）。"""
    history = list(history or [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    return history


if __name__ == "__main__":
    # 冒烟测试：含历史与不含历史
    hist = [
        {"role": "user", "content": "什么是 ReAct？"},
        {"role": "assistant", "content": "ReAct 是 Reasoning + Acting 的交替循环。"},
    ]
    print("改写（带历史）:", rewrite("那它和 CoT 什么关系？", hist))
    print("改写（无历史）:", rewrite("LoRA 的原理是什么？"))