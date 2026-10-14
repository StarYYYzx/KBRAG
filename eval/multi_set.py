# -*- coding: utf-8 -*-
"""多轮追问题（对话链路专用评测）v2 —— 去术语化难点

设计原则（对照实验要有区分度）：
- 追问里【不出现】目标块的关键术语，只有代词/省略（"它""那个""那""这个机制"）。
- 因此孤立地把追问丢给检索器（baseline）必 MISS——没有可独立检索的词。
- 只有改写器结合 history 反推出实体（如"它"→Agent Memory），改写后 query 才
  含可检索术语，才能 HIT。
- 这样的对照组才能真实量化"Rewrite 是否不可或缺"。

评估：rewriter 改写后检索 vs 不改写直接检索，块命中差 = 改写增益。
"""
MULTI_SET = [
    {
        "dataset": "zh_multi",
        "history": [
            {"role": "user", "content": "Agent 和普通 Workflow 有什么区别？"},
            {"role": "assistant", "content": "Agent 能自主决策、调用工具，Workflow 是固定流程。"},
        ],
        "question": "那它的行为是怎么一步步推导出来的？能具体说说那个机制吗？",
        "correct_chunk_ids": ["162"],  # 01章: 02 ReAct 核心思想
        "note": "追问只有'它/那个机制'，无术语；目标实体=ReAct(162)，与历史 Workflow 主题不同需反推",
    },
    {
        "dataset": "zh_multi",
        "history": [
            {"role": "user", "content": "Agent 里记录过去任务经验的东西，存多了会不会爆？"},
            {"role": "assistant", "content": "会，尤其长期记忆积累后需要管理，避免无限增长。"},
        ],
        "question": "那么它具体应该怎么淘汰和清理，有什么边界条件？",
        "correct_chunk_ids": ["167"],  # 01章: 07 Agent Memory 写入策略与遗忘机制
        "note": "'它'→长期记忆，全程无'记忆'关键词；需改写为 Memory 遗忘/淘汰策略",
    },
    {
        "dataset": "zh_multi",
        "history": [
            {"role": "user", "content": "给 Agent 配的工具越多越好吗？"},
            {"role": "assistant", "content": "不是，工具太多模型容易混乱，得靠描述挑对的那个。"},
        ],
        "question": "那描述到底写到多详细才算够，能举例说说规范吗？",
        "correct_chunk_ids": ["164"],  # 01章: 04 Agent 工具集设计与工具描述
        "note": "'描述'= 工具描述，但原句不含'工具'；需改写为工具描述书写规范",
    },
    {
        "dataset": "zh_multi",
        "history": [
            {"role": "user", "content": "编排复杂任务应该用哪个框架？"},
            {"role": "assistant", "content": "如果你要按状态图细粒度控制，那个框架挺合适。"},
        ],
        "question": "你说的那个框架，和拖拽式的相比优势到底在哪？",
        "correct_chunk_ids": ["172"],  # 01章: 12 常见 Agent 框架对比 LangGraph/CrewAI/AutoGen
        "note": "'那个框架/拖拽式'→LangGraph vs 可视化平台(LangFlow)；需反推实体且理解对比语境",
    },
    {
        "dataset": "zh_multi",
        "history": [
            {"role": "user", "content": "Agent 跑着跑着卡住不动了，通常是怎么个死法？"},
            {"role": "assistant", "content": "常见是调工具出错没兜住、自己绕圈子出不来、上下文撑爆。"},
        ],
        "question": "那遇到这种卡死，工程上有哪些抢救和兜底的招？",
        "correct_chunk_ids": ["175"],  # 01章: 15 Agent 为什么经常失败 / 失败模式
        "note": "'这种卡死/抢救兜底'→Agent 失败模式及其降级处理；无术语，需改写",
    },
]