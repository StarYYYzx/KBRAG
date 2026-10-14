# -*- coding: utf-8 -*-
"""生成评测集 eval_set.jsonl（50 条：中文 20 + 英文 30）

设计口径（用户已确认）：
- 正确答案锚点 = 找到正确的块（块级别），同时输出文档级别命中率（命中同 source 文件即算文档命中）。
- 中文库：从 Agent 教科书抽取术语/考点型问题，锚点 chunk_id（可 1~2 个）。
- 英文库：从 FlashRAG quick_start 的 general_knowledge.jsonl 抽 30 条，
  每条恰好是一块（id = 块标识，title = 问句，contents = 答案正文），天然自洽：
  评测问题用该条 title（问题原样），正确块 = 该条 id。
"""
import json
import random

# ---------------------------- 中文 20 条（手工设计） ----------------------------
# (问题, [正确chunk_id], 数据是否依赖该块的正文)
# 选题锚定在术语/考点上，覆盖 01/02/03/04/05/07/08/13/附录A
zh_questions = [
    ("Agent 和 Chatbot 的本质区别是什么？", [11], "01"),
    ("什么是 Agent 的核心模块？它包括哪些部分？", [14, 13], "01"),
    ("Transformer 的核心架构组件有哪些？", [21], "02"),
    ("什么是 Function Calling？它为什么是 Agent 的地基？", [25, 26], "02"),
    ("大模型的 CoT 推理能力是什么？", [24], "02"),
    ("Prompt 设计的基本原则有哪些？", [33], "03"),
    ("什么是 System Prompt？它有什么作用？", [35], "03"),
    ("ReAct 是什么？它与普通问答的本质区别是什么？", [44], "04"),
    ("Agent Loop 是如何实现的？它和条件判断有什么关系？", [45, 46, 47], "04"),
    ("Agent Loop 的工程化要考虑哪些问题（如死循环）？", [48], "04"),
    ("Agent 的记忆分为哪几类？", [55], "05"),
    ("上下文管理的三板斧指的是什么？", [56], "05"),
    ("什么是 MCP？它解决什么问题？", [74, 75], "07"),
    ("MCP 的三大原语是什么？", [75], "07"),
    ("MCP 和 Function Calling 有什么区别？", [80], "07"),
    ("为什么需要多 Agent？单 Agent 的边界在哪？", [86], "08"),
    ("多 Agent 有哪些协作模式？", [88], "08"),
    ("什么是 Agentic RAG？", [144], "13"),
    ("Agent 方向常见的前沿研究方向有哪些？", [143], "13"),
    ("conda 不在 PATH 时如何激活环境？", [152, 153], "附录"),
]

zh_records = []
for q, cids, zone in zh_questions:
    zh_records.append({
        "dataset": "zh_agent",
        "question": q,
        "correct_chunk_ids": [str(c) for c in cids],
        # 文档锚点由 run_eval 用 chunks.json 反查 source_file，这里不硬编码
        "zone": zone,
    })

# ---------------------------- 英文 30 条（固定种子抽样） ----------------------------
english_jsonl = r"E:\360MoveData\Users\Administrator\Desktop\learning\RAG学习\FlashRAG\examples\quick_start\indexes\general_knowledge.jsonl"
with open(english_jsonl, encoding="utf-8") as f:
    eng = [json.loads(line) for line in f if line.strip()]
print("英文语料总条数:", len(eng))

random.seed(42)
sample_idx = sorted(random.sample(range(len(eng)), 30))
print("抽样索引:", sample_idx)

en_records = []
for i in sample_idx:
    item = eng[i]
    # title 本身是问题（带问号），作为评测 query；该条 id 作为一个"块"的锚点
    en_records.append({
        "dataset": "en_knowledge",
        "question": item["title"],
        "correct_chunk_ids": [str(item["id"])],
        # contents 作为参考保留（不参与，仅 inspect 用）
        "_ref_contents": item["contents"][:200],
        "zone": "en",
    })

# ---------------------------- 落盘 ----------------------------
import os
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "eval_set.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for r in zh_records + en_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("写入:", out_path)
print("中文条数:", len(zh_records), "英文条数:", len(en_records), "总计:", len(zh_records) + len(en_records))
# 打印英文前 5 条 question 供检视
print("\n===== 英文前 5 条问题 =====")
for r in en_records[:5]:
    print(" -", r["question"][:90])