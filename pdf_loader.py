# -*- coding: utf-8 -*-
"""
pdf_loader：解析 PDF（墨圆面试题）→ 按"章/题"切块

为什么单独写，而不是复用 chunk_markdown：
- 墨圆 PDF 是"第X章：... / XX. 问题 / 答案"的文本层级，没有 markdown 的 # 标题。
- 直接复用会退化成"整章一块再字符截断"，丢掉"题号"作为检索单元的粒度，
  也就无法做到增量2要求的"引用溯源到具体章节"。
- 因此按 PDF 自身的上下层级切：先按"第X章"分章，章内再按"题号"分块，
  tile=章名、section_path=章 > 题，text=题目+答案。
"""
from __future__ import annotations
import re
from pathlib import Path
from loader import Chunk

_CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十\d]+章[:：].*)$")
# 主题目：形如 "01. " "12. "（两位数字题号）；追问 "1." "2." 不切块，并入主题目。
_QUESTION_RE = re.compile(r"^(\d{2})\.\s")


def _extract_text(pdf_path) -> str:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def _split_by_chapter(text: str):
    """按"第X章"行切：返回 [(章标题, 章正文)]。"""
    chapters, cur_title, cur_lines = [], None, []
    for line in text.splitlines():
        line = line.rstrip()
        m = _CHAPTER_RE.match(line.strip())
        if m:
            if cur_title:
                chapters.append((cur_title, "\n".join(cur_lines)))
            cur_title = line.strip()
            cur_lines = []
        elif cur_title is not None:
            cur_lines.append(line)
    if cur_title:
        chapters.append((cur_title, "\n".join(cur_lines)))
    return chapters


def _split_chapter_questions(chapter_title: str, chapter_body: str):
    """章内按主题目切：返回 [(题号, 题目标题, 题体)]。"""
    qs, cur_no, cur_title, cur_lines = [], None, None, []
    for line in chapter_body.splitlines():
        line = line.rstrip()
        m = _QUESTION_RE.match(line.strip())
        if m:
            if cur_title:
                qs.append((cur_no, cur_title, "\n".join(cur_lines)))
            cur_no = m.group(1)
            cur_title = line.strip()
            cur_lines = []
        elif cur_title is not None:
            cur_lines.append(line)
    if cur_title:
        qs.append((cur_no, cur_title, "\n".join(cur_lines)))
    return qs


def load_chunks(pdf_path: str | Path, first_id: int = 0) -> list:
    """解析 PDF 为 Chunk 列表（按 章 > 题 层级，每题一个块）。

    - source_file: 文件名
    - title      : 章标题（如 "第一章：Agent"）
    - section_path: "章 > 题号 题目标题"（溯源）
    - text       : "题目标题\n答案"（块即一个完整知识点）
    """
    pdf_path = Path(pdf_path)
    text = _extract_text(pdf_path)
    chunks, cid = [], first_id

    for ch_title, ch_body in _split_by_chapter(text):
        for no, q_title, q_body in _split_chapter_questions(ch_title, ch_body):
            body = q_body.strip()
            if not body:
                continue
            chunks.append(Chunk(
                text=f"{q_title}\n{body}",
                source_file=pdf_path.name,
                title=ch_title,
                section_path=f"{ch_title} > {q_title}",
                chunk_id=cid,
            ))
            cid += 1
    return chunks


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"E:\360MoveData\Users\Administrator\Desktop\learning\墨圆大模型面试必知必会_V1版.pdf"
    cs = load_chunks(p)
    print(f"共 {len(cs)} 个 PDF chunk")
    for c in cs[:12]:
        print(f"[{c.chunk_id}] {c.title} :: {c.section_path}  ({len(c.text)}字)")