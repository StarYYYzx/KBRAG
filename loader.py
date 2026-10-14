# -*- coding: utf-8 -*-
"""
loader：扫描目录 + 读取 markdown，按标题层级切块

设计说明：
- 块是"检索单元"，应尽量对齐一个完整知识点。
- 教科书用 # 章、## 节、### 小节组织，默认以 # 起新块；
  若某 # 块超过 CHUNK_MAX_CHARS，则继续按 ## 二次切分。
- 每个 chunk 携带来源元数据：文件名 + 标题路径，供溯源使用。
"""
import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source_file: str        # 文件名，如 "04_核心循环_ReAct与AgentLoop.md"
    title: str              # 该块对应的标题（溯源用），如 "# 04 · 核心循环..."
    section_path: str = ""  # 标题层级路径，如 "章 > 节"
    chunk_id: int = 0

    def to_dict(self):
        return {
            "text": self.text,
            "source_file": self.source_file,
            "title": self.title,
            "section_path": self.section_path,
            "chunk_id": self.chunk_id,
        }


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_by_level(md_text: str, level: int):
    """把 md 文本按指定层级（如 1 或 2）的标题切成若干 (标题, 正文) 段。"""
    sections = []
    cur_title = None
    cur_lines = []
    pattern = re.compile(r"^(#{%d})\s+(.*)$" % level)

    for line in md_text.splitlines():
        m = pattern.match(line)
        if m:
            if cur_title is not None:
                sections.append((cur_title, "\n".join(cur_lines)))
            cur_title = line.strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_title is not None:
        sections.append((cur_title, "\n".join(cur_lines)))
    return sections


def _chunk_long_section(title: str, body: str, max_chars: int):
    """正文超长时按 ## 二次切；每个片段保留所属 # 标题作前缀。"""
    subs = _split_by_level(body, 2)
    if not subs:
        return [(title, body)]
    pieces = []
    for sub_title, sub_body in subs:
        text = sub_body.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            pieces.append((f"{title} | {sub_title}", text))
        else:
            # 仍超长：粗暴按字符切，留作兜底
            for i in range(0, len(text), max_chars):
                pieces.append((f"{title} | {sub_title} (续{i // max_chars + 1})", text[i:i + max_chars]))
    return pieces


def chunk_markdown(md_text: str, source_file: str, max_chars: int = 800):
    """把一篇 md 文档切成 Chunk 列表（按 # 一级标题为界）。"""
    chunks = []
    secs = _split_by_level(md_text, 1)
    if not secs:
        # 文档没有 # 标题：整篇作为一块
        if md_text.strip():
            chunks.append(Chunk(text=md_text.strip(), source_file=source_file, title="(整篇)"))
        return chunks

    for title, body in secs:
        text = body.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append(Chunk(text=f"{title}\n{text}", source_file=source_file, title=title))
        else:
            for sub_title, sub_body in _chunk_long_section(title, text, max_chars):
                if sub_body.strip():
                    chunks.append(Chunk(text=f"{sub_title}\n{sub_body.strip()}",
                                        source_file=source_file, title=sub_title))
    return chunks


def scan_kb(directory) -> list:
    """扫描目录下所有 .md，返回文档 (文件名, 全文) 列表。"""
    docs = []
    for fp in sorted(Path(directory).glob("*.md")):
        text = fp.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            docs.append((fp.name, text))
    return docs


def build_chunks_from_dir(directory, max_chars: int = 800) -> list:
    """扫描目录 -> 每篇按标题切块 -> 汇总所有 Chunk（带全局序号）。"""
    all_chunks = []
    for filename, text in scan_kb(directory):
        for c in chunk_markdown(text, filename, max_chars):
            c.chunk_id = len(all_chunks)
            all_chunks.append(c)
    return all_chunks


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else r"E:\360MoveData\Users\Administrator\Desktop\learning\Agent\Agent教科书"
    chunks = build_chunks_from_dir(d)
    print(f"共 {len(chunks)} 个 chunk")
    for c in chunks[:8]:
        print(f"[{c.chunk_id}] {c.source_file} :: {c.title}  ({len(c.text)} 字符)")
