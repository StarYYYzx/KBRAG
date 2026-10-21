# -*- coding: utf-8 -*-
"""
KBRAG 配置模块
- 路径集中管理
- 从 .env 读取 API 配置（本地文件，不提交云端）
"""
import os
from pathlib import Path

# 项目根目录（本文件所在目录）
BASE_DIR = Path(__file__).resolve().parent

# 数据与模型路径
# 知识库目录：优先读环境变量 KBRAG_KB_DIR，否则回退到项目内 kb/（自带语料不入库，需自备）
KB_DIR_DEFAULT = Path(os.environ.get("KBRAG_KB_DIR") or (BASE_DIR / "kb"))
# PDF 数据源：环境变量 KBRAG_PDF_SOURCES 传多个路径，用 os.pathsep 分隔（Windows 为 ';'）
PDF_SOURCES = [
    Path(p) for p in os.environ.get("KBRAG_PDF_SOURCES", "").split(os.pathsep) if p.strip()
]
MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"
RERANKER_DIR = BASE_DIR / "models" / "bge-reranker-v2-m3"

# FAISS 索引：Windows 下 faiss C++ 层无法写含中文的路径，
# 故索引放到纯 ASCII 的用户缓存目录；chunks.json 仍放项目内（Python 读写无碍）。
import os as _os
_ASCII_CACHE = Path(_os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "kbrag"
_ASCII_CACHE.mkdir(parents=True, exist_ok=True)
INDEX_PATH = _ASCII_CACHE / "index.faiss"
META_PATH = BASE_DIR / "data" / "chunks.json"

# 检索与分块参数
CHUNK_MAX_CHARS = 800     # 单块最大字符数，超过则按 ## 二次切
EMBED_BATCH_SIZE = 16
TOP_K = 5

# Multi-Query 融合（增量 7 修复）：每条 query 的先取多深再并集。
# 之前每条只投 top5，"仅单 query 命中、且在被它命中 query 里排 #1"的正确块会被平票噪声挤掉；
# 故融合前先深池召回 MQ_POOL 条，融合排序用（出现次数, RRF名次分）。
MQ_POOL = 20              # 每条 query 进入融合前的候选深度

# Multi-Query 主链路开关（增量 8）：默认开启，query/chat 走"多 query 检索 + 融合"。
# 关闭（KBRAG_MULTI_QUERY=0）时退回单 query 改写与检索，便于对照与提速。
MULTI_QUERY = os.environ.get("KBRAG_MULTI_QUERY", "1") == "1"

# ---------- 重排序（增量 4，方案 2：大池召回 → rerank 精排） ----------
USE_RERANK = os.environ.get("KBRAG_USE_RERANK", "1") == "1"  # 是否启用重排（可用环境变量覆盖，便于评测提速）
RERANK_POOL = 20                   # 召回到"重排池"的候选数（这里是 top20，不同于混合检索 HYBRID_POOL=20）
RERANK_FINAL_K = 5                 # 重排后最终输出的条数（喂给生成器）
RERANK_MAX_LENGTH = 512            # CrossEncoder 截断长度（够用，省显存）

# ---------- API 配置：从 .env 读取 ----------
def load_env():
    """读取 KBRAG/.env（若存在）。格式 OPENAI_BASE_URL=... / OPENAI_API_KEY=..."""
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_api_config():
    load_env()
    return {
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }
