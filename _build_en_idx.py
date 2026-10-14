# -*- coding: utf-8 -*-
"""搭建英文库：把 general_knowledge.jsonl 全部建为索引（全量检索池），
英文 query 从中召回。模型用 bge-small-en-v1.5。
"""
import json
import os
import sys

from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIR = os.path.join(BASE, "models", "bge-small-en-v1.5")
ENG_JSONL = r"E:\360MoveData\Users\Administrator\Desktop\learning\RAG学习\FlashRAG\examples\quick_start\indexes\general_knowledge.jsonl"

def tokenize(s):
    return s.replace("\\n", "\n")

def main():
    # 1. 载模型
    print("加载模型:", MODEL_NAME, "...")
    model = SentenceTransformer(MODEL_DIR) if os.path.isdir(MODEL_DIR) else SentenceTransformer(MODEL_NAME)
    print("模型就绪")

    # 2. 读全量语料
    with open(ENG_JSONL, encoding="utf-8") as f:
        eng = [json.loads(line) for line in f if line.strip()]
    print("全量语料:", len(eng))

    # 3. 构造块（题目即正文，title 仅作元数据）
    texts = [tokenize(it["contents"]) for it in eng]
    ids = [str(it["id"]) for it in eng]

    # 4. 编码（分批）
    print("编码全部块（这可能要几分钟）...")
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # 5. 落盘：向量 + 元数据
    import numpy as np
    np.save(os.path.join(BASE, "models", "en_index_vectors.npy"), vecs)
    with open(os.path.join(BASE, "eval", "en_meta.jsonl"), "w", encoding="utf-8") as fm:
        for it, vid in zip(eng, ids):
            fm.write(json.dumps({"chunk_id": vid, "title": it["title"]}, ensure_ascii=False) + "\n")
    print("英文索引向量已保存:", vecs.shape)

if __name__ == "__main__":
    main()