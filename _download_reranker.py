# -*- coding: utf-8 -*-
"""重试下载 bge-reranker-v2-m3：禁用 Xet 通道（镜像上 Xet 401）"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"   # 关键：绕开 xethub 加速通道

from huggingface_hub import snapshot_download

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-reranker-v2-m3")
print("禁用 Xet，重试下载到:", dst)
path = snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir=dst,
    revision="main",
)
print("完成:", path)
print("文件:", os.listdir(dst))