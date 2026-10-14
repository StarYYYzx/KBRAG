# -*- coding: utf-8 -*-
"""下载 BAAI/bge-small-en-v1.5 到本地 models/ 目录（走 hf-mirror 镜像）"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-small-en-v1.5")
print("下载到:", dst)
path = snapshot_download(
    repo_id="BAAI/bge-small-en-v1.5",
    local_dir=dst,
    revision="main",
    local_dir_use_symlinks=False,
)
print("完成:", path)
print("文件:", os.listdir(dst))