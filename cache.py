# -*- coding: utf-8 -*-
"""
cache：思路1（多轮检索缓存）

需求背景：
- 多轮对话里用户常"就同一批资料反复追问"。若每轮都重新混合检索+重排，
  重复计算浪费 token 与延迟。
- 缓存近期"改写后 query → 检索结果"，再次命中时零成本复用，提升响应速度与
  省 token。本质是结果缓存（query → 返回块），属性能层，不改准确性。

设计：
- 语义键：用改写后 query 的哈希（保留 query 原文，便于命中判断）。
- LRU：容量上限 CACHE_CAP（默认 64 条），淘汰最久未用的条目。
- 命中回调：记录命中次数，便于观察评估缓存带来的节流收益。

说明：这是对话链路的"附加增强"。准确性由检索本身保证，缓存只做节流。
"""
import time
from collections import OrderedDict

import hashlib


class RetrieverCache:
    """最近检索结果缓存（LRU）。key = 改写后 query 的 sha256 摘要。"""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self.hit = 0
        self.miss = 0

    def _key(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    # ---- 内部：按键存取（单 query 与多 query 两条路径共用同一套 LRU 逻辑）----
    def _get_by_key(self, key: str):
        if key not in self._store:
            self.miss += 1
            return None
        self.hit += 1
        self._store.move_to_end(key)  # 标记为"最近使用"
        return self._store[key]["results"]

    def _put_by_key(self, key: str, results: list[dict]) -> None:
        if key in self._store:
            self._store[key]["results"] = results
            self._store.move_to_end(key)
            return
        self._store[key] = {"ts": time.time(), "results": results}
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # 淘汰最久未用的

    # ---- 单 query 路径（签名保持不变，兼容 run_eval.py 等既有调用方）----
    def get(self, query: str):
        """命中返回检索结果列表并按访问刷新 LRU；未命中返回 None。"""
        return self._get_by_key(self._key(query))

    def put(self, query: str, results: list[dict]) -> None:
        """写入检索结果，超出容量淘汰最久未用者。"""
        self._put_by_key(self._key(query), results)

    # ---- 多 query 路径（增量 8：主链路默认走这条）----
    def _key_multi(self, queries: list[str]) -> str:
        """多 query 的键 = sha256("multi:" + 排序后拼接)。

        排序 ⇒ 与 query 的生成顺序无关，集合相同即命中；
        加 `multi:` 前缀 ⇒ 与单 query 键空间隔离：否则 ["A"] 与单条 "A" 会算出
        同一个键，而两条路径的候选池深度不同、结果可能不一致，会互相污染。
        """
        return self._key("multi:" + "\n".join(sorted(queries)))

    def get_multi(self, queries: list[str]):
        """多 query 路径：命中返回融合结果，未命中返回 None。"""
        return self._get_by_key(self._key_multi(queries))

    def put_multi(self, queries: list[str], results: list[dict]) -> None:
        """多 query 路径：写入融合结果。"""
        self._put_by_key(self._key_multi(queries), results)

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)


# 模块级单例：整个对话会话共享同一个缓存
_cache = RetrieverCache()


def get_cache() -> RetrieverCache:
    return _cache


if __name__ == "__main__":
    c = RetrieverCache(capacity=2)
    r1 = [{"chunk_id": "1"}, {"chunk_id": "2"}]
    c.put("什么是 ReAct？", r1)
    print("命中:", c.get("什么是 ReAct？") == r1, "| miss:", c.miss, "hit:", c.hit)
    c.put("B", [{"chunk_id": "3"}])
    c.put("C", [{"chunk_id": "4"}])  # 溢出淘汰最早"A"
    print("A 被淘汰:", c.get("什么是 ReAct？") is None, "| len:", len(c))