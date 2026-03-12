"""Disk-based JSON cache with TTL and per-source rate limiting."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from citation_tree.config import CACHE_DIR, RATE_LIMIT

# Cile-system cache keyed by MD5 hash of a string key
class Cache:

    def __init__(self, directory: str = CACHE_DIR, ttl_days: int = 7):
        self.dir = directory
        self.ttl = ttl_days * 86400

    def _path(self, key: str) -> str:
        return os.path.join(
            self.dir, hashlib.md5(key.encode()).hexdigest() + ".json"
        )

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("_ts", 0) < self.ttl:
                return data.get("v")
        except Exception:
            pass
        return None

    def set(self, key: str, value: Any):
        try:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump({"_ts": time.time(), "v": value}, f)
        except Exception:
            pass

# Rate limiter to ensure we don't exceed API limits via calls to the API
class RateLimiter:

    def __init__(self, interval: float = RATE_LIMIT):
        self.interval = interval
        self.last = 0.0

    def wait(self):
        delta = time.time() - self.last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last = time.time()
