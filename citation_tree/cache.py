"""Disk-based JSON cache with TTL and per-source rate limiting."""

from __future__ import annotations

import hashlib
import json
import os
import time
from threading import BoundedSemaphore, Lock
from typing import Any

from citation_tree.config import CACHE_DIR, GLOBAL_HTTP_MAX_CONCURRENCY, RATE_LIMIT

# File-system cache keyed by MD5 hash of a string key
class Cache:

    def __init__(self, directory: str = CACHE_DIR, ttl_days: int = 7):
        self.dir = directory
        self.ttl = ttl_days * 86400
        self._lock = Lock()

    # creating a file path by hashing the key
    def _path(self, key: str) -> str:
        return os.path.join(
            self.dir, hashlib.md5(key.encode()).hexdigest() + ".json"
        )
    
    # retrieves a value from the cache if ti exists and if it's not expired
    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with self._lock:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if time.time() - data.get("_ts", 0) < self.ttl:
                return data.get("v")
        except Exception:
            pass
        return None

    # adds a value to the cache with the current timestamp
    def set(self, key: str, value: Any):
        try:
            with self._lock:
                with open(self._path(key), "w", encoding="utf-8") as f:
                    json.dump({"_ts": time.time(), "v": value}, f)
        except Exception:
            pass

# Rate limiter to ensure we don't exceed API limits via calls to the API
class RateLimiter:

    def __init__(self, interval: float = RATE_LIMIT):
        self.interval = interval
        self.last = 0.0
    
    # waits until the minimum interval has passed since the last call
    def wait(self):
        delta = time.time() - self.last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last = time.time()


# global gate to limit total concurrency and rate of all HTTP requests across the app, to avoid overwhelming APIs or hitting local resource limits
class GlobalRequestGate:

    _sem = BoundedSemaphore(max(1, GLOBAL_HTTP_MAX_CONCURRENCY))
    _lock = Lock()
    _last_by_group: dict[str, float] = {}
    
    # ensures that calls to the same API group are spaced out by at least min_interval seconds
    @classmethod
    def _wait_group_interval(cls, group: str, min_interval: float):
        if min_interval <= 0:
            return
        with cls._lock:
            now = time.time()
            last = cls._last_by_group.get(group, 0.0)
            delay = min_interval - (now - last)
            if delay > 0:
                time.sleep(delay)
                now = time.time()
            cls._last_by_group[group] = now
    
    # waits if there are too many concurrent requests, and ensures that calls to the same API group are spaced out by at least min_interval seconds
    @classmethod
    def request(cls, http_client, method: str, url: str, *, group: str, min_interval: float, **kwargs,):
        cls._sem.acquire()
        try:
            cls._wait_group_interval(group, min_interval)
            return http_client.request(method, url, **kwargs)
        finally:
            cls._sem.release()
