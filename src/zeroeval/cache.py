from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Generic, Hashable, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    def __init__(self, ttl_seconds: float = 60.0, maxsize: int = 512) -> None:
        self._ttl = float(ttl_seconds)
        self._maxsize = int(maxsize)
        self._data: "OrderedDict[K, Tuple[float, V]]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: K) -> Optional[V]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            ts, value = item
            if now - ts > self._ttl:
                # Expired
                self._data.pop(key, None)
                return None
            # Move to end (LRU)
            self._data.move_to_end(key)
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


