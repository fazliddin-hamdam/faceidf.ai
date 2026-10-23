"""Kalıcı, TTL'li çeviri önbelleği.

Aynı Türkçe tarif tekrar geldiğinde Gemini'ye gitmeden önceki İngilizce
karşılığı döndürür. Böylece ücretsiz kota korunur ve yanıt hızlanır.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable


class TranslationCache:
    """Sözlük tabanlı, LRU + TTL destekli ve JSON'a yazılabilen önbellek."""

    def __init__(
        self,
        path: Path,
        max_size: int = 1024,
        ttl_seconds: int = 7 * 24 * 3600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._clock = clock
        self._items: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._dirty = False
        self._sets_since_save = 0

    def get(self, key: str) -> str | None:
        now = self._clock()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            value, expires_at = item
            if expires_at <= now:
                del self._items[key]
                self._dirty = True
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: str) -> None:
        expires_at = self._clock() + self.ttl_seconds
        with self._lock:
            self._items[key] = (value, expires_at)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)
            self._dirty = True
            self._sets_since_save += 1

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return

        now = self._clock()
        with self._lock:
            for key, item in raw.items():
                if not isinstance(key, str) or not isinstance(item, dict):
                    continue
                value = item.get("value")
                expires_at = item.get("expires_at")
                if not isinstance(value, str) or not isinstance(expires_at, (int, float)):
                    continue
                if expires_at > now:
                    self._items[key] = (value, float(expires_at))
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            data = {
                key: {"value": value, "expires_at": expires_at}
                for key, (value, expires_at) in self._items.items()
            }
            self._dirty = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_path, self.path)
        except OSError:
            # Yazma başarısız olduysa sonraki denemede tekrar yazılsın.
            with self._lock:
                self._dirty = True
            raise

    def maybe_save(self, every: int = 25) -> None:
        """Her istekte disk yazmamak için belirli aralıkla kaydeder."""
        with self._lock:
            if self._sets_since_save < every:
                return
            self._sets_since_save = 0
        self.save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
