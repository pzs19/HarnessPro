from __future__ import annotations

import time
from contextlib import contextmanager
from threading import Condition
from typing import Iterator

DEFAULT_CODEX_CONCURRENCY = 30


class CodexCliPool:
    def __init__(self, *, max_concurrency: int = DEFAULT_CODEX_CONCURRENCY) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._max_concurrency = max_concurrency
        self._condition = Condition()
        self._in_flight = 0
        self._queued = 0
        self._submitted = 0
        self._completed = 0
        self._total_wait_time_s = 0.0

    @property
    def max_concurrency(self) -> int:
        with self._condition:
            return self._max_concurrency

    def configure(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        with self._condition:
            if self._in_flight or self._queued:
                raise RuntimeError(
                    "cannot reconfigure Codex CLI pool while requests are active "
                    f"(in_flight={self._in_flight}, queued={self._queued})"
                )
            self._max_concurrency = max_concurrency
            self._condition.notify_all()

    @contextmanager
    def acquire(self) -> Iterator[None]:
        start = time.monotonic()
        admitted = False
        with self._condition:
            self._submitted += 1
            self._queued += 1
            try:
                while self._in_flight >= self._max_concurrency:
                    self._condition.wait()
                self._queued -= 1
                self._in_flight += 1
                self._total_wait_time_s += time.monotonic() - start
                admitted = True
            except BaseException:
                if not admitted:
                    self._queued -= 1
                    self._condition.notify_all()
                raise

        try:
            yield
        finally:
            with self._condition:
                self._in_flight -= 1
                self._completed += 1
                self._condition.notify_all()

    def snapshot(self) -> dict[str, int | float]:
        with self._condition:
            return {
                "max_concurrency": self._max_concurrency,
                "in_flight": self._in_flight,
                "queued": self._queued,
                "submitted": self._submitted,
                "completed": self._completed,
                "total_wait_time_s": self._total_wait_time_s,
            }


_global_codex_pool = CodexCliPool()


def global_codex_pool() -> CodexCliPool:
    return _global_codex_pool


def configure_global_codex_pool(max_concurrency: int) -> None:
    _global_codex_pool.configure(max_concurrency)


def _reset_global_codex_pool_for_tests() -> None:
    global _global_codex_pool
    _global_codex_pool = CodexCliPool()
