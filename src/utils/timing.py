"""Timing helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    """Simple wall-clock timer holding the elapsed seconds."""

    elapsed: float = 0.0


@contextmanager
def timed() -> Iterator[Timer]:
    """Context manager that records elapsed wall time in the yielded Timer."""
    t = Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start
