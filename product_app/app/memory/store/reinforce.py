"""Usage reinforcement (legacy stub).

The CE Top-k read path no longer ranks by window strength. This module is kept
so old imports do not break; it does not write to memory_window_strength.
"""
from __future__ import annotations

from typing import List

from product_app.app.memory.models import Span
from product_app.app.memory.store.repository import TraceStore


def reinforce_windows(
    store: TraceStore,
    *,
    user_id: int,
    conversation_id: int,
    windows: List[Span],
    now: int | None = None,
) -> None:
    del store, user_id, conversation_id, windows, now
    return
