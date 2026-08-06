"""Stable keys / timestamps for experience windows."""
from __future__ import annotations

import hashlib
from typing import Sequence

from product_app.app.memory.models import Span


def window_key(window: Span) -> str:
    """Deterministic id from conversation + message ids (order-independent)."""
    mids = sorted(int(t.message_id) for t in window.messages)
    raw = f"{int(window.conversation_id)}:" + ",".join(str(m) for m in mids)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def earliest_ts(window: Span) -> int:
    if window.fragment is not None:
        return int(window.fragment.created_at)
    if not window.messages:
        return 0
    return min(int(t.created_at) for t in window.messages)


def message_ids(window: Span) -> Sequence[int]:
    return [int(t.message_id) for t in window.messages]
