"""Usage reinforcement: strengthen windows that were injected into the prompt."""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import Span
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.store.select import decay_strength
from product_app.app.memory.store.window_key import earliest_ts, window_key

logger = logging.getLogger(__name__)


def reinforce_windows(
    store: TraceStore,
    *,
    user_id: int,
    conversation_id: int,
    windows: List[Span],
    now: int | None = None,
) -> int:
    """
    w+ = w- + eta * (1 - w-), where w- is the decayed strength.
    At most once per window per conversation.
    """
    if not mem_cfg.reinforce_enabled or user_id <= 0 or not windows:
        return 0
    now_i = int(time.time() if now is None else now)
    eta = float(mem_cfg.reinforce_eta)
    updated = 0
    for window in windows:
        key = window_key(window)
        row = store.get_window_strength(user_id=user_id, window_key=key)
        if row is not None:
            strength, reinforced_at, last_conv = row
            if int(last_conv) == int(conversation_id):
                continue
            w_minus = decay_strength(strength, reinforced_at, now=now_i)
        else:
            w_minus = decay_strength(
                1.0, earliest_ts(window) or now_i, now=now_i
            )
        w_plus = w_minus + eta * (1.0 - w_minus)
        store.upsert_window_strength(
            user_id=user_id,
            window_key=key,
            strength=w_plus,
            reinforced_at=now_i,
            conversation_id=conversation_id,
        )
        updated += 1
    if updated and mem_cfg.observability:
        logger.info(
            "memory_reinforce user=%s conv=%s windows=%s",
            user_id,
            conversation_id,
            updated,
        )
    return updated
