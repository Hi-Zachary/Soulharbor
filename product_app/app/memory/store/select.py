"""Decay-aware MMR selection over experience windows."""
from __future__ import annotations

import math
import time
from typing import List, Optional, Set

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import Span
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.store.text_sim import char_bigrams, jaccard, tokens
from product_app.app.memory.store.window_key import earliest_ts, window_key


def _content_bigrams(window: Span) -> Set[str]:
    text = "\n".join(t.content for t in window.messages)
    return char_bigrams(text)


def latest_ts(window: Span) -> int:
    if not window.messages:
        return 0
    return max(int(t.created_at) for t in window.messages)


def week_bucket(window: Span, *, seconds: int = 7 * 86400) -> int:
    return int(latest_ts(window)) // int(seconds)


def content_relevance(query: str, window: Span) -> float:
    """R_content = 2 * fused_RRF + 1.5 * Jaccard(q, W)."""
    q = tokens(query)
    text = "\n".join(t.content for t in window.messages)
    overlap = jaccard(q, tokens(text))
    fused = float(window.fused_score or 0.0)
    return fused * 2.0 + overlap * 1.5


def decay_strength(
    strength: float,
    reinforced_at: int,
    *,
    now: int | None = None,
    w_min: float | None = None,
    tau_d: float | None = None,
) -> float:
    """D_W(t) = w_min + (w - w_min) * exp(-(t - t_W) / tau_d)."""
    w_min = float(mem_cfg.decay_w_min if w_min is None else w_min)
    tau_d = float(mem_cfg.decay_tau_sec if tau_d is None else tau_d)
    w = max(float(strength), w_min)
    if tau_d <= 0:
        return w
    now_i = int(time.time() if now is None else now)
    dt = max(0, now_i - int(reinforced_at))
    return w_min + (w - w_min) * math.exp(-dt / tau_d)


def decayed_relevance(
    query: str,
    window: Span,
    *,
    strength: float,
    reinforced_at: int,
    now: int | None = None,
) -> float:
    """R_decay(W) = R_content(W) * [alpha + (1-alpha) * D_W(t)]."""
    if not mem_cfg.decay_enabled:
        return content_relevance(query, window)
    d = decay_strength(strength, reinforced_at, now=now)
    alpha = float(mem_cfg.decay_alpha)
    return content_relevance(query, window) * (alpha + (1.0 - alpha) * d)


def _window_sim(left: Span, right: Span) -> float:
    return jaccard(_content_bigrams(left), _content_bigrams(right))


def recency_score(window: Span, *, t_ref: int, tau_sec: float | None = None) -> float:
    tau = float(mem_cfg.mmr_recency_tau_sec if tau_sec is None else tau_sec)
    if tau <= 0:
        return 1.0
    dt = max(0, int(t_ref) - latest_ts(window))
    return math.exp(-dt / tau)


def sort_by_time(bundles: List[Span]) -> List[Span]:
    return sorted(bundles, key=lambda w: (earliest_ts(w), w.conversation_id, w.bundle_id))


def select_windows(
    *,
    query: str,
    bundles: List[Span],
    store: Optional[TraceStore] = None,
    user_id: int = 0,
    limit: int | None = None,
    now: int | None = None,
) -> List[Span]:
    """
    Default `mmr`: decay-aware MMR over legacy R_content.
    `topk`: keep upstream order, take first N.
    Legacy alias: `coverage` → `mmr`.
    """
    mode = (mem_cfg.evidence_selection_mode or "mmr").lower()
    if mode == "coverage":
        mode = "mmr"
    limit = int(limit or mem_cfg.bundle_top_k)
    if not bundles:
        return []
    if mode != "mmr":
        return bundles[:limit]

    now_i = int(time.time() if now is None else now)
    strengths: dict[str, tuple[float, int]] = {}
    if store is not None and user_id > 0:
        keys = [window_key(w) for w in bundles]
        strengths = store.get_window_strengths(user_id=user_id, window_keys=keys)

    t_ref = max((latest_ts(w) for w in bundles), default=now_i)
    beta = float(mem_cfg.mmr_recency_beta)
    week_bonus = float(mem_cfg.mmr_week_bonus)
    remaining = list(bundles)
    chosen: List[Span] = []
    seen_weeks: Set[int] = set()
    lam = float(mem_cfg.mmr_lambda)

    while remaining and len(chosen) < limit:
        best_idx = -1
        best_score = -1e9
        for idx, window in enumerate(remaining):
            key = window_key(window)
            w0, t0 = strengths.get(key, (1.0, earliest_ts(window) or now_i))
            rel = decayed_relevance(
                query, window, strength=w0, reinforced_at=t0, now=now_i
            )
            recent = recency_score(window, t_ref=t_ref) if beta > 0 else 0.0
            week = week_bucket(window)
            bucket = week_bonus if (week_bonus > 0 and week not in seen_weeks) else 0.0
            r_eff = rel + beta * recent + bucket
            redundancy = 0.0
            if chosen:
                redundancy = max(_window_sim(window, s) for s in chosen)
            score = lam * r_eff - (1.0 - lam) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        if chosen and best_score < 0.0 and mem_cfg.mmr_stop_on_negative:
            break

        pick = remaining.pop(best_idx)
        chosen.append(pick)
        if week_bonus > 0:
            seen_weeks.add(week_bucket(pick))

    return chosen
