"""Turn retrieved windows into prompt lines."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import Span, ProfileItem, SpanTurn
from product_app.app.memory.store.text_sim import cosine, jaccard, tokens

# Prefer full text. Only snip extreme long messages.
# Global <memory> token budget still drops whole lines when needed.
_SNIP_DEFAULT = 800
_SNIP_SEED = 1000

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")
_CLAUSE_RE = re.compile(r"[^，、；;\n]+[，、；;\n]?")

# Process-local sentence embedding cache (sha1 → dense vec).
_SENT_VEC_CACHE: Dict[str, List[float]] = {}
_SENT_VEC_CACHE_MAX = 2048


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(str(mem_cfg.memory_timezone or "Asia/Shanghai"))
    except Exception:
        return ZoneInfo("Asia/Shanghai")


def current_date_label() -> str:
    """Today's calendar date in the configured memory timezone."""
    return datetime.now(_tz()).strftime("%Y-%m-%d")


def _date_label(ts: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=_tz()).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _relevance(turn: SpanTurn, query: str) -> float:
    q_tokens = tokens(query)
    overlap = jaccard(tokens(turn.content), q_tokens) if q_tokens else 0.0
    bonus = 0.25 if turn.is_anchor else 0.0
    return overlap * 2.0 + bonus


def _split_sentences(text: str) -> List[str]:
    """Chinese punctuation split; fall back to clauses for one long block."""
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.findall(raw) if len(p.strip()) >= 4]
    if len(parts) <= 1 and len(raw) > 120:
        parts = [p.strip() for p in _CLAUSE_RE.findall(raw) if len(p.strip()) >= 4]
    return parts or [raw]


def _tail_snip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "…" + text[-limit:]


def _flat(text: str) -> str:
    return (text or "").strip().replace("\n", " ")


def _snip_around_match(text: str, matched_chunk: str, *, max_chars: int) -> str:
    """Keep a local span that includes the retrieval hit chunk."""
    cleaned = _flat(text)
    chunk = _flat(matched_chunk)
    limit = max(64, int(max_chars))
    if len(cleaned) <= limit:
        return cleaned
    if not chunk:
        return _tail_snip(cleaned, limit)

    idx = cleaned.find(chunk)
    if idx < 0:
        return _tail_snip(cleaned, limit)

    chunk_len = len(chunk)
    if chunk_len >= limit:
        snippet = cleaned[idx : idx + limit]
        prefix = "…" if idx > 0 else ""
        suffix = "…" if idx + limit < len(cleaned) else ""
        return f"{prefix}{snippet}{suffix}"

    remaining = limit - chunk_len
    left = remaining // 2
    right = remaining - left
    start = max(0, idx - left)
    end = min(len(cleaned), idx + chunk_len + right)
    if end - start < limit:
        if start == 0:
            end = min(len(cleaned), start + limit)
        elif end == len(cleaned):
            start = max(0, end - limit)

    snippet = cleaned[start:end]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(cleaned) else ""
    return f"{prefix}{snippet}{suffix}"


def _match_in_content(content: str, matched_chunk: Optional[str]) -> bool:
    if not matched_chunk:
        return False
    if matched_chunk in content:
        return True
    return _flat(matched_chunk) in _flat(content)


def _cache_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _cached_sentence_vectors(
    sentences: Sequence[str],
    embedder: MemoryEmbedder,
) -> List[List[float]]:
    """Embed sentences with a small process-local LRU-ish cache."""
    out: List[Optional[List[float]]] = [None] * len(sentences)
    missing: List[Tuple[int, str]] = []
    for i, sent in enumerate(sentences):
        key = _cache_key(sent)
        cached = _SENT_VEC_CACHE.get(key)
        if cached is not None:
            out[i] = cached
        else:
            missing.append((i, sent))

    if missing:
        vectors = embedder.embed_batch([s for _, s in missing])
        for (i, sent), vec in zip(missing, vectors):
            out[i] = list(vec or [])
            if len(_SENT_VEC_CACHE) >= _SENT_VEC_CACHE_MAX:
                # Drop an arbitrary old entry (dict preserves insertion order).
                _SENT_VEC_CACHE.pop(next(iter(_SENT_VEC_CACHE)))
            _SENT_VEC_CACHE[_cache_key(sent)] = out[i]  # type: ignore[assignment]

    return [v or [] for v in out]


def _span_from_scores(
    sentences: Sequence[str],
    scores: Sequence[float],
    limit: int,
) -> str:
    """Take top-1/2 sentences, pad with neighbors, keep original order under budget."""
    if not sentences:
        return ""
    if len(sentences) == 1:
        return _tail_snip(sentences[0], limit)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = ranked[:2]
    # If #2 is far from #1 and much weaker, keep only the best + neighbors.
    if len(top) == 2:
        i0, i1 = top[0], top[1]
        far = abs(i0 - i1) > 2
        weak = scores[i1] < 0.55 * max(scores[i0], 1e-6)
        if far and weak:
            top = [i0]

    start = max(0, min(top) - 1)
    end = min(len(sentences), max(top) + 2)
    chosen = list(range(start, end))

    # Shrink from the lower-scoring side until under budget.
    while chosen and sum(len(sentences[i]) for i in chosen) > limit:
        if len(chosen) == 1:
            return _tail_snip(sentences[chosen[0]], limit)
        left, right = chosen[0], chosen[-1]
        if scores[left] <= scores[right]:
            chosen.pop(0)
        else:
            chosen.pop()

    snippet = "".join(sentences[i] for i in chosen).strip()
    prefix = "…" if chosen and chosen[0] > 0 else ""
    suffix = "…" if chosen and chosen[-1] + 1 < len(sentences) else ""
    return f"{prefix}{snippet}{suffix}"


def _score_sentences(
    query: str,
    sentences: Sequence[str],
    embedder: MemoryEmbedder,
    *,
    query_vec: Optional[List[float]] = None,
) -> List[float]:
    if not sentences:
        return []
    try:
        qv = query_vec if query_vec is not None else list(embedder.embed(query) or [])
        if not qv:
            return []
        svs = _cached_sentence_vectors(sentences, embedder)
        return [cosine(qv, sv) for sv in svs]
    except Exception:
        return []


def _snip_for_query(
    text: str,
    query: str,
    *,
    max_chars: int,
    embedder: Optional[MemoryEmbedder] = None,
    query_vec: Optional[List[float]] = None,
    sentences: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
) -> str:
    """
    Prefer full text. Only when over max_chars: sentence-level embedding
    similarity to locate the query-relevant local span (bi-encoder / BGE).
    On embed failure → keep the tail (conclusions often land late).
    """
    cleaned = (text or "").strip().replace("\n", " ")
    limit = max(64, int(max_chars))
    if len(cleaned) <= limit:
        return cleaned

    units = sentences if sentences is not None else _split_sentences(cleaned)
    if not units:
        return _tail_snip(cleaned, limit)

    if scores is None:
        model = embedder or MemoryEmbedder.shared()
        scores = _score_sentences(query, units, model, query_vec=query_vec)

    if not scores or max(scores) <= 0.0:
        return _tail_snip(cleaned, limit)

    return _span_from_scores(units, scores, limit)


def _pick_user_turns(
    window: Span,
    *,
    query: str = "",
    max_msgs: int = 6,
) -> List[SpanTurn]:
    """Keep user lines only; hard-retain anchors, then fill with relevant neighbors."""
    users = [turn for turn in window.messages if turn.role == "user"]
    if not users:
        return []

    limit = max(1, int(max_msgs))
    anchors = [turn for turn in users if turn.is_anchor]
    others = [turn for turn in users if not turn.is_anchor]

    # Never silently drop anchors when a merged window carries many of them.
    limit = max(limit, len(anchors))

    chosen = list(anchors)
    remaining = limit - len(chosen)
    if remaining > 0:
        others.sort(
            key=lambda turn: _relevance(turn, query),
            reverse=True,
        )
        chosen.extend(others[:remaining])

    return sorted(chosen, key=lambda turn: turn.position)


def _budget_for(turn: SpanTurn) -> int:
    return _SNIP_SEED if turn.is_anchor else _SNIP_DEFAULT


def _lines_for_window(
    window: Span,
    *,
    query: str = "",
    max_msgs: int = 6,
    embedder: Optional[MemoryEmbedder] = None,
    query_vec: Optional[List[float]] = None,
    snip_plan: Optional[Dict[int, Tuple[List[str], List[float]]]] = None,
) -> List[str]:
    lines: List[str] = []
    for turn in _pick_user_turns(window, query=query, max_msgs=max_msgs):
        budget = _budget_for(turn)
        if turn.is_anchor and _match_in_content(turn.content, turn.matched_chunk):
            text = _snip_around_match(
                turn.content,
                turn.matched_chunk or "",
                max_chars=budget,
            )
        else:
            plan = (snip_plan or {}).get(id(turn))
            if plan is not None:
                sents, scores = plan
                text = _snip_for_query(
                    turn.content,
                    query,
                    max_chars=budget,
                    sentences=sents,
                    scores=scores,
                )
            else:
                text = _snip_for_query(
                    turn.content,
                    query,
                    max_chars=budget,
                    embedder=embedder,
                    query_vec=query_vec,
                )
        day = _date_label(turn.created_at)
        star = "★ " if turn.is_anchor else ""
        when = f"记录于 {day}：" if day else ""
        lines.append(f"- {star}{when}用户：{text}")
        lines.append(
            f"  来源：conversation={window.conversation_id}, "
            f"message={turn.message_id}, pos={turn.position}"
        )
    return lines


def _earliest_ts(window: Span) -> int:
    if not window.messages:
        return 0
    return min(t.created_at for t in window.messages)


def _prepare_snip_batch(
    windows: Sequence[Span],
    *,
    query: str,
    embedder: MemoryEmbedder,
) -> Tuple[Optional[List[float]], Dict[int, Tuple[List[str], List[float]]]]:
    """
    One query embed + one sentence embed_batch for all long user messages
    that will actually be formatted.
    """
    jobs: List[Tuple[int, List[str]]] = []  # turn id(obj) → sentences
    all_sentences: List[str] = []

    for window in windows:
        for turn in _pick_user_turns(window, query=query):
            if turn.is_anchor and _match_in_content(turn.content, turn.matched_chunk):
                continue
            cleaned = (turn.content or "").strip().replace("\n", " ")
            if len(cleaned) <= _budget_for(turn):
                continue
            sents = _split_sentences(cleaned)
            jobs.append((id(turn), sents))
            all_sentences.extend(sents)

    if not jobs:
        return None, {}

    try:
        query_vec = list(embedder.embed(query) or [])
        if not query_vec:
            return None, {}
        # One batched embed for every sentence across long messages.
        _cached_sentence_vectors(all_sentences, embedder)
        plan: Dict[int, Tuple[List[str], List[float]]] = {}
        for turn_key, sents in jobs:
            scores = _score_sentences(query, sents, embedder, query_vec=query_vec)
            plan[turn_key] = (sents, scores)
        return query_vec, plan
    except Exception:
        return None, {}


def format_sections(
    *,
    bundles: List[Span],
    profiles: List[ProfileItem],
    query: str = "",
    embedder: Optional[MemoryEmbedder] = None,
) -> List[str]:
    # No episodic evidence → empty (profiles injected separately as <user_profile>).
    if not bundles:
        return []

    parts: List[str] = [f"当前日期：{current_date_label()}。"]
    model = embedder
    query_vec: Optional[List[float]] = None
    snip_plan: Dict[int, Tuple[List[str], List[float]]] = {}

    if bundles:
        # Chronological injection: no explicit cross-session chain_id.
        ordered = sorted(bundles, key=lambda w: (_earliest_ts(w), w.conversation_id))
        # Batch semantic snips only when at least one message exceeds the soft cap.
        needs_snip = any(
            not (t.is_anchor and _match_in_content(t.content, t.matched_chunk))
            and len((t.content or "").strip()) > _budget_for(t)
            for w in ordered
            for t in _pick_user_turns(w, query=query)
        )
        if needs_snip:
            model = model or MemoryEmbedder.shared()
            query_vec, snip_plan = _prepare_snip_batch(ordered, query=query, embedder=model)

        parts.append("[相关经历证据]")
        for window in ordered:
            parts.extend(
                _lines_for_window(
                    window,
                    query=query,
                    embedder=model,
                    query_vec=query_vec,
                    snip_plan=snip_plan,
                )
            )
            parts.append("")

    while parts and parts[-1] == "":
        parts.pop()
    return parts
