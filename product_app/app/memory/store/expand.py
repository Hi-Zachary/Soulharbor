"""Expand message and segment retrieval anchors."""
from __future__ import annotations

from typing import List, Optional, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder, cosine_similarity
from product_app.app.memory.models import RetrievalAnchor, SegmentRegion
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.token_utils import count_tokens


def _similarity(query: str, text: str, embedder: MemoryEmbedder) -> float:
    q = (query or "").strip()
    body = (text or "").strip()
    if not q or not body:
        return 0.0
    try:
        qv = embedder.embed(q)
        tv = embedder.embed(body)
        if not qv or not tv:
            return 0.0
        return float(cosine_similarity(qv, tv))
    except Exception:
        return 0.0


def expand_message_anchor(
    store: TraceStore,
    embedder: MemoryEmbedder,
    *,
    user_id: int,
    anchor: RetrievalAnchor,
    query: str,
) -> tuple[List[int], List[str], List[int], List[str], Optional[int], str]:
    """Return earlier/later user message ids+contents and optional reply context."""
    msg = store.get_message(user_id=user_id, message_id=int(anchor.parent_message_id))
    if not msg:
        return [], [], [], [], None, ""

    threshold = float(mem_cfg.expansion_similarity_threshold)
    max_before = int(mem_cfg.message_max_before_users)
    max_after = int(mem_cfg.message_max_after_users)
    max_scan = int(mem_cfg.message_max_scan_distance)

    earlier_ids: List[int] = []
    earlier_contents: List[str] = []
    later_ids: List[int] = []
    later_contents: List[str] = []

    scanned = 0
    pos = int(msg["position"]) - 1
    while scanned < max_scan and len(earlier_ids) < max_before and pos >= 1:
        rows = store.scan_conversation_messages(
            user_id=user_id,
            conversation_id=int(msg["conversation_id"]),
            start_position=pos,
            direction="before",
            limit=1,
        )
        if not rows:
            break
        row = rows[0]
        scanned += 1
        pos = int(row["position"]) - 1
        if str(row["role"]) != "user":
            continue
        if _similarity(query, str(row["content"]), embedder) < threshold:
            break
        earlier_ids.insert(0, int(row["message_id"]))
        earlier_contents.insert(0, str(row["content"]))

    scanned = 0
    pos = int(msg["position"]) + 1
    while scanned < max_scan and len(later_ids) < max_after:
        rows = store.scan_conversation_messages(
            user_id=user_id,
            conversation_id=int(msg["conversation_id"]),
            start_position=pos,
            direction="after",
            limit=1,
        )
        if not rows:
            break
        row = rows[0]
        scanned += 1
        pos = int(row["position"]) + 1
        if str(row["role"]) != "user":
            continue
        if _similarity(query, str(row["content"]), embedder) < threshold:
            break
        later_ids.append(int(row["message_id"]))
        later_contents.append(str(row["content"]))

    reply_id: Optional[int] = None
    reply_content = ""
    if str(anchor.role) == "assistant":
        rid = msg.get("reply_to_message_id")
        if rid:
            reply = store.get_message(user_id=user_id, message_id=int(rid))
            if reply:
                reply_id = int(reply["message_id"])
                reply_content = str(reply["content"])

    return earlier_ids, earlier_contents, later_ids, later_contents, reply_id, reply_content


def expand_segment_anchor(
    store: TraceStore,
    embedder: MemoryEmbedder,
    *,
    user_id: int,
    parent_message_id: int,
    core_segment_ids: List[int],
    query: str,
) -> SegmentRegion:
    segments = store.list_message_segments(user_id=user_id, parent_message_id=int(parent_message_id))
    if not segments:
        return SegmentRegion(
            parent_message_id=int(parent_message_id),
            role="user",
            core_segment_ids=list(core_segment_ids),
            before_segment_ids=[],
            after_segment_ids=[],
            omitted_before=False,
            omitted_after=False,
            token_count=0,
            total_segment_count=0,
        )

    by_id = {int(s["id"]): s for s in segments}
    core = sorted(
        [by_id[sid] for sid in core_segment_ids if sid in by_id],
        key=lambda s: int(s["segment_index"]),
    )
    if not core:
        core = [segments[0]]
    msg = store.get_message(user_id=user_id, message_id=int(parent_message_id))
    role = str((msg or {}).get("role", "user"))
    threshold = float(mem_cfg.expansion_similarity_threshold)
    first_idx = int(core[0]["segment_index"])
    last_idx = int(core[-1]["segment_index"])
    before_ids: List[int] = []
    after_ids: List[int] = []

    if int(mem_cfg.segment_max_before) > 0 and first_idx > 0:
        prev = next((s for s in segments if int(s["segment_index"]) == first_idx - 1), None)
        if prev and _similarity(query, str(prev["content"]), embedder) >= threshold:
            before_ids.append(int(prev["id"]))

    if int(mem_cfg.segment_max_after) > 0 and last_idx < len(segments) - 1:
        nxt = next((s for s in segments if int(s["segment_index"]) == last_idx + 1), None)
        if nxt and _similarity(query, str(nxt["content"]), embedder) >= threshold:
            after_ids.append(int(nxt["id"]))

    included = before_ids + [int(s["id"]) for s in core] + after_ids
    token_count = sum(count_tokens(str(by_id[sid]["content"])) for sid in included if sid in by_id)
    return SegmentRegion(
        parent_message_id=int(parent_message_id),
        role=str(role),
        core_segment_ids=[int(s["id"]) for s in core],
        before_segment_ids=before_ids,
        after_segment_ids=after_ids,
        omitted_before=first_idx > 0 and not before_ids,
        omitted_after=last_idx < len(segments) - 1 and not after_ids,
        token_count=token_count,
        total_segment_count=len(segments),
    )


def build_reply_context_segment_excerpt(
    store: TraceStore,
    embedder: MemoryEmbedder,
    *,
    user_id: int,
    reply_message_id: int,
    query: str,
) -> str:
    msg = store.get_message(user_id=user_id, message_id=int(reply_message_id))
    if not msg:
        return ""
    if not bool(msg.get("has_segments")):
        return str(msg["content"])
    segments = store.list_message_segments(user_id=user_id, parent_message_id=int(reply_message_id))
    if not segments:
        return str(msg["content"])
    scored = [
        (_similarity(query, str(seg["content"]), embedder), int(seg["id"]))
        for seg in segments
    ]
    scored.sort(reverse=True)
    best_id = scored[0][1]
    region = expand_segment_anchor(
        store,
        embedder,
        user_id=user_id,
        parent_message_id=int(reply_message_id),
        core_segment_ids=[best_id],
        query=query,
    )
    by_id = {int(s["id"]): s for s in segments}
    parts = []
    for sid in region.before_segment_ids + region.core_segment_ids + region.after_segment_ids:
        parts.append(str(by_id[sid]["content"]))
    return "\n".join(parts)
