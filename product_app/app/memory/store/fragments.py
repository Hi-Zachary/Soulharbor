"""Build retrieved fragments from ranked anchors."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import RankedHit, RetrievalAnchor, RetrievedFragment, SegmentRegion
from product_app.app.memory.store.expand import (
    build_reply_context_segment_excerpt,
    expand_message_anchor,
    expand_segment_anchor,
)
from product_app.app.memory.store.repository import TraceStore
from product_app.app.memory.token_utils import count_tokens, truncate_by_tokens


def _fragment_token_count(fragment: RetrievedFragment) -> int:
    total = 0
    if fragment.fragment_type == "message":
        core = fragment.core_message_content or (
            fragment.core_contents[0] if fragment.core_contents else ""
        )
        total += count_tokens(core)
    else:
        total += sum(count_tokens(x) for x in fragment.core_contents)
    total += sum(count_tokens(x) for x in fragment.expanded_contents)
    total += sum(count_tokens(x) for x in fragment.earlier_user_contents)
    total += sum(count_tokens(x) for x in fragment.later_user_contents)
    if fragment.reply_context_content:
        total += count_tokens(fragment.reply_context_content)
    return total


def _snapshot_fragment(
    fragment: RetrievedFragment,
    *,
    later_contents: List[str],
    later_ids: List[int],
    earlier_contents: List[str],
    earlier_ids: List[int],
    reply_content: str,
    core_message: str,
    core_contents: List[str],
    expanded_contents: List[str],
    region: SegmentRegion | None,
    expanded_unit_ids: List[int],
) -> RetrievedFragment:
    return RetrievedFragment(
        fragment_type=fragment.fragment_type,
        anchor_role=fragment.anchor_role,
        parent_message_id=fragment.parent_message_id,
        score=fragment.score,
        core_unit_ids=list(fragment.core_unit_ids),
        expanded_unit_ids=list(expanded_unit_ids),
        reply_context_message_id=fragment.reply_context_message_id,
        earlier_user_message_ids=list(earlier_ids),
        later_user_message_ids=list(later_ids),
        omitted_before=fragment.omitted_before,
        omitted_after=fragment.omitted_after,
        token_count=0,
        conversation_id=fragment.conversation_id,
        created_at=fragment.created_at,
        segment_region=region,
        core_contents=list(core_contents),
        expanded_contents=list(expanded_contents),
        reply_context_content=reply_content,
        earlier_user_contents=list(earlier_contents),
        later_user_contents=list(later_contents),
        core_message_content=core_message,
    )


def trim_fragment_to_budget(fragment: RetrievedFragment) -> RetrievedFragment:
    """Trim expansions before core; core segments are never removed."""
    budget = (
        int(mem_cfg.max_message_fragment_tokens)
        if fragment.fragment_type == "message"
        else int(mem_cfg.max_segment_fragment_tokens)
    )

    later_contents = list(fragment.later_user_contents)
    later_ids = list(fragment.later_user_message_ids)
    earlier_contents = list(fragment.earlier_user_contents)
    earlier_ids = list(fragment.earlier_user_message_ids)
    reply_content = str(fragment.reply_context_content or "")
    core_message = str(fragment.core_message_content or "")
    core_contents = list(fragment.core_contents)
    expanded_contents = list(fragment.expanded_contents)
    region = fragment.segment_region
    expanded_unit_ids = list(fragment.expanded_unit_ids)

    while True:
        current = _snapshot_fragment(
            fragment,
            later_contents=later_contents,
            later_ids=later_ids,
            earlier_contents=earlier_contents,
            earlier_ids=earlier_ids,
            reply_content=reply_content,
            core_message=core_message,
            core_contents=core_contents,
            expanded_contents=expanded_contents,
            region=region,
            expanded_unit_ids=expanded_unit_ids,
        )
        if _fragment_token_count(current) <= budget:
            current.token_count = _fragment_token_count(current)
            return current

        if fragment.fragment_type == "segment":
            if region and region.after_segment_ids:
                dropped = region.after_segment_ids.pop()
                if len(expanded_contents) > len(region.before_segment_ids):
                    expanded_contents.pop()
                expanded_unit_ids = [x for x in expanded_unit_ids if x != dropped]
                continue
            if region and region.before_segment_ids:
                dropped = region.before_segment_ids.pop(0)
                if expanded_contents:
                    expanded_contents.pop(0)
                expanded_unit_ids = [x for x in expanded_unit_ids if x != dropped]
                continue
            if reply_content:
                reply_content = truncate_by_tokens(
                    reply_content,
                    max_tokens=int(mem_cfg.max_reply_context_tokens),
                )
                continue
            break

        if later_contents:
            later_contents.pop()
            later_ids.pop()
            continue
        if earlier_contents:
            earlier_contents.pop(0)
            earlier_ids.pop(0)
            continue
        if reply_content:
            reply_content = truncate_by_tokens(
                reply_content,
                max_tokens=int(mem_cfg.max_reply_context_tokens),
            )
            continue
        if core_message:
            core_message = truncate_by_tokens(core_message, max_tokens=max(32, budget // 2))
            core_contents = [core_message]
            continue
        break

    final = _snapshot_fragment(
        fragment,
        later_contents=later_contents,
        later_ids=later_ids,
        earlier_contents=earlier_contents,
        earlier_ids=earlier_ids,
        reply_content=reply_content,
        core_message=core_message,
        core_contents=core_contents,
        expanded_contents=expanded_contents,
        region=region,
        expanded_unit_ids=expanded_unit_ids,
    )
    final.token_count = _fragment_token_count(final)
    return final


def hit_to_anchor(hit: RankedHit) -> RetrievalAnchor:
    unit_type = str(hit.unit_type or "message")
    unit_id = int(hit.unit_id or hit.message_id)
    return RetrievalAnchor(
        unit_type=unit_type,
        unit_id=unit_id,
        parent_message_id=int(hit.parent_message_id or hit.message_id),
        role=str(hit.role),
        content=str(hit.content),
        score=max(float(hit.rerank_score or 0.0), float(hit.fused_score or 0.0)),
        source_query=str(hit.source_query or ""),
        chunk_id=int(hit.chunk_id),
        user_id=int(hit.user_id),
        conversation_id=int(hit.conversation_id),
        position=int(hit.position),
        created_at=int(hit.created_at),
        segment_id=int(hit.segment_id or 0),
        segment_index=int(hit.segment_index),
    )


def collapse_parent_anchors(anchors: List[RetrievalAnchor]) -> List[RetrievalAnchor]:
    """Dedupe message anchors; for segments keep best region per parent message."""
    by_message: Dict[int, List[RetrievalAnchor]] = {}
    message_anchors: Dict[int, RetrievalAnchor] = {}

    for anchor in anchors:
        if anchor.unit_type == "message":
            previous = message_anchors.get(anchor.parent_message_id)
            if previous is None or anchor.score > previous.score:
                message_anchors[anchor.parent_message_id] = anchor
            continue
        by_message.setdefault(int(anchor.parent_message_id), []).append(anchor)

    collapsed: List[RetrievalAnchor] = list(message_anchors.values())
    for parent_id, group in by_message.items():
        if parent_id in message_anchors:
            continue
        group.sort(key=lambda a: (a.segment_index, -a.score))
        regions: List[List[RetrievalAnchor]] = []
        current: List[RetrievalAnchor] = []
        last_idx = -999
        for item in group:
            if not current or int(item.segment_index) <= last_idx + 1:
                current.append(item)
            else:
                regions.append(current)
                current = [item]
            last_idx = int(item.segment_index)
        if current:
            regions.append(current)
        best_region = max(
            regions,
            key=lambda region: max(a.score for a in region),
        )
        best = max(best_region, key=lambda a: a.score)
        core_ids = [int(a.unit_id) for a in sorted(best_region, key=lambda a: a.segment_index)]
        collapsed.append(
            RetrievalAnchor(
                unit_type="segment",
                unit_id=int(best.unit_id),
                parent_message_id=int(parent_id),
                role=str(best.role),
                content=str(best.content),
                score=max(float(a.score) for a in best_region),
                source_query=str(best.source_query),
                chunk_id=int(best.chunk_id),
                user_id=int(best.user_id),
                conversation_id=int(best.conversation_id),
                position=int(best.position),
                created_at=int(best.created_at),
                segment_id=int(best.segment_id or best.unit_id),
                segment_index=min(int(a.segment_index) for a in best_region),
                core_unit_ids=core_ids,
            )
        )
    collapsed.sort(key=lambda a: float(a.score), reverse=True)
    return collapsed


class FragmentBuilder:
    def __init__(self, store: TraceStore, embedder: Optional[MemoryEmbedder] = None) -> None:
        self._store = store
        self._embedder = embedder or MemoryEmbedder.shared()

    def build(
        self,
        *,
        user_id: int,
        hits: List[RankedHit],
        query: str,
    ) -> List[RetrievedFragment]:
        anchors = collapse_parent_anchors([hit_to_anchor(h) for h in hits])
        fragments: List[RetrievedFragment] = []
        for anchor in anchors[: int(mem_cfg.max_retrieved_fragments)]:
            if anchor.unit_type == "message":
                fragments.append(
                    trim_fragment_to_budget(
                        self._build_message_fragment(user_id=user_id, anchor=anchor, query=query)
                    )
                )
            else:
                fragments.append(
                    trim_fragment_to_budget(
                        self._build_segment_fragment(user_id=user_id, anchor=anchor, query=query)
                    )
                )
        return fragments

    def _build_message_fragment(
        self,
        *,
        user_id: int,
        anchor: RetrievalAnchor,
        query: str,
    ) -> RetrievedFragment:
        msg = self._store.get_message(user_id=user_id, message_id=int(anchor.parent_message_id))
        core_content = str(msg["content"]) if msg else anchor.content
        earlier_ids, earlier_contents, later_ids, later_contents, reply_id, reply_content = (
            expand_message_anchor(
                self._store,
                self._embedder,
                user_id=user_id,
                anchor=anchor,
                query=query,
            )
        )
        if str(anchor.role) == "assistant" and reply_id and not reply_content:
            reply_content = build_reply_context_segment_excerpt(
                self._store,
                self._embedder,
                user_id=user_id,
                reply_message_id=int(reply_id),
                query=query,
            )
        token_count = count_tokens(core_content)
        token_count += sum(count_tokens(x) for x in earlier_contents + later_contents)
        token_count += count_tokens(reply_content)
        return RetrievedFragment(
            fragment_type="message",
            anchor_role=str(anchor.role),
            parent_message_id=int(anchor.parent_message_id),
            score=float(anchor.score),
            core_unit_ids=[int(anchor.unit_id)],
            expanded_unit_ids=[],
            reply_context_message_id=reply_id,
            earlier_user_message_ids=earlier_ids,
            later_user_message_ids=later_ids,
            omitted_before=False,
            omitted_after=False,
            token_count=token_count,
            conversation_id=int(anchor.conversation_id),
            created_at=int(anchor.created_at),
            core_message_content=core_content,
            core_contents=[core_content],
            reply_context_content=reply_content,
            earlier_user_contents=earlier_contents,
            later_user_contents=later_contents,
        )

    def _build_segment_fragment(
        self,
        *,
        user_id: int,
        anchor: RetrievalAnchor,
        query: str,
    ) -> RetrievedFragment:
        core_segment_ids = (
            list(anchor.core_unit_ids)
            if anchor.core_unit_ids
            else [int(anchor.segment_id or anchor.unit_id)]
        )
        region = expand_segment_anchor(
            self._store,
            self._embedder,
            user_id=user_id,
            parent_message_id=int(anchor.parent_message_id),
            core_segment_ids=core_segment_ids,
            query=query,
        )
        segments = self._store.list_message_segments(
            user_id=user_id, parent_message_id=int(anchor.parent_message_id)
        )
        by_id = {int(s["id"]): s for s in segments}
        core_contents = [str(by_id[sid]["content"]) for sid in region.core_segment_ids if sid in by_id]
        expanded_contents = [
            str(by_id[sid]["content"])
            for sid in region.before_segment_ids + region.after_segment_ids
            if sid in by_id
        ]
        reply_id = None
        reply_content = ""
        if str(region.role) == "assistant":
            msg = self._store.get_message(user_id=user_id, message_id=int(anchor.parent_message_id))
            rid = (msg or {}).get("reply_to_message_id")
            if rid:
                reply_id = int(rid)
                reply_content = build_reply_context_segment_excerpt(
                    self._store,
                    self._embedder,
                    user_id=user_id,
                    reply_message_id=reply_id,
                    query=query,
                )
        return RetrievedFragment(
            fragment_type="segment",
            anchor_role=str(region.role),
            parent_message_id=int(anchor.parent_message_id),
            score=float(anchor.score),
            core_unit_ids=list(region.core_segment_ids),
            expanded_unit_ids=region.before_segment_ids + region.after_segment_ids,
            reply_context_message_id=reply_id,
            earlier_user_message_ids=[],
            later_user_message_ids=[],
            omitted_before=bool(region.omitted_before),
            omitted_after=bool(region.omitted_after),
            token_count=int(region.token_count) + count_tokens(reply_content),
            conversation_id=int(anchor.conversation_id),
            created_at=int(anchor.created_at),
            segment_region=region,
            core_contents=core_contents,
            expanded_contents=expanded_contents,
            reply_context_content=reply_content,
        )
