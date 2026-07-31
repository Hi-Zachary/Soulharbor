"""Grow each search seed into a short conversation window."""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.embeddings import MemoryEmbedder
from product_app.app.memory.models import EpisodeWindow, RankedHit, WindowTurn
from product_app.app.memory.store.repository import EpisodeStore
from product_app.app.memory.store.text_sim import blob_of, cosine, entities, jaccard, tokens


def _as_turn(row: dict, conversation_id: int, *, is_seed: bool = False) -> WindowTurn:
    return WindowTurn(
        message_id=int(row["message_id"]),
        conversation_id=int(conversation_id),
        role=str(row["role"]),
        position=int(row["position"]),
        content=str(row["content"]),
        created_at=int(row["created_at"]),
        is_seed=is_seed,
    )


def _window_span(turns: Sequence[WindowTurn]) -> int:
    if not turns:
        return 0
    positions = [t.position for t in turns]
    return max(positions) - min(positions) + 1


class WindowExpander:
    """Turn ranked hits into multi-turn windows (adaptive walk or fixed neighbors)."""

    def __init__(self, store: EpisodeStore, embedder: Optional[MemoryEmbedder] = None) -> None:
        self._store = store
        self._embedder = embedder
        self._vec_cache: Dict[str, List[float]] = {}

    def expand(
        self,
        *,
        user_id: int,
        seeds: List[RankedHit],
        queries: Optional[List[str]] = None,
    ) -> List[EpisodeWindow]:
        if (mem_cfg.expand_mode or "adaptive").lower() == "fixed":
            return self._fixed_windows(user_id, seeds, queries)
        return self._adaptive_windows(user_id, seeds, queries)

    # --- embeddings ---------------------------------------------------------

    @staticmethod
    def _cache_key(text: str) -> str:
        raw = text or ""
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return f"{len(raw)}:{digest}"

    def _vector(self, text: str) -> List[float]:
        # Key on full text so long messages that share a prefix never collide.
        # Embed the full string; the encoder applies its own max-length truncate.
        key = self._cache_key(text)
        cached = self._vec_cache.get(key)
        if cached is not None:
            return cached
        embedder = self._embedder or MemoryEmbedder.shared()
        vec = list(embedder.embed(text or "") or [])
        self._vec_cache[key] = vec
        return vec

    # --- adaptive scoring ---------------------------------------------------

    def _continuity_score(
        self,
        candidate: WindowTurn,
        seed: WindowTurn,
        selected: Sequence[WindowTurn],
        query: str,
    ) -> float:
        to_seed = cosine(self._vector(candidate.content), self._vector(seed.content))
        to_window = cosine(self._vector(candidate.content), self._vector(blob_of(list(selected))))
        entity_overlap = jaccard(
            entities(candidate.content),
            entities(seed.content) | entities(query),
        )
        positions = [t.position for t in selected]
        dist = min(abs(candidate.position - min(positions)), abs(candidate.position - max(positions)))
        if dist <= 1:
            near = 1.0
        elif dist == 2:
            near = 0.5
        else:
            near = 0.0
        query_overlap = jaccard(tokens(candidate.content), tokens(query))
        return (
            0.40 * to_seed
            + 0.22 * to_window
            + 0.18 * entity_overlap
            + 0.10 * near
            + 0.10 * query_overlap
        )

    def _keep_neighbor(
        self,
        candidate: WindowTurn,
        seed: WindowTurn,
        selected: Sequence[WindowTurn],
        query: str,
        threshold: float,
    ) -> bool:
        score = self._continuity_score(candidate, seed, selected, query)
        if score >= threshold:
            return True
        # weak embedding but strong shared entities — still keep
        shared = jaccard(entities(candidate.content), entities(seed.content))
        return shared >= 0.35 and score >= threshold * 0.75

    def _grow_from_hit(self, user_id: int, hit: RankedHit, query: str) -> EpisodeWindow:
        conv_id = int(hit.conversation_id)
        rows = self._store.list_conversation_messages(user_id=user_id, conversation_id=conv_id)
        by_pos = {int(r["position"]): r for r in rows}

        if hit.position not in by_pos:
            seed = WindowTurn(
                message_id=hit.message_id,
                conversation_id=conv_id,
                role=hit.role,
                position=hit.position,
                content=hit.content,
                created_at=hit.created_at,
                is_seed=True,
            )
            return EpisodeWindow(
                bundle_id=f"c{conv_id}-{hit.message_id}",
                conversation_id=conv_id,
                seed_ids=[hit.message_id],
                messages=[seed],
                fused_score=float(hit.fused_score or 0.0),
            )

        seed = _as_turn(by_pos[hit.position], conv_id, is_seed=True)
        selected: List[WindowTurn] = [seed]
        threshold = float(mem_cfg.expansion_continuity_threshold)
        max_span = int(mem_cfg.expansion_max_span)
        max_msgs = int(mem_cfg.bundle_max_messages)

        # walk left
        misses = 0
        pos = hit.position - 1
        while pos >= 1 and len(selected) < max_msgs and _window_span(selected) < max_span:
            row = by_pos.get(pos)
            if row is None:
                pos -= 1
                continue
            cand = _as_turn(row, conv_id)
            if self._keep_neighbor(cand, seed, selected, query, threshold):
                selected.insert(0, cand)
                misses = 0
            else:
                misses += 1
                if misses >= 2:
                    break
            pos -= 1

        # walk right
        misses = 0
        pos = hit.position + 1
        last_pos = max(by_pos) if by_pos else hit.position
        while pos <= last_pos and len(selected) < max_msgs and _window_span(selected) < max_span:
            row = by_pos.get(pos)
            if row is None:
                pos += 1
                continue
            cand = _as_turn(row, conv_id)
            if self._keep_neighbor(cand, seed, selected, query, threshold):
                selected.append(cand)
                misses = 0
            else:
                misses += 1
                if misses >= 2:
                    break
            pos += 1

        return EpisodeWindow(
            bundle_id=f"c{conv_id}-{hit.message_id}",
            conversation_id=conv_id,
            seed_ids=[hit.message_id],
            messages=selected,
            fused_score=float(hit.fused_score or 0.0),
        )

    def _adaptive_windows(
        self,
        user_id: int,
        seeds: List[RankedHit],
        queries: Optional[List[str]],
    ) -> List[EpisodeWindow]:
        joined = " ".join(queries or [])
        windows = [
            self._grow_from_hit(user_id, hit, joined or hit.content) for hit in seeds
        ]
        for w in windows:
            w.retrieval_queries = list(queries or [])
        windows.sort(key=lambda w: w.fused_score, reverse=True)
        keep = max(int(mem_cfg.bundle_top_k) * 2, int(mem_cfg.seed_top_k))
        return windows[:keep]

    # --- fixed neighbor windows ---------------------------------------------

    def _fixed_windows(
        self,
        user_id: int,
        seeds: List[RankedHit],
        queries: Optional[List[str]],
    ) -> List[EpisodeWindow]:
        before = int(mem_cfg.neighbor_before)
        after = int(mem_cfg.neighbor_after)
        max_msgs = int(mem_cfg.bundle_max_messages)

        # group raw ranges by conversation
        ranges: Dict[int, List[Tuple[int, int, RankedHit]]] = {}
        for hit in seeds:
            ranges.setdefault(hit.conversation_id, []).append(
                (hit.position - before, hit.position + after, hit)
            )

        out: List[EpisodeWindow] = []
        for conv_id, items in ranges.items():
            items.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int, List[RankedHit]]] = []
            for lo, hi, hit in items:
                if merged and lo <= merged[-1][1] + 1:
                    prev_lo, prev_hi, hits = merged[-1]
                    merged[-1] = (prev_lo, max(prev_hi, hi), hits + [hit])
                else:
                    merged.append((lo, hi, [hit]))

            for _lo, _hi, hits in merged:
                by_id: Dict[int, WindowTurn] = {}
                seed_ids = sorted({h.message_id for h in hits})
                best_score = max(h.fused_score for h in hits)
                for hit in hits:
                    for row in self._store.neighbor_messages(
                        user_id=user_id,
                        conversation_id=conv_id,
                        position=hit.position,
                        before=before,
                        after=after,
                    ):
                        mid = int(row["message_id"])
                        marked = mid in seed_ids
                        existing = by_id.get(mid)
                        if existing is None:
                            by_id[mid] = _as_turn(row, conv_id, is_seed=marked)
                        elif marked:
                            existing.is_seed = True

                turns = sorted(by_id.values(), key=lambda t: t.position)
                if len(turns) > max_msgs:
                    seed_pos = {t.position for t in turns if t.is_seed}
                    if not seed_pos:
                        turns = turns[:max_msgs]
                    else:
                        center = sum(seed_pos) / len(seed_pos)
                        turns = sorted(
                            turns,
                            key=lambda t: (0 if t.is_seed else 1, abs(t.position - center)),
                        )[:max_msgs]
                        turns = sorted(turns, key=lambda t: t.position)

                out.append(
                    EpisodeWindow(
                        bundle_id=f"c{conv_id}-" + "-".join(str(i) for i in seed_ids[:4]),
                        conversation_id=conv_id,
                        seed_ids=seed_ids,
                        messages=turns,
                        fused_score=best_score,
                        retrieval_queries=list(queries or []),
                    )
                )

        out.sort(key=lambda w: w.fused_score, reverse=True)
        return out[: int(mem_cfg.bundle_top_k)]
