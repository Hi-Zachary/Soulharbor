"""Character / n-gram BM25 search over episode text."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import RankedHit
from product_app.app.memory.store.repository import EpisodeStore

_TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> List[str]:
    """Unigrams for CJK chars, plus adjacent bigrams; lowercase latin tokens."""
    pieces = _TOKEN.findall(text or "")
    out: List[str] = []
    cjk_run: List[str] = []

    def flush_cjk() -> None:
        nonlocal cjk_run
        if len(cjk_run) >= 2:
            for i in range(len(cjk_run) - 1):
                out.append(cjk_run[i] + cjk_run[i + 1])
        cjk_run = []

    for piece in pieces:
        is_cjk = len(piece) == 1 and "\u4e00" <= piece <= "\u9fff"
        if is_cjk:
            cjk_run.append(piece)
            out.append(piece)
        else:
            flush_cjk()
            out.append(piece.lower())
    flush_cjk()
    return out


class LexicalSearcher:
    def __init__(self, store: EpisodeStore) -> None:
        self._store = store

    def search(
        self,
        *,
        user_id: int,
        query: str,
        limit: int | None = None,
        exclude_message_ids: Optional[Set[int]] = None,
    ) -> List[RankedHit]:
        text = (query or "").strip()
        if not text:
            return []

        query_terms = _tokenize(text)
        if not query_terms:
            return []

        skip = {int(x) for x in (exclude_message_ids or set())}
        top_n = int(limit or mem_cfg.lexical_top_k)
        rows = [
            row
            for row in self._store.list_active_with_embeddings(user_id, limit=5000)
            if row.message_id not in skip
        ]
        if not rows:
            return []

        docs = [_tokenize(row.content) for row in rows]
        doc_freq: Dict[str, int] = defaultdict(int)
        for terms in docs:
            for term in set(terms):
                doc_freq[term] += 1

        n_docs = len(docs)
        avg_len = sum(len(d) for d in docs) / max(1, n_docs)
        wanted = set(query_terms)
        assistant_scale = float(mem_cfg.assistant_weight)

        scored: List[Tuple[float, RankedHit]] = []
        for row, terms in zip(rows, docs):
            if not terms:
                continue
            tf = Counter(terms)
            score = 0.0
            doc_len = len(terms)
            for term in wanted:
                if term not in tf:
                    continue
                idf = math.log(1.0 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                freq = tf[term]
                denom = freq + _K1 * (1.0 - _B + _B * doc_len / avg_len)
                score += idf * (freq * (_K1 + 1.0)) / denom
            if row.role == "assistant":
                score *= assistant_scale
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    RankedHit(
                        chunk_id=row.id,
                        user_id=row.user_id,
                        conversation_id=row.conversation_id,
                        message_id=row.message_id,
                        role=row.role,
                        position=row.position,
                        content=row.content,
                        created_at=row.created_at,
                    ),
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits: List[RankedHit] = []
        for rank, (_score, hit) in enumerate(scored[:top_n], start=1):
            hit.lexical_rank = rank
            hits.append(hit)
        return hits
