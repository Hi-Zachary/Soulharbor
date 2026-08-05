"""Per-user FAISS cache for semantic retrieval.

Vectors are already stored in SQLite; this only accelerates nearest-neighbor
lookup (IndexFlatIP on L2-normalized embeddings ≈ cosine). Fingerprint-based
invalidation avoids wiring every write path.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from product_app.app.memory.models import Block

logger = logging.getLogger(__name__)

_Fingerprint = Tuple[int, int, int]  # count, max_chunk_id, max_updated_at
_CacheKey = Tuple[int, str]  # user_id, index kind (e.g. user_only)


@dataclass
class _UserIndex:
    fingerprint: _Fingerprint
    index: object  # faiss.Index
    rows: List[Block]
    kind: str = "user_only"


class UserAnnCache:
    """Process-local FAISS indexes keyed by (user_id, kind)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_key: Dict[_CacheKey, _UserIndex] = {}
        self._faiss = None
        self._faiss_error: Optional[str] = None

    def clear(self, user_id: Optional[int] = None) -> None:
        with self._lock:
            if user_id is None:
                self._by_key.clear()
                return
            uid = int(user_id)
            for key in [k for k in self._by_key if k[0] == uid]:
                self._by_key.pop(key, None)

    def _ensure_faiss(self):
        if self._faiss is not None:
            return self._faiss
        if self._faiss_error is not None:
            return None
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            return faiss
        except Exception as exc:
            self._faiss_error = str(exc)
            logger.warning("faiss unavailable; semantic search falls back to Python loop (%s)", exc)
            return None

    @staticmethod
    def _matrix(rows: Sequence[Block]) -> Tuple[np.ndarray, List[Block]]:
        kept: List[Block] = []
        vectors: List[List[float]] = []
        dim = 0
        for row in rows:
            emb = row.embedding
            if not emb:
                continue
            if dim == 0:
                dim = len(emb)
            if len(emb) != dim:
                continue
            kept.append(row)
            vectors.append(emb)
        if not kept:
            return np.zeros((0, 0), dtype=np.float32), []
        mat = np.asarray(vectors, dtype=np.float32)
        # Cosine via inner product on L2-normalized rows.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms
        return mat, kept

    def _build(
        self,
        fingerprint: _Fingerprint,
        rows: Sequence[Block],
        *,
        kind: str,
    ) -> Optional[_UserIndex]:
        faiss = self._ensure_faiss()
        if faiss is None:
            return None
        mat, kept = self._matrix(rows)
        if mat.size == 0:
            return _UserIndex(fingerprint=fingerprint, index=None, rows=[], kind=kind)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        return _UserIndex(fingerprint=fingerprint, index=index, rows=kept, kind=kind)

    def peek(self, user_id: int, *, kind: str = "user_only") -> Optional[_UserIndex]:
        with self._lock:
            return self._by_key.get((int(user_id), str(kind)))

    def get_or_build(
        self,
        user_id: int,
        fingerprint: _Fingerprint,
        rows: Sequence[Block],
        *,
        kind: str = "user_only",
    ) -> Optional[_UserIndex]:
        key = (int(user_id), str(kind))
        with self._lock:
            cached = self._by_key.get(key)
            if cached is not None and cached.fingerprint == fingerprint:
                return cached
            built = self._build(fingerprint, rows, kind=str(kind))
            if built is None:
                return None
            self._by_key[key] = built
            return built

    def search(
        self,
        user_index: _UserIndex,
        query_vec: Sequence[float],
        *,
        top_k: int,
    ) -> List[Tuple[float, Block]]:
        if user_index.index is None or not user_index.rows or top_k <= 0:
            return []
        q = np.asarray(list(query_vec), dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(q))
        if norm <= 0:
            return []
        q = q / norm
        k = min(int(top_k), len(user_index.rows))
        scores, indices = user_index.index.search(q, k)
        out: List[Tuple[float, Block]] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(user_index.rows):
                continue
            out.append((float(score), user_index.rows[idx]))
        return out


_CACHE = UserAnnCache()
# Drop any legacy mixed user+assistant indexes from older process lifetimes.
_CACHE.clear()


def ann_cache() -> UserAnnCache:
    return _CACHE
