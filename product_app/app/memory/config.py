"""Runtime knobs for the long-term memory stack."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _b(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


@dataclass(frozen=True)
class MemorySettings:
    # legacy | aer
    backend: str = os.environ.get("MEMORY_BACKEND", "aer")
    store_enabled: bool = _b("MEMORY_STORE_ENABLED", "1")
    profile_enabled: bool = _b("MEMORY_PROFILE_ENABLED", "1")
    split_query_enabled: bool = _b("MEMORY_SPLIT_QUERY_ENABLED", "1")
    rerank_enabled: bool = _b("MEMORY_BUNDLE_RERANK_ENABLED", "1")
    observability: bool = _b("MEMORY_OBSERVABILITY", "1")

    # Adaptive Experience Reconstruction
    # fixed | adaptive
    expand_mode: str = os.environ.get("MEMORY_EXPAND_MODE", "adaptive")
    expansion_continuity_threshold: float = float(
        os.environ.get("MEMORY_CONTINUITY_THRESHOLD", "0.28")
    )
    expansion_max_span: int = int(os.environ.get("MEMORY_EXPAND_MAX_SPAN", "12"))
    # topk | coverage
    evidence_selection_mode: str = os.environ.get("EVIDENCE_SELECTION_MODE", "coverage")
    cross_session_linking: bool = _b("MEMORY_CROSS_SESSION_LINKING", "1")
    link_score_threshold: float = float(os.environ.get("MEMORY_LINK_THRESHOLD", "0.22"))

    context_token_budget: int = int(os.environ.get("MEMORY_CONTEXT_TOKEN_BUDGET", "1600"))
    semantic_top_k: int = int(os.environ.get("MEMORY_SEMANTIC_TOP_K", "30"))
    lexical_top_k: int = int(os.environ.get("MEMORY_LEXICAL_TOP_K", "30"))
    seed_top_k: int = int(os.environ.get("MEMORY_SEED_TOP_K", "8"))
    bundle_top_k: int = int(os.environ.get("MEMORY_WINDOW_TOP_K", "6"))
    neighbor_before: int = int(os.environ.get("MEMORY_NEIGHBOR_BEFORE", "2"))
    neighbor_after: int = int(os.environ.get("MEMORY_NEIGHBOR_AFTER", "2"))
    bundle_max_messages: int = int(os.environ.get("MEMORY_WINDOW_MAX_MESSAGES", "10"))
    chunk_soft_limit: int = int(os.environ.get("MEMORY_CHUNK_SOFT_LIMIT", "500"))
    chunk_target_min: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MIN", "150"))
    chunk_target_max: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MAX", "350"))
    assistant_weight: float = float(os.environ.get("MEMORY_ASSISTANT_WEIGHT", "0.55"))
    embed_retry_max: int = int(os.environ.get("MEMORY_EMBED_RETRY_MAX", "5"))
    # FAISS IndexFlatIP over stored BGE vectors (fallback: Python cosine loop).
    ann_enabled: bool = _b("MEMORY_ANN_ENABLED", "1")


# module-level settings used by store / retrieval
mem_cfg = MemorySettings()
memory_settings = mem_cfg
