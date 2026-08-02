"""Runtime knobs for the long-term memory stack."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _b(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def _env(*names: str, default: str) -> str:
    """Prefer the first defined env var (supports brief legacy aliases)."""
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return default


@dataclass(frozen=True)
class MemorySettings:
    # aer only (other values disable long-term memory)
    backend: str = os.environ.get("MEMORY_BACKEND", "aer")
    store_enabled: bool = _b("MEMORY_STORE_ENABLED", "1")
    profile_enabled: bool = _b("MEMORY_PROFILE_ENABLED", "1")
    # After assistant turns, Chinese LLM may propose pending prefs (consent still required).
    profile_llm_propose: bool = _b("MEMORY_PROFILE_LLM_PROPOSE", "1")
    profile_llm_propose_max: int = int(os.environ.get("MEMORY_PROFILE_LLM_PROPOSE_MAX", "1"))
    profile_llm_skip_if_pending: bool = _b("MEMORY_PROFILE_LLM_SKIP_IF_PENDING", "1")
    # Batch-gated propose: after N new messages or age (seconds).
    profile_llm_trigger_messages: int = int(
        os.environ.get("MEMORY_PROFILE_LLM_TRIGGER_MESSAGES", "5")
    )
    profile_llm_trigger_age_sec: int = int(
        os.environ.get("MEMORY_PROFILE_LLM_TRIGGER_AGE_SEC", "300")
    )
    split_query_enabled: bool = _b("MEMORY_SPLIT_QUERY_ENABLED", "1")
    rerank_enabled: bool = _b("MEMORY_BUNDLE_RERANK_ENABLED", "1")
    observability: bool = _b("MEMORY_OBSERVABILITY", "1")

    # Adaptive Experience Reconstruction
    # fixed | adaptive
    stitch_mode: str = _env("MEMORY_STITCH_MODE", "MEMORY_EXPAND_MODE", default="adaptive")
    stitch_continuity_threshold: float = float(
        _env(
            "MEMORY_STITCH_CONTINUITY_THRESHOLD",
            "MEMORY_CONTINUITY_THRESHOLD",
            default="0.28",
        )
    )
    stitch_max_span: int = int(
        _env("MEMORY_STITCH_MAX_SPAN", "MEMORY_EXPAND_MAX_SPAN", default="12")
    )
    # topk | coverage
    evidence_selection_mode: str = os.environ.get("EVIDENCE_SELECTION_MODE", "coverage")
    cross_session_linking: bool = _b("MEMORY_CROSS_SESSION_LINKING", "1")
    link_score_threshold: float = float(os.environ.get("MEMORY_LINK_THRESHOLD", "0.22"))

    context_token_budget: int = int(os.environ.get("MEMORY_CONTEXT_TOKEN_BUDGET", "1600"))
    semantic_top_k: int = int(os.environ.get("MEMORY_SEMANTIC_TOP_K", "30"))
    lexical_top_k: int = int(os.environ.get("MEMORY_LEXICAL_TOP_K", "30"))
    focus_top_k: int = int(_env("MEMORY_FOCUS_TOP_K", "MEMORY_SEED_TOP_K", default="8"))
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
