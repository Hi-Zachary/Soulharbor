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


def _backend() -> str:
    raw = os.environ.get("MEMORY_BACKEND", "er").strip().lower()
    # Legacy alias kept so old start scripts still enable long-term memory.
    return "er" if raw in ("er", "aer") else raw


@dataclass(frozen=True)
class MemorySettings:
    # er only (legacy alias: aer); other values disable long-term memory
    backend: str = _backend()
    store_enabled: bool = _b("MEMORY_STORE_ENABLED", "1")
    profile_enabled: bool = _b("MEMORY_PROFILE_ENABLED", "1")
    # After assistant turns, Chinese LLM may extract allowlisted long-term facts.
    profile_llm_propose: bool = _b("MEMORY_PROFILE_LLM_PROPOSE", "1")
    profile_llm_propose_max: int = int(os.environ.get("MEMORY_PROFILE_LLM_PROPOSE_MAX", "3"))
    # Legacy flag (pending queue removed); kept so old env files still load.
    profile_llm_skip_if_pending: bool = _b("MEMORY_PROFILE_LLM_SKIP_IF_PENDING", "0")
    # Batch-gated extract: after N new messages or age (seconds).
    profile_llm_trigger_messages: int = int(
        os.environ.get("MEMORY_PROFILE_LLM_TRIGGER_MESSAGES", "5")
    )
    profile_llm_trigger_age_sec: int = int(
        os.environ.get("MEMORY_PROFILE_LLM_TRIGGER_AGE_SEC", "300")
    )
    split_query_enabled: bool = _b("MEMORY_SPLIT_QUERY_ENABLED", "1")

    # Multi-query coverage CE on raw anchors (formal final path).
    # Candidate caps only: CE may keep up to 12 anchors, and post-merge Top-k may
    # keep up to 12 windows. Token budget decides how many are actually injected.
    rrf_top_k: int = int(os.environ.get("MEMORY_RRF_TOP_K", "40"))
    anchor_ce_top_k: int = int(os.environ.get("MEMORY_ANCHOR_CE_TOP_K", "12"))
    anchor_ce_orig_top: int = int(os.environ.get("MEMORY_ANCHOR_CE_ORIG_TOP", "4"))
    anchor_ce_sub_top: int = int(os.environ.get("MEMORY_ANCHOR_CE_SUB_TOP", "3"))
    anchor_ce_min_per_query: int = int(os.environ.get("MEMORY_ANCHOR_CE_MIN_PER_QUERY", "1"))
    rerank_max_length: int = int(os.environ.get("MEMORY_RERANK_MAX_LENGTH", "1024"))
    rerank_batch_size: int = int(os.environ.get("MEMORY_RERANK_BATCH_SIZE", "16"))
    observability: bool = _b("MEMORY_OBSERVABILITY", "1")

    # Experience Rebuild (ER): adaptive / fixed stitch
    stitch_mode: str = _env("MEMORY_STITCH_MODE", "MEMORY_EXPAND_MODE", default="adaptive")
    stitch_cos_threshold: float = float(
        _env(
            "MEMORY_STITCH_COS_THRESHOLD",
            "MEMORY_STITCH_CONTINUITY_THRESHOLD",
            "MEMORY_CONTINUITY_THRESHOLD",
            default="0.40",
        )
    )
    stitch_max_misses: int = int(os.environ.get("MEMORY_STITCH_MAX_MISSES", "2"))
    stitch_max_span: int = int(
        _env("MEMORY_STITCH_MAX_SPAN", "MEMORY_EXPAND_MAX_SPAN", default="12")
    )
    # Final evidence selection: CE Top-k (legacy aliases map here).
    evidence_selection_mode: str = os.environ.get("EVIDENCE_SELECTION_MODE", "topk")
    cross_session_linking: bool = _b("MEMORY_CROSS_SESSION_LINKING", "0")
    link_score_threshold: float = float(os.environ.get("MEMORY_LINK_THRESHOLD", "0.22"))

    context_token_budget: int = int(os.environ.get("MEMORY_CONTEXT_TOKEN_BUDGET", "1600"))
    semantic_top_k: int = int(os.environ.get("MEMORY_SEMANTIC_TOP_K", "50"))
    lexical_top_k: int = int(os.environ.get("MEMORY_LEXICAL_TOP_K", "50"))
    anchor_top_k: int = int(
        _env("MEMORY_ANCHOR_TOP_K", "MEMORY_FOCUS_TOP_K", "MEMORY_SEED_TOP_K", default="8")
    )
    bundle_top_k: int = int(os.environ.get("MEMORY_WINDOW_TOP_K", "12"))
    neighbor_before: int = int(os.environ.get("MEMORY_NEIGHBOR_BEFORE", "2"))
    neighbor_after: int = int(os.environ.get("MEMORY_NEIGHBOR_AFTER", "2"))
    bundle_max_messages: int = int(os.environ.get("MEMORY_WINDOW_MAX_MESSAGES", "8"))
    chunk_soft_limit: int = int(os.environ.get("MEMORY_CHUNK_SOFT_LIMIT", "500"))
    chunk_target_min: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MIN", "150"))
    chunk_target_max: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MAX", "350"))
    assistant_weight: float = float(os.environ.get("MEMORY_ASSISTANT_WEIGHT", "0.55"))
    embed_retry_max: int = int(os.environ.get("MEMORY_EMBED_RETRY_MAX", "5"))
    ann_enabled: bool = _b("MEMORY_ANN_ENABLED", "1")

    def fuse_top_k(self) -> int:
        return int(self.rrf_top_k) if self.rrf_top_k > 0 else int(self.anchor_top_k)

    @property
    def stitch_continuity_threshold(self) -> float:
        return self.stitch_cos_threshold


mem_cfg = MemorySettings()
memory_settings = mem_cfg
