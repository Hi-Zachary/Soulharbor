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
    # Feature rerank removed from default ER path; flag kept for ablation/legacy.
    rerank_enabled: bool = _b("MEMORY_BUNDLE_RERANK_ENABLED", "0")
    observability: bool = _b("MEMORY_OBSERVABILITY", "1")

    # Experience Rebuild (ER): adaptive / fixed stitch
    stitch_mode: str = _env("MEMORY_STITCH_MODE", "MEMORY_EXPAND_MODE", default="adaptive")
    # Cosine threshold vs anchor (legacy alias: CONTINUITY_THRESHOLD).
    stitch_cos_threshold: float = float(
        _env(
            "MEMORY_STITCH_COS_THRESHOLD",
            "MEMORY_STITCH_CONTINUITY_THRESHOLD",
            "MEMORY_CONTINUITY_THRESHOLD",
            default="0.40",
        )
    )
    # Adjacent entity escape: max position distance from current window edge.
    stitch_entity_dist: int = int(os.environ.get("MEMORY_STITCH_ENTITY_DIST", "2"))
    stitch_max_misses: int = int(os.environ.get("MEMORY_STITCH_MAX_MISSES", "2"))
    stitch_max_span: int = int(
        _env("MEMORY_STITCH_MAX_SPAN", "MEMORY_EXPAND_MAX_SPAN", default="12")
    )
    # mmr | topk  (legacy: coverage → mmr)
    evidence_selection_mode: str = os.environ.get("EVIDENCE_SELECTION_MODE", "mmr")
    # Explicit cross-session chain scoring removed; chronological injection handles order.
    cross_session_linking: bool = _b("MEMORY_CROSS_SESSION_LINKING", "0")
    link_score_threshold: float = float(os.environ.get("MEMORY_LINK_THRESHOLD", "0.22"))

    # Decay-aware MMR
    decay_enabled: bool = _b("MEMORY_DECAY_ENABLED", "1")
    decay_w_min: float = float(os.environ.get("MEMORY_DECAY_W_MIN", "0.2"))
    decay_tau_sec: float = float(os.environ.get("MEMORY_DECAY_TAU_SEC", str(90 * 86400)))
    decay_alpha: float = float(os.environ.get("MEMORY_DECAY_ALPHA", "0.5"))
    mmr_lambda: float = float(os.environ.get("MEMORY_MMR_LAMBDA", "0.7"))
    # Relative recency inside the candidate pool (helps current_state / anti-stale).
    mmr_recency_beta: float = float(os.environ.get("MEMORY_MMR_RECENCY_BETA", "0.8"))
    mmr_recency_tau_sec: float = float(
        os.environ.get("MEMORY_MMR_RECENCY_TAU_SEC", str(21 * 86400))
    )
    mmr_week_bonus: float = float(os.environ.get("MEMORY_MMR_WEEK_BONUS", "0.5"))
    reinforce_enabled: bool = _b("MEMORY_REINFORCE_ENABLED", "1")
    reinforce_eta: float = float(os.environ.get("MEMORY_REINFORCE_ETA", "0.1"))

    context_token_budget: int = int(os.environ.get("MEMORY_CONTEXT_TOKEN_BUDGET", "1600"))
    semantic_top_k: int = int(os.environ.get("MEMORY_SEMANTIC_TOP_K", "30"))
    lexical_top_k: int = int(os.environ.get("MEMORY_LEXICAL_TOP_K", "30"))
    anchor_top_k: int = int(
        _env("MEMORY_ANCHOR_TOP_K", "MEMORY_FOCUS_TOP_K", "MEMORY_SEED_TOP_K", default="8")
    )
    bundle_top_k: int = int(os.environ.get("MEMORY_WINDOW_TOP_K", "6"))
    neighbor_before: int = int(os.environ.get("MEMORY_NEIGHBOR_BEFORE", "2"))
    neighbor_after: int = int(os.environ.get("MEMORY_NEIGHBOR_AFTER", "2"))
    bundle_max_messages: int = int(os.environ.get("MEMORY_WINDOW_MAX_MESSAGES", "8"))
    chunk_soft_limit: int = int(os.environ.get("MEMORY_CHUNK_SOFT_LIMIT", "500"))
    chunk_target_min: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MIN", "150"))
    chunk_target_max: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MAX", "350"))
    assistant_weight: float = float(os.environ.get("MEMORY_ASSISTANT_WEIGHT", "0.55"))
    embed_retry_max: int = int(os.environ.get("MEMORY_EMBED_RETRY_MAX", "5"))
    # FAISS IndexFlatIP over stored BGE vectors (fallback: Python cosine loop).
    ann_enabled: bool = _b("MEMORY_ANN_ENABLED", "1")

    # Back-compat alias used by older call sites / tests.
    @property
    def stitch_continuity_threshold(self) -> float:
        return self.stitch_cos_threshold


# module-level settings used by store / retrieval
mem_cfg = MemorySettings()
memory_settings = mem_cfg
