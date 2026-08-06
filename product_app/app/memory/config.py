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
    # After user turns, Chinese LLM may maintain long-term profile facts.
    profile_llm_propose: bool = _b("MEMORY_PROFILE_LLM_PROPOSE", "1")
    profile_llm_propose_max: int = int(os.environ.get("MEMORY_PROFILE_LLM_PROPOSE_MAX", "3"))
    profile_max_operations: int = int(
        os.environ.get(
            "MEMORY_PROFILE_MAX_OPERATIONS",
            os.environ.get("MEMORY_PROFILE_LLM_PROPOSE_MAX", "3"),
        )
    )
    # Hard caps on active profile rows / tokens (code-enforced).
    profile_max_active: int = int(os.environ.get("MEMORY_PROFILE_MAX_ACTIVE", "20"))
    profile_item_target_chars: int = int(
        os.environ.get("MEMORY_PROFILE_ITEM_TARGET_CHARS", "40")
    )
    profile_item_max_chars: int = int(
        os.environ.get("MEMORY_PROFILE_ITEM_MAX_CHARS", "64")
    )
    profile_item_max_tokens: int = int(
        os.environ.get("MEMORY_PROFILE_ITEM_MAX_TOKENS", "48")
    )
    profile_block_max_tokens: int = int(
        os.environ.get("MEMORY_PROFILE_BLOCK_MAX_TOKENS", "640")
    )
    # Legacy flags kept so old env files still load.
    profile_llm_skip_if_pending: bool = _b("MEMORY_PROFILE_LLM_SKIP_IF_PENDING", "0")
    profile_llm_trigger_messages: int = int(
        os.environ.get("MEMORY_PROFILE_LLM_TRIGGER_MESSAGES", "5")
    )
    profile_llm_trigger_age_sec: int = int(
        os.environ.get("MEMORY_PROFILE_LLM_TRIGGER_AGE_SEC", "300")
    )
    split_query_enabled: bool = _b("MEMORY_SPLIT_QUERY_ENABLED", "1")

    # Anchor CE: direct Top-K vs split per-subquery ∪ original fill.
    # Candidate caps only; token budget decides how many windows are injected.
    rrf_top_k: int = int(os.environ.get("MEMORY_RRF_TOP_K", "40"))
    anchor_ce_top_k: int = int(os.environ.get("MEMORY_ANCHOR_CE_TOP_K", "10"))
    anchor_ce_direct_k: int = int(os.environ.get("MEMORY_ANCHOR_CE_DIRECT_K", "10"))
    anchor_ce_per_subquery_k: int = int(
        os.environ.get("MEMORY_ANCHOR_CE_PER_SUBQUERY_K", "3")
    )
    rerank_max_length: int = int(os.environ.get("MEMORY_RERANK_MAX_LENGTH", "1024"))
    rerank_batch_size: int = int(os.environ.get("MEMORY_RERANK_BATCH_SIZE", "16"))
    observability: bool = _b("MEMORY_OBSERVABILITY", "1")
    # When true, retrieval exceptions propagate instead of returning empty memory.
    raise_retrieval_errors: bool = _b("MEMORY_RAISE_RETRIEVAL_ERRORS", "0")

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
    max_episodic_tokens: int = int(os.environ.get("MEMORY_MAX_EPISODIC_TOKENS", "2400"))
    max_retrieved_fragments: int = int(os.environ.get("MEMORY_MAX_RETRIEVED_FRAGMENTS", "10"))
    max_message_fragment_tokens: int = int(
        os.environ.get("MEMORY_MAX_MESSAGE_FRAGMENT_TOKENS", "520")
    )
    max_segment_fragment_tokens: int = int(
        os.environ.get("MEMORY_MAX_SEGMENT_FRAGMENT_TOKENS", "320")
    )
    max_reply_context_tokens: int = int(os.environ.get("MEMORY_MAX_REPLY_CONTEXT_TOKENS", "220"))

    message_split_threshold: int = int(os.environ.get("MEMORY_MESSAGE_SPLIT_THRESHOLD", "320"))
    segment_target_tokens: int = int(os.environ.get("MEMORY_SEGMENT_TARGET_TOKENS", "140"))
    segment_hard_max_tokens: int = int(os.environ.get("MEMORY_SEGMENT_HARD_MAX_TOKENS", "200"))
    segment_max_before: int = int(os.environ.get("MEMORY_SEGMENT_MAX_BEFORE", "1"))
    segment_max_after: int = int(os.environ.get("MEMORY_SEGMENT_MAX_AFTER", "1"))
    message_max_before_users: int = int(os.environ.get("MEMORY_MESSAGE_MAX_BEFORE_USERS", "2"))
    message_max_after_users: int = int(os.environ.get("MEMORY_MESSAGE_MAX_AFTER_USERS", "2"))
    message_max_scan_distance: int = int(os.environ.get("MEMORY_MESSAGE_MAX_SCAN_DISTANCE", "8"))
    expansion_similarity_threshold: float = float(
        os.environ.get("MEMORY_EXPANSION_SIMILARITY_THRESHOLD", "0.40")
    )
    role_scope_fallback_min_hits: int = int(
        os.environ.get("MEMORY_ROLE_SCOPE_FALLBACK_MIN_HITS", "2")
    )
    semantic_top_k: int = int(os.environ.get("MEMORY_SEMANTIC_TOP_K", "50"))
    lexical_top_k: int = int(os.environ.get("MEMORY_LEXICAL_TOP_K", "50"))
    anchor_top_k: int = int(
        _env("MEMORY_ANCHOR_TOP_K", "MEMORY_FOCUS_TOP_K", "MEMORY_SEED_TOP_K", default="8")
    )
    bundle_top_k: int = int(os.environ.get("MEMORY_WINDOW_TOP_K", "10"))
    neighbor_before: int = int(os.environ.get("MEMORY_NEIGHBOR_BEFORE", "2"))
    neighbor_after: int = int(os.environ.get("MEMORY_NEIGHBOR_AFTER", "2"))
    bundle_max_messages: int = int(os.environ.get("MEMORY_WINDOW_MAX_MESSAGES", "8"))
    chunk_soft_limit: int = int(os.environ.get("MEMORY_CHUNK_SOFT_LIMIT", "500"))
    chunk_target_min: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MIN", "150"))
    chunk_target_max: int = int(os.environ.get("MEMORY_CHUNK_TARGET_MAX", "350"))
    assistant_weight: float = float(os.environ.get("MEMORY_ASSISTANT_WEIGHT", "0.55"))
    embed_retry_max: int = int(os.environ.get("MEMORY_EMBED_RETRY_MAX", "5"))
    ann_enabled: bool = _b("MEMORY_ANN_ENABLED", "1")
    # Display timezone for「记录于」labels and the dynamic「当前日期」hint.
    memory_timezone: str = os.environ.get("MEMORY_TIMEZONE", "Asia/Shanghai")

    def fuse_top_k(self) -> int:
        return int(self.rrf_top_k) if self.rrf_top_k > 0 else int(self.anchor_top_k)

    @property
    def stitch_continuity_threshold(self) -> float:
        return self.stitch_cos_threshold

    def __post_init__(self) -> None:
        if int(self.context_token_budget) < int(self.profile_block_max_tokens):
            raise ValueError(
                "context token budget must cover profile block "
                f"({self.context_token_budget} < {self.profile_block_max_tokens})"
            )


mem_cfg = MemorySettings()
memory_settings = mem_cfg
