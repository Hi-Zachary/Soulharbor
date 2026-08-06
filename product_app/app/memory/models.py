from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Role = Literal["user", "assistant"]
RoleScope = Literal["user", "assistant", "both"]
UnitType = Literal["message", "segment"]


def searchable_text(role: str, content: str) -> str:
    label = "用户" if str(role) == "user" else "助手"
    body = str(content or "").strip()
    return f"[{label}]\n{body}" if body else f"[{label}]"


def retrieval_text(role: str, content: str) -> str:
    return searchable_text(role, content)


@dataclass(frozen=True)
class Turn:
    user_id: int
    conversation_id: int
    message_id: int
    role: Role
    content: str
    position: int
    created_at: int
    turn_id: int = 0
    reply_to_message_id: int | None = None
    retrievable: bool = True
    visible_to_user: bool = True
    is_final: bool = True


@dataclass
class Block:
    id: int
    user_id: int
    conversation_id: int
    message_id: int
    role: str
    position: int
    chunk_index: int
    content: str
    created_at: int
    turn_id: int = 0
    is_deleted: bool = False
    embedding: Optional[List[float]] = None
    retrievable: bool = True
    visible_to_user: bool = True
    is_final: bool = True
    unit_type: str = "message"
    parent_message_id: int = 0
    segment_id: int = 0
    segment_index: int = -1


@dataclass
class RankedHit:
    chunk_id: int
    user_id: int
    conversation_id: int
    message_id: int
    role: str
    position: int
    content: str
    created_at: int
    turn_id: int = 0
    unit_type: str = "message"
    unit_id: int = 0
    parent_message_id: int = 0
    segment_id: int = 0
    segment_index: int = -1
    source_query: str = ""
    semantic_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    fused_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class RetrievalAnchor:
    unit_type: str
    unit_id: int
    parent_message_id: int
    role: str
    content: str
    score: float
    source_query: str
    chunk_id: int = 0
    user_id: int = 0
    conversation_id: int = 0
    position: int = 0
    created_at: int = 0
    segment_id: int = 0
    segment_index: int = -1
    core_unit_ids: List[int] = field(default_factory=list)


@dataclass
class RankedTurn:
    """Legacy turn-level anchor used by SpanStitcher."""
    conversation_id: int
    turn_id: int
    score: float
    anchor_message_ids: List[int]
    anchor_roles: List[str]
    hits: List[RankedHit] = field(default_factory=list)


@dataclass
class SegmentRegion:
    parent_message_id: int
    role: str
    core_segment_ids: List[int]
    before_segment_ids: List[int]
    after_segment_ids: List[int]
    omitted_before: bool
    omitted_after: bool
    token_count: int
    total_segment_count: int


@dataclass
class RetrievedFragment:
    fragment_type: str
    anchor_role: str
    parent_message_id: int
    score: float
    core_unit_ids: List[int]
    expanded_unit_ids: List[int]
    reply_context_message_id: int | None
    earlier_user_message_ids: List[int]
    later_user_message_ids: List[int]
    omitted_before: bool
    omitted_after: bool
    token_count: int
    conversation_id: int = 0
    created_at: int = 0
    segment_region: SegmentRegion | None = None
    core_contents: List[str] = field(default_factory=list)
    expanded_contents: List[str] = field(default_factory=list)
    reply_context_content: str = ""
    earlier_user_contents: List[str] = field(default_factory=list)
    later_user_contents: List[str] = field(default_factory=list)
    core_message_content: str = ""


@dataclass(frozen=True)
class RoutedQuery:
    query: str
    role_scope: str = "both"


@dataclass
class RetrievalPlan:
    mode: str  # direct | split
    queries: List[str]
    subqueries: List[RoutedQuery] = field(default_factory=list)
    original_role_scope: str = "both"


@dataclass
class ProfileItem:
    id: str
    user_id: int
    content: str
    origin: str
    source_message_ids: List[int]
    status: str
    created_at: int
    updated_at: int


@dataclass
class PendingProfile:
    id: str
    user_id: int
    content: str
    source_message_ids: List[int]
    created_at: int


@dataclass
class RetrievalTrace:
    mode: str = "direct"
    subquery_count: int = 1
    semantic_hits: int = 0
    lexical_hits: int = 0
    anchors: int = 0
    bundles: int = 0
    selected_bundles: int = 0
    profile_hits: int = 0
    memory_tokens: int = 0
    latency_ms: int = 0
    fallback: bool = False
    enough: bool = True
    queries: List[str] = field(default_factory=list)
    stitch_mode: str = ""
    selection_mode: str = ""
    linked_chains: int = 0
    original_query: str = ""
    planner_subqueries: List[Dict[str, Any]] = field(default_factory=list)
    candidate_unit_ids: List[int] = field(default_factory=list)
    reranked_unit_ids: List[int] = field(default_factory=list)
    selected_core_unit_ids: List[int] = field(default_factory=list)
    expanded_segment_ids: List[int] = field(default_factory=list)
    reply_context_message_ids: List[int] = field(default_factory=list)
    expanded_user_message_ids: List[int] = field(default_factory=list)
    included_parent_message_ids: List[int] = field(default_factory=list)
    included_unit_ids: List[int] = field(default_factory=list)
    fragment_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "subquery_count": self.subquery_count,
            "semantic_hits": self.semantic_hits,
            "lexical_hits": self.lexical_hits,
            "anchors": self.anchors,
            "bundles": self.bundles,
            "selected_bundles": self.selected_bundles,
            "profile_hits": self.profile_hits,
            "memory_tokens": self.memory_tokens,
            "latency_ms": self.latency_ms,
            "fallback": self.fallback,
            "enough": self.enough,
            "stitch_mode": self.stitch_mode,
            "selection_mode": self.selection_mode,
            "linked_chains": self.linked_chains,
            "original_query": self.original_query,
            "planner_subqueries": list(self.planner_subqueries),
            "candidate_unit_ids": list(self.candidate_unit_ids),
            "reranked_unit_ids": list(self.reranked_unit_ids),
            "selected_core_unit_ids": list(self.selected_core_unit_ids),
            "expanded_segment_ids": list(self.expanded_segment_ids),
            "reply_context_message_ids": list(self.reply_context_message_ids),
            "expanded_user_message_ids": list(self.expanded_user_message_ids),
            "included_parent_message_ids": list(self.included_parent_message_ids),
            "included_unit_ids": list(self.included_unit_ids),
            "fragment_count": self.fragment_count,
            "extra": dict(self.extra),
        }


# Legacy window types kept for builder compatibility during migration.
@dataclass
class SpanTurn:
    message_id: int
    conversation_id: int
    role: str
    position: int
    content: str
    created_at: int
    turn_id: int = 0
    segment: str = "anchor"
    is_anchor: bool = False
    matched_chunk: Optional[str] = None


@dataclass
class Span:
    bundle_id: str
    conversation_id: int
    anchor_ids: List[int]
    messages: List[SpanTurn]
    fused_score: float
    anchor_turn_id: int = 0
    rerank_score: Optional[float] = None
    retrieval_queries: Optional[List[str]] = None
    chain_id: Optional[str] = None
    chain_index: int = 0
    fragment: RetrievedFragment | None = None
