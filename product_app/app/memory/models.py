from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    user_id: int
    conversation_id: int
    message_id: int
    role: Role
    content: str
    position: int
    created_at: int


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
    is_deleted: bool = False
    embedding: Optional[List[float]] = None


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
    semantic_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    fused_score: float = 0.0


@dataclass
class SpanTurn:
    message_id: int
    conversation_id: int
    role: str
    position: int
    content: str
    created_at: int
    is_anchor: bool = False


@dataclass
class Span:
    bundle_id: str
    conversation_id: int
    anchor_ids: List[int]
    messages: List[SpanTurn]
    fused_score: float
    rerank_score: Optional[float] = None
    retrieval_queries: Optional[List[str]] = None
    chain_id: Optional[str] = None
    chain_index: int = 0


@dataclass
class ProfileItem:
    id: str
    user_id: int
    content: str
    origin: str  # explicit | confirmed
    source_message_ids: List[int]
    status: str  # active | deleted
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
class RetrievalPlan:
    mode: str  # direct | split
    queries: List[str]


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
        }
