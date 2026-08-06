"""Profile maintainer operations: shapes, normalize, validate (no evidence from LLM)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Set

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.token_utils import TokenCounter, count_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileOperation:
    op: Literal["add", "update", "delete"]
    target_id: str
    content: str


@dataclass(frozen=True)
class ProfileDecision:
    operations: list[ProfileOperation]


@dataclass
class AppliedProfileChanges:
    added_ids: List[str] = field(default_factory=list)
    updated_ids: List[str] = field(default_factory=list)
    deleted_ids: List[str] = field(default_factory=list)
    pruned_ids: List[str] = field(default_factory=list)
    skipped: int = 0

    def summary(self) -> Optional[str]:
        if not (
            self.added_ids
            or self.updated_ids
            or self.deleted_ids
            or self.pruned_ids
        ):
            return None
        return (
            "profile_maintained:"
            f"add={len(self.added_ids)},"
            f"upd={len(self.updated_ids)},"
            f"del={len(self.deleted_ids)},"
            f"prune={len(self.pruned_ids)}"
        )


def normalize_profile_content(content: str) -> str:
    value = re.sub(r"\s+", " ", content or "").strip()
    if not value:
        return ""
    if value[-1] not in "。！？!?":
        value += "。"
    return value


def validate_profile_operation(
    operation: ProfileOperation,
    *,
    active_profiles: Dict[str, ProfileItem],
    token_counter: TokenCounter,
    max_chars: int,
    max_tokens: int,
) -> tuple[bool, str, Optional[ProfileOperation]]:
    op = str(operation.op).strip().lower()
    target_id = str(operation.target_id or "").strip()
    content = normalize_profile_content(operation.content)

    if op not in {"add", "update", "delete"}:
        return False, "invalid_op", None

    if op == "add":
        if target_id:
            return False, "add_has_target", None
        if not content:
            return False, "empty_content", None
    elif op == "update":
        current = active_profiles.get(target_id)
        if current is None:
            return False, "unknown_target", None
        if not content:
            return False, "empty_content", None
        if normalize_profile_content(current.content) == content:
            return False, "unchanged", None
    elif op == "delete":
        if target_id not in active_profiles:
            return False, "unknown_target", None
        if content:
            return False, "delete_has_content", None
        return (
            True,
            "ok",
            ProfileOperation(op="delete", target_id=target_id, content=""),
        )

    if len(content) > int(max_chars):
        logger.info("profile op rejected: content_too_long len=%s", len(content))
        return False, "content_too_long", None
    if count_tokens(content, token_counter) > int(max_tokens):
        logger.info("profile op rejected: content_too_many_tokens")
        return False, "content_too_many_tokens", None

    return (
        True,
        "ok",
        ProfileOperation(op=op, target_id=target_id, content=content),  # type: ignore[arg-type]
    )


def filter_profile_operations(
    operations: Sequence[ProfileOperation],
    *,
    active_profiles: Dict[str, ProfileItem],
    token_counter: TokenCounter,
    max_operations: int,
    max_chars: int,
    max_tokens: int,
) -> List[ProfileOperation]:
    accepted: List[ProfileOperation] = []
    touched_ids: Set[str] = set()
    active_contents = {
        normalize_profile_content(item.content) for item in active_profiles.values()
    }
    pending_contents: Set[str] = set()

    for operation in list(operations)[: max(1, int(max_operations))]:
        valid, _reason, normalized = validate_profile_operation(
            operation,
            active_profiles=active_profiles,
            token_counter=token_counter,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )
        if not valid or normalized is None:
            continue
        if normalized.target_id and normalized.target_id in touched_ids:
            continue
        if normalized.op in {"add", "update"}:
            if normalized.content in active_contents:
                continue
            if normalized.content in pending_contents:
                continue
            pending_contents.add(normalized.content)
        if normalized.target_id:
            touched_ids.add(normalized.target_id)
        accepted.append(normalized)
    return accepted
