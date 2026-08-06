"""Build message-level or segment-level index units (mutually exclusive)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.models import searchable_text
from product_app.app.memory.store.message_segment import SplitSegment, split_message
from product_app.app.memory.token_utils import count_tokens


@dataclass(frozen=True)
class IndexUnit:
    unit_type: str  # message | segment
    unit_id: int
    parent_message_id: int
    role: str
    content: str
    segment_index: int = -1
    segment_id: int = 0
    start_offset: int = 0
    end_offset: int = 0
    token_count: int = 0


def build_index_units(
    *,
    message_id: int,
    role: str,
    content: str,
    segment_id_start: int = 1,
) -> tuple[bool, List[IndexUnit], List[SplitSegment]]:
    body = (content or "").strip()
    token_count = count_tokens(body)
    if token_count <= int(mem_cfg.message_split_threshold):
        return (
            False,
            [
                IndexUnit(
                    unit_type="message",
                    unit_id=int(message_id),
                    parent_message_id=int(message_id),
                    role=str(role),
                    content=body,
                    token_count=token_count,
                )
            ],
            [],
        )

    segments = split_message(body)
    if not segments:
        return (
            False,
            [
                IndexUnit(
                    unit_type="message",
                    unit_id=int(message_id),
                    parent_message_id=int(message_id),
                    role=str(role),
                    content=body,
                    token_count=token_count,
                )
            ],
            [],
        )

    units: List[IndexUnit] = []
    for offset, seg in enumerate(segments):
        sid = int(segment_id_start) + offset
        units.append(
            IndexUnit(
                unit_type="segment",
                unit_id=sid,
                parent_message_id=int(message_id),
                role=str(role),
                content=seg.content,
                segment_index=int(seg.segment_index),
                segment_id=sid,
                start_offset=int(seg.start_offset),
                end_offset=int(seg.end_offset),
                token_count=int(seg.token_count),
            )
        )
    return True, units, segments


def index_searchable_text(unit: IndexUnit) -> str:
    return searchable_text(unit.role, unit.content)
