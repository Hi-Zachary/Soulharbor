"""Split long messages into retrieval segments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.token_utils import count_tokens

_PARAGRAPH = re.compile(r"\n\s*\n+")
_SENTENCE = re.compile(r"(?<=[。！？!?])")
_SEMICOLON = re.compile(r"(?<=[；;])")
_COMMA = re.compile(r"(?<=[，、,])")


@dataclass(frozen=True)
class SplitSegment:
    segment_index: int
    content: str
    start_offset: int
    end_offset: int
    token_count: int


def _split_by_pattern(text: str, pattern: re.Pattern[str]) -> List[str]:
    parts = [p.strip() for p in pattern.split(text) if p and p.strip()]
    return parts if parts else [text.strip()]


def _merge_to_token_budget(pieces: List[str], *, target: int, hard_max: int) -> List[str]:
    if not pieces:
        return []
    merged: List[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        candidate = current + piece
        if count_tokens(candidate) <= hard_max:
            current = candidate
            continue
        if count_tokens(current) >= max(1, target // 2):
            merged.append(current)
            current = piece
        else:
            current = candidate
    if current:
        merged.append(current)

    out: List[str] = []
    for chunk in merged:
        if count_tokens(chunk) <= hard_max:
            out.append(chunk)
            continue
        # Hard token split as last resort.
        step = max(1, hard_max * 2)
        for i in range(0, len(chunk), step):
            out.append(chunk[i : i + step])
    return out


def split_message(content: str) -> List[SplitSegment]:
    text = (content or "").strip()
    if not text:
        return []

    paragraphs = _split_by_pattern(text, _PARAGRAPH)
    units: List[str] = []
    for para in paragraphs:
        if count_tokens(para) <= int(mem_cfg.segment_hard_max_tokens):
            units.append(para)
            continue
        for level in (_SENTENCE, _SEMICOLON, _COMMA):
            parts = _split_by_pattern(para, level)
            if len(parts) > 1:
                units.extend(
                    _merge_to_token_budget(
                        parts,
                        target=int(mem_cfg.segment_target_tokens),
                        hard_max=int(mem_cfg.segment_hard_max_tokens),
                    )
                )
                break
        else:
            units.extend(
                _merge_to_token_budget(
                    [para],
                    target=int(mem_cfg.segment_target_tokens),
                    hard_max=int(mem_cfg.segment_hard_max_tokens),
                )
            )

    segments: List[SplitSegment] = []
    cursor = 0
    for idx, piece in enumerate(units):
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        cursor = end
        segments.append(
            SplitSegment(
                segment_index=idx,
                content=piece,
                start_offset=start,
                end_offset=end,
                token_count=count_tokens(piece),
            )
        )
    return segments
