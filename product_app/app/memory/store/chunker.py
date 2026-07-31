"""Split long messages into smaller pieces for embedding."""
from __future__ import annotations

import re
from typing import List

from product_app.app.memory.config import mem_cfg

# Keep punctuation attached to the preceding clause.
_CLAUSE_BREAK = re.compile(r"(?<=[。！？；\n])")


def chunk_text(content: str) -> List[str]:
    text = (content or "").strip()
    if not text:
        return []

    soft_limit = int(mem_cfg.chunk_soft_limit)
    if len(text) <= soft_limit:
        return [text]

    clauses = [c.strip() for c in _CLAUSE_BREAK.split(text) if c and c.strip()]
    if not clauses:
        # Fall back to hard slices when we cannot find sentence boundaries.
        return [text[i : i + soft_limit] for i in range(0, len(text), soft_limit)]

    min_len = int(mem_cfg.chunk_target_min)
    max_len = int(mem_cfg.chunk_target_max)
    chunks: List[str] = []
    current = ""

    for clause in clauses:
        if not current:
            current = clause
            continue
        if len(current) + len(clause) <= max_len:
            current += clause
            continue
        if len(current) < min_len and chunks:
            chunks[-1] += current
        else:
            chunks.append(current)
        current = clause

    if current:
        if len(current) < min_len and chunks:
            chunks[-1] += current
        else:
            chunks.append(current)

    return chunks if chunks else [text]
