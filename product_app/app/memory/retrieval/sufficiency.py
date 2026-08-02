"""Whether we have anything useful to inject."""
from __future__ import annotations

from typing import List, Optional

from product_app.app.memory.models import Span, ProfileItem


def is_enough(
    windows: List[Span],
    profiles: Optional[List[ProfileItem]] = None,
    *,
    min_items: int = 1,
) -> bool:
    return len(windows) + len(profiles or []) >= min_items
