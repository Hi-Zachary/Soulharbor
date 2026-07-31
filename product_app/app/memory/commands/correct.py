from __future__ import annotations

from typing import Optional

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.service import ProfileService


def handle_correct(
    profile: ProfileService,
    *,
    user_id: int,
    old_keyword: str,
    new_content: str | None,
    source_message_ids: list[int],
) -> Optional[ProfileItem]:
    """Drop preferences matching old_keyword; optionally write a replacement."""
    profile.forget_matching(user_id, old_keyword)
    replacement = (new_content or "").strip()
    if not replacement:
        return None
    return profile.create_explicit(
        user_id=user_id,
        content=replacement,
        source_message_ids=source_message_ids,
    )
