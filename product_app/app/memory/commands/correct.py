from __future__ import annotations

from typing import Optional, Sequence

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.service import ProfileService


def handle_correct(
    profile: ProfileService,
    *,
    user_id: int,
    old_keyword: str,
    new_content: str | None,
    source_message_ids: Sequence[int],
) -> Optional[ProfileItem]:
    """Thin wrapper kept for command exports; logic lives on ProfileService."""
    return profile.correct(
        user_id=user_id,
        old_keyword=old_keyword,
        new_content=new_content,
        source_message_ids=list(source_message_ids),
    )
