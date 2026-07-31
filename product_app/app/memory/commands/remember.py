from __future__ import annotations

from typing import Optional

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.service import ProfileService


def handle_remember(
    profile: ProfileService,
    *,
    user_id: int,
    content: str,
    source_message_ids: list[int],
) -> Optional[ProfileItem]:
    """Explicit 'please remember …' path used by admin/tools."""
    return profile.create_explicit(
        user_id=user_id,
        content=content,
        source_message_ids=source_message_ids,
    )
