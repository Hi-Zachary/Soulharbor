from __future__ import annotations

from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.store.repository import EpisodeStore


def handle_forget(
    *,
    repo: EpisodeStore,
    profile: ProfileService,
    user_id: int,
    profile_id: str | None = None,
    keyword: str | None = None,
    message_id: int | None = None,
) -> dict:
    """Delete profile rows and/or episode chunks. Returns simple counters."""
    result = {"profiles": 0, "episodes": 0}
    if profile_id:
        result["profiles"] += int(profile.delete(user_id, profile_id))
    if keyword:
        result["profiles"] += profile.forget_matching(user_id, keyword)
        result["episodes"] += repo.soft_delete_by_keyword(user_id, keyword)
    if message_id is not None:
        result["episodes"] += repo.soft_delete_message(user_id, message_id)
    return result
