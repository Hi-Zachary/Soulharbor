from __future__ import annotations

from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.store.repository import EpisodeStore


def handle_inspect(*, repo: EpisodeStore, profile: ProfileService, user_id: int) -> dict:
    """Compact snapshot for 'what do you remember about me?'."""
    prefs = profile.list_active(user_id)
    stats = repo.index_stats(user_id)
    return {
        "support_preferences": [
            {"id": item.id, "content": item.content, "origin": item.origin}
            for item in prefs
        ],
        "episode_chunks": stats.get("chunks", 0),
        "embeddings": stats.get("embeddings", 0),
    }
