"""Render atomic profiles into a single <user_profile> injection block."""
from __future__ import annotations

from collections.abc import Sequence
from html import escape

from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.operations import normalize_profile_content


def render_user_profile(profiles: Sequence[ProfileItem]) -> str:
    contents: list[str] = []
    for profile in profiles:
        content = normalize_profile_content(profile.content)
        if content:
            contents.append(escape(content, quote=False))
    if not contents:
        return ""
    return "<user_profile>\n" + "".join(contents) + "\n</user_profile>"
