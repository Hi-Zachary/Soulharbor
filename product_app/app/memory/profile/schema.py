"""Strict allowlist for long-term user profile fields (narrower than MemMachine)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple

# tag -> allowed features. Only these may be written.
ALLOWED: Dict[str, FrozenSet[str]] = {
    "identity": frozenset({"name", "gender", "age"}),
    "education": frozenset({"school", "grade", "major", "degree"}),
    "location": frozenset({"hometown", "residence"}),
    "preference": frozenset({"communication", "hobby", "lifestyle", "other"}),
}

TAG_LABELS = {
    "identity": "身份",
    "education": "教育",
    "location": "地域",
    "preference": "偏好",
}

FEATURE_LABELS = {
    "name": "姓名",
    "gender": "性别",
    "age": "年龄",
    "school": "学校",
    "grade": "年级",
    "major": "专业",
    "degree": "学历",
    "hometown": "家乡",
    "residence": "常住地",
    "communication": "沟通偏好",
    "hobby": "兴趣爱好",
    "lifestyle": "生活习惯",
    "other": "其他偏好",
}


@dataclass(frozen=True)
class ProfileFact:
    tag: str
    feature: str
    value: str

    @property
    def key(self) -> str:
        return f"{self.tag}/{self.feature}"

    def to_content(self) -> str:
        label = FEATURE_LABELS.get(self.feature, self.feature)
        return f"[{self.key}] {label}：{self.value}"


def parse_content_key(content: str) -> Optional[Tuple[str, str]]:
    """Extract tag/feature from stored content prefix `[tag/feature] …`."""
    text = (content or "").strip()
    if not text.startswith("["):
        return None
    end = text.find("]")
    if end <= 1:
        return None
    key = text[1:end].strip()
    if "/" not in key:
        return None
    tag, feature = key.split("/", 1)
    tag, feature = tag.strip(), feature.strip()
    if tag and feature:
        return tag, feature
    return None


def is_allowed(tag: str, feature: str) -> bool:
    feats = ALLOWED.get((tag or "").strip())
    if not feats:
        return False
    return (feature or "").strip() in feats


def normalize_fact(
    *,
    tag: str,
    feature: str,
    value: str,
) -> Optional[ProfileFact]:
    t = (tag or "").strip().lower()
    f = (feature or "").strip().lower()
    v = (value or "").strip()
    if not t or not f or not v:
        return None
    if not is_allowed(t, f):
        return None
    if len(v) > 80:
        v = v[:80].rstrip()
    if len(v) < 1:
        return None
    return ProfileFact(tag=t, feature=f, value=v)


def allowed_prompt_block() -> str:
    lines = []
    for tag, feats in ALLOWED.items():
        label = TAG_LABELS.get(tag, tag)
        feat_txt = "、".join(
            f"{f}({FEATURE_LABELS.get(f, f)})" for f in sorted(feats)
        )
        lines.append(f"- {tag}（{label}）：{feat_txt}")
    return "\n".join(lines)


def all_keys() -> Set[str]:
    return {f"{t}/{f}" for t, feats in ALLOWED.items() for f in feats}
