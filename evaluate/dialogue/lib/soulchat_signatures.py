from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Set


def _norm(t: str) -> str:
    return "".join((t or "").split()).lower()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def signature_from_messages(msgs: List[Dict[str, Any]], *, max_msgs: int = 6) -> str:
    """
    Signature used to match SoulChatCorpus examples across different packagings.
    Mirrors the approach in scripts/alignment/build_dpo_pairs_from_soulchat_exclusive.py.
    """
    parts: List[str] = []
    for m in (msgs or [])[:max_msgs]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}:{_norm(content)}")
    return _sha1("|".join(parts))


def signature_from_sharegpt_conversations(convs: List[Dict[str, Any]], *, max_msgs: int = 6) -> str:
    """
    Compute a compatible signature from ShareGPT-style conversations:
      [{"from":"human"|"gpt","value": "..."}...]
    """
    parts: List[str] = []
    for m in (convs or [])[:max_msgs]:
        if not isinstance(m, dict):
            continue
        fr = str(m.get("from") or "").strip().lower()
        val = str(m.get("value") or "").strip()
        if not val:
            continue
        # Map to SoulChatCorpus role naming to match signature_from_messages()
        role = "user" if fr == "human" else ("assistant" if fr == "gpt" else fr)
        parts.append(f"{role}:{_norm(val)}")
    return _sha1("|".join(parts))


def load_used_sigs_from_sharegpt_jsonl(path: str, *, max_msgs: int = 6) -> Set[str]:
    import json
    from pathlib import Path

    sigs: Set[str] = set()
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            convs = obj.get("conversations")
            if not isinstance(convs, list) or not convs:
                continue
            sigs.add(signature_from_sharegpt_conversations(convs, max_msgs=max_msgs))
    return sigs


def load_used_sigs_from_dpo_jsonl(path: str, *, max_msgs: int = 6) -> Set[str]:
    """
    DPO JSONL format also contains 'conversations' with ShareGPT style.
    """
    return load_used_sigs_from_sharegpt_jsonl(path, max_msgs=max_msgs)

