"""API-backed replacements for SoulHarborLLM and MemoryEmbedder (duck-typed, no project change)."""
from __future__ import annotations

from typing import Any, Dict, List

from api_client import chat_completion, embed_texts

try:
    from product_app.app.generation_log import GenerationLog
except Exception:  # pragma: no cover
    GenerationLog = None  # type: ignore[misc,assignment]


def _normalize_messages(messages: Any, system_text: str = "") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if system_text:
        out.append({"role": "system", "content": str(system_text)})
    if isinstance(messages, str):
        out.append({"role": "user", "content": messages})
    elif isinstance(messages, list):
        for m in messages:
            out.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
    return out


class APILLM:
    """Drop-in for SoulHarborLLM. generate_base == generate_structured for API (no LoRA)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def _chat(
        self,
        messages: Any,
        max_new_tokens: int,
        system_text: str,
        temperature: float,
        *,
        route: str,
    ) -> str:
        mt = int(max_new_tokens)
        # M2.x can't disable thinking -> reasoning eats the budget; bump so the answer still fits.
        # M3 with thinking=disabled needs no bump.
        if not self.cfg.get("thinking"):
            mt = max(mt, 2048)
        norm = _normalize_messages(messages, system_text)
        out = chat_completion(self.cfg, norm, max_tokens=mt, temperature=temperature)
        if GenerationLog is not None:
            GenerationLog.record(
                route=route,
                messages=norm,
                output=out or "",
                system_text=system_text or "",
                max_new_tokens=mt,
                temperature=temperature,
                chat_model=str(self.cfg.get("chat_model") or ""),
            )
        return out

    def generate_structured(self, messages: Any, *, max_new_tokens: int = 256, system_text: str = "") -> str:
        return self._chat(messages, max_new_tokens, system_text, 0.0, route="structured")

    def generate_base(self, messages: Any, *, max_new_tokens: int = 512, system_text: str = "") -> str:
        return self._chat(messages, max_new_tokens, system_text, 0.0, route="base")

    def generate_summary(self, messages: Any, *, max_new_tokens: int = 512, system_text: str = "") -> str:
        return self._chat(messages, max_new_tokens, system_text, 0.3, route="summary")

    def generate(self, messages: Any, *, max_new_tokens: int = 512, system_text: str = "", **kw: Any) -> str:
        return self._chat(messages, max_new_tokens, system_text, 0.7, route="chat")

    def count_tokens(self, text: str) -> int:
        return max(1, len(text or "") // 2)


class APIEmbedder:
    """Drop-in for MemoryEmbedder. Used only when config.embed_model is set."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self._dim: int = 0

    @property
    def model_id(self) -> str:
        return str(self.cfg.get("embed_model") or "api-embed")

    @property
    def embed_dim(self) -> int:
        return self._dim or 1024

    def embed(self, text: str) -> List[float]:
        vecs = embed_texts(self.cfg, [text])
        if vecs and vecs[0]:
            self._dim = len(vecs[0])
            return vecs[0]
        return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = embed_texts(self.cfg, texts)
        if vecs and vecs[0]:
            self._dim = len(vecs[0])
        return vecs
