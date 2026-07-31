"""Thin adapter over official mem0ai.Memory — no custom memory logic.

Shares one SentenceTransformer across adapters so parallel workers don't
reload bge-m3 repeatedly. Encode calls are locked (model isn't fully
thread-safe under concurrent encode).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ST_CACHE: Dict[str, Any] = {}
_ST_LOCK = threading.Lock()
_ENCODE_LOCK = threading.RLock()
_PATCHED = False
_PATCH_LOCK = threading.Lock()


def _ensure_shared_sentence_transformer() -> None:
    """Monkeypatch SentenceTransformer so identical model paths share one instance."""
    global _PATCHED
    if _PATCHED:
        return
    with _PATCH_LOCK:
        if _PATCHED:
            return
        import sentence_transformers as st_mod

        Orig = st_mod.SentenceTransformer

        def _factory(*args: Any, **kwargs: Any) -> Any:
            model_id = str(args[0] if args else kwargs.get("model_name_or_path") or "")
            device = str(kwargs.get("device") or "cpu")
            key = f"{model_id}::{device}"
            with _ST_LOCK:
                hit = _ST_CACHE.get(key)
                if hit is not None:
                    return hit
            # Load outside lock; only one winner is kept.
            obj = Orig(*args, **kwargs)
            orig_encode = obj.encode

            def _locked_encode(*a: Any, **k: Any) -> Any:
                with _ENCODE_LOCK:
                    return orig_encode(*a, **k)

            obj.encode = _locked_encode  # type: ignore[method-assign]
            with _ST_LOCK:
                existing = _ST_CACHE.get(key)
                if existing is not None:
                    return existing
                _ST_CACHE[key] = obj
                return obj

        st_mod.SentenceTransformer = _factory  # type: ignore[misc,assignment]
        try:
            import mem0.embeddings.huggingface as hf_mod

            hf_mod.SentenceTransformer = _factory  # type: ignore[attr-defined]
        except Exception:
            pass
        _PATCHED = True


def build_mem0_config(
    *,
    api_key: str,
    api_base_url: str,
    chat_model: str,
    embed_model_path: str,
    qdrant_path: str,
    collection_name: str = "mem0",
    embedding_dims: int = 1024,
) -> Dict[str, Any]:
    """Config for Memory.from_config: MiniMax LLM + local HF embedder + on-disk FAISS."""
    return {
        "llm": {
            "provider": "minimax",
            "config": {
                "model": chat_model,
                "api_key": api_key,
                "minimax_base_url": api_base_url,
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": embed_model_path,
                "embedding_dims": int(embedding_dims),
                "model_kwargs": {"device": "cpu"},
            },
        },
        "vector_store": {
            "provider": "faiss",
            "config": {
                "path": str(qdrant_path),
                "collection_name": collection_name,
                "embedding_model_dims": int(embedding_dims),
                "distance_strategy": "cosine",
            },
        },
        "history_db_path": str(Path(qdrant_path).parent / "history.db"),
        "version": "v1.1",
    }


class Mem0Adapter:
    """Wraps mem0.Memory with the eval harness ingest/retrieve/list_facts surface."""

    def __init__(
        self,
        *,
        work_dir: Path,
        api_cfg: Dict[str, Any],
        embed_model_path: str,
        user_id: str,
        top_k: int = 8,
    ) -> None:
        from mem0 import Memory

        _ensure_shared_sentence_transformer()
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._user_id = str(user_id)
        self._top_k = int(top_k)
        cfg = build_mem0_config(
            api_key=str(api_cfg["api_key"]),
            api_base_url=str(api_cfg.get("api_base_url") or "https://api.minimaxi.com/v1"),
            chat_model=str(api_cfg.get("chat_model") or "MiniMax-M3"),
            embed_model_path=embed_model_path,
            qdrant_path=str(work_dir / "faiss"),
            collection_name=f"mem0_{self._user_id}"[:64],
        )
        self._memory = Memory.from_config(cfg)

    def ingest_messages(self, messages: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        """Feed one conversation turn into Mem0 (official extract/update path)."""
        msgs = [
            {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
            for m in messages
            if str(m.get("content") or "").strip()
        ]
        if not msgs:
            return {"results": []}
        return self._memory.add(msgs, user_id=self._user_id)

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> str:
        k = int(top_k or self._top_k)
        out = self._memory.search(
            query or "",
            top_k=k,
            filters={"user_id": self._user_id},
            threshold=0.0,
        )
        rows = _unwrap_results(out)
        if not rows:
            return ""
        lines = ["[Mem0 memories]"]
        for row in rows:
            text = str(row.get("memory") or row.get("text") or "").strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def list_facts(self) -> List[str]:
        out = self._memory.get_all(filters={"user_id": self._user_id})
        rows = _unwrap_results(out)
        facts: List[str] = []
        for row in rows:
            text = str(row.get("memory") or row.get("text") or "").strip()
            if text:
                facts.append(text)
        return facts

    def store_snapshot(self) -> Dict[str, Any]:
        facts = self.list_facts()
        return {"n_facts": len(facts), "facts": facts}


def _unwrap_results(out: Any) -> List[Dict[str, Any]]:
    if isinstance(out, dict):
        rows = out.get("results")
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
        for key in ("memories", "data"):
            rows = out.get(key)
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        return []
    if isinstance(out, list):
        return [r for r in out if isinstance(r, dict)]
    return []
