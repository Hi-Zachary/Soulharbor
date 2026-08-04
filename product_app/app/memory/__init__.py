"""Long-term memory package.

Submodules are importable without loading the chat LLM stack.
`MemoryService` / `MemoryEngine` resolve lazily on attribute access.
"""
from __future__ import annotations

from typing import Any

__all__ = ["MemoryService", "MemoryEngine"]


def __getattr__(name: str) -> Any:
    if name == "MemoryService":
        from product_app.app.memory.service import MemoryService

        return MemoryService
    if name == "MemoryEngine":
        from product_app.app.memory.engine import MemoryEngine

        return MemoryEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
