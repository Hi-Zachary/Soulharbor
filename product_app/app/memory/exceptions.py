"""Memory-domain errors. Chat must keep going even if recall fails."""
from __future__ import annotations


class MemoryError(Exception):
    """Base class for memory failures."""


class MemoryIngestError(MemoryError):
    """Failed while writing a turn into the store."""


class MemoryRetrievalError(MemoryError):
    """Failed while building a recall block."""


class MemoryProfileError(MemoryError):
    """Failed while updating long-term profile facts."""
