"""Guardrails for memory candidate caps and token-budget packing."""
from __future__ import annotations

import importlib
import os
import unittest

from product_app.app.memory.context.builder import pack_windows_to_budget
from product_app.app.memory.models import Span, SpanTurn


def _make_window(
    tag: str,
    *,
    score: float,
    body: str | None = None,
) -> Span:
    text = body if body is not None else f"WINDOW_{tag}_payload"
    return Span(
        bundle_id=tag,
        conversation_id=abs(hash(tag)) % 10_000,
        anchor_ids=[abs(hash(tag)) % 10_000],
        messages=[
            SpanTurn(
                message_id=abs(hash(tag)) % 10_000,
                conversation_id=abs(hash(tag)) % 10_000,
                role="user",
                position=1,
                content=text,
                created_at=1_700_000_000,
                is_anchor=True,
                matched_chunk=text[: min(32, len(text))],
            )
        ],
        fused_score=float(score),
        rerank_score=float(score),
    )


def _make_windows(n: int) -> list[Span]:
    return [
        _make_window(f"w{i}", score=1.0 - i * 0.01, body=f"WINDOW_w{i}_" + ("字" * 80))
        for i in range(n)
    ]


class DefaultLimitTests(unittest.TestCase):
    def test_default_limits_are_twelve(self) -> None:
        """Defaults must stay aligned so CE and post-merge caps do not diverge."""
        saved = {
            key: os.environ.pop(key)
            for key in ("MEMORY_ANCHOR_CE_TOP_K", "MEMORY_WINDOW_TOP_K")
            if key in os.environ
        }
        try:
            import product_app.app.memory.config as config_mod

            config_mod = importlib.reload(config_mod)
            self.assertEqual(config_mod.mem_cfg.anchor_ce_top_k, 12)
            self.assertEqual(config_mod.mem_cfg.bundle_top_k, 12)
        finally:
            os.environ.update(saved)
            import product_app.app.memory.config as config_mod

            importlib.reload(config_mod)


class BudgetPackTests(unittest.TestCase):
    def test_budget_can_pack_fewer_than_twelve(self) -> None:
        """Top-k=12 is a candidate ceiling; budget may inject fewer windows."""
        windows = _make_windows(12)

        def counter(text: str) -> int:
            # Each marked window costs ~250 tokens; overhead for tags.
            cost = 40
            for i in range(12):
                if f"WINDOW_w{i}_" in text:
                    cost += 250
            return cost

        packed = pack_windows_to_budget(
            windows,
            profiles=[],
            token_budget=900,
            token_counter=counter,
            query="test",
        )
        self.assertLess(len(packed), 12)
        self.assertGreaterEqual(len(packed), 1)

    def test_budget_skips_oversized_window_and_continues(self) -> None:
        """A middle window that does not fit must not stop later shorter candidates."""
        windows = [
            _make_window("a", score=0.9, body="WINDOW_a_short"),
            _make_window("b", score=0.8, body="WINDOW_b_huge"),
            _make_window("c", score=0.7, body="WINDOW_c_short"),
        ]
        per_window = {"a": 400, "b": 2000, "c": 200}

        def counter(text: str) -> int:
            cost = 50
            for tag, tokens in per_window.items():
                if f"WINDOW_{tag}_" in text:
                    cost += tokens
            return cost

        packed = pack_windows_to_budget(
            windows,
            profiles=[],
            token_budget=700,
            token_counter=counter,
            query="test",
        )
        self.assertEqual(packed, [windows[0], windows[2]])


if __name__ == "__main__":
    unittest.main()
