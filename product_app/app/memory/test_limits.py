"""Guardrails for ER read-path invariants."""
from __future__ import annotations

import importlib
import os
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from product_app.app.memory.context.builder import pack_windows_to_budget
from product_app.app.memory.context.formatter import _date_label, current_date_label
from product_app.app.memory.models import Block, RankedHit, Span, SpanTurn
from product_app.app.memory.store.lexical_search import LexicalSearcher
from product_app.app.memory.store.merge import merge_windows
from product_app.app.memory.store.rerank import _take_unique_messages


def _turn(
    *,
    mid: int,
    pos: int,
    conv: int = 1,
    anchor: bool = False,
    content: str | None = None,
) -> SpanTurn:
    return SpanTurn(
        message_id=mid,
        conversation_id=conv,
        role="user",
        position=pos,
        content=content or f"msg-{mid}",
        created_at=1_700_000_000 + pos,
        is_anchor=anchor,
        matched_chunk=f"msg-{mid}" if anchor else None,
    )


def _window(
    tag: str,
    turns: list[SpanTurn],
    *,
    conv: int = 1,
    score: float = 1.0,
) -> Span:
    return Span(
        bundle_id=tag,
        conversation_id=conv,
        anchor_ids=[t.message_id for t in turns if t.is_anchor],
        messages=turns,
        fused_score=score,
        rerank_score=score,
    )


def _make_window(
    tag: str,
    *,
    score: float,
    body: str | None = None,
) -> Span:
    text = body if body is not None else f"WINDOW_{tag}_payload"
    return _window(
        tag,
        [
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
        conv=abs(hash(tag)) % 10_000,
        score=score,
    )


def _make_windows(n: int) -> list[Span]:
    return [
        _make_window(f"w{i}", score=1.0 - i * 0.01, body=f"WINDOW_w{i}_" + ("字" * 80))
        for i in range(n)
    ]


def _hit(mid: int, content: str, *, role: str = "user", fused: float = 0.0) -> RankedHit:
    return RankedHit(
        chunk_id=mid,
        user_id=1,
        conversation_id=1,
        message_id=mid,
        role=role,
        position=1,
        content=content,
        created_at=0,
        fused_score=fused,
    )


class DefaultLimitTests(unittest.TestCase):
    def test_default_limits_are_twelve(self) -> None:
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
        windows = _make_windows(12)

        def counter(text: str) -> int:
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


class MergeTests(unittest.TestCase):
    def test_merge_keeps_all_anchors_beyond_max_messages(self) -> None:
        """Soft max_msgs only trims neighbors; CE anchors must survive."""
        # 9 anchors + 2 neighbors → would drop anchors under old [:max_msgs] logic.
        anchors = [_turn(mid=i, pos=i, anchor=True) for i in range(1, 10)]
        neighbors = [_turn(mid=100 + i, pos=20 + i, anchor=False) for i in range(2)]
        left = _window("L", anchors[:5] + neighbors[:1], score=0.9)
        right = _window("R", anchors[4:] + neighbors[1:], score=0.8)
        merged = merge_windows([left, right])
        self.assertEqual(len(merged), 1)
        kept_anchors = [t for t in merged[0].messages if t.is_anchor]
        self.assertEqual(len(kept_anchors), 9)
        self.assertEqual({t.message_id for t in kept_anchors}, set(range(1, 10)))

    def test_sparse_overlap_windows_merge_via_connected_component(self) -> None:
        """A:1,2,4 and C:4,5 share 4 even if B:3 sits between them in sort order."""
        a = _window("A", [_turn(mid=1, pos=1, anchor=True), _turn(mid=2, pos=2), _turn(mid=4, pos=4, anchor=True)], score=0.9)
        b = _window("B", [_turn(mid=3, pos=3, anchor=True)], score=0.8)
        c = _window("C", [_turn(mid=4, pos=4, anchor=True), _turn(mid=5, pos=5)], score=0.7)
        merged = merge_windows([a, b, c])
        # A∪C form one component; B has no shared positions → separate.
        self.assertEqual(len(merged), 2)
        positions = [sorted(t.position for t in w.messages) for w in merged]
        self.assertIn([1, 2, 4, 5], positions)
        self.assertIn([3], positions)


class RetrievalFilterTests(unittest.TestCase):
    def test_lexical_search_skips_assistant_candidates(self) -> None:
        class FakeStore:
            def list_active_with_embeddings(self, user_id, limit=5000):
                del user_id, limit
                return [
                    Block(
                        id=1,
                        user_id=1,
                        conversation_id=1,
                        message_id=1,
                        role="assistant",
                        position=1,
                        chunk_index=0,
                        content="推免 导师 科研 推免 导师 科研 推免",
                        created_at=1,
                        embedding=[0.1],
                    ),
                    Block(
                        id=2,
                        user_id=1,
                        conversation_id=1,
                        message_id=2,
                        role="user",
                        position=2,
                        chunk_index=0,
                        content="推免",
                        created_at=2,
                        embedding=[0.1],
                    ),
                ]

        hits = LexicalSearcher(FakeStore()).search(user_id=1, query="推免 导师", limit=10)
        self.assertTrue(hits)
        self.assertTrue(all(h.role == "user" for h in hits))
        self.assertEqual({h.message_id for h in hits}, {2})


class CeFallbackTests(unittest.TestCase):
    def test_take_unique_messages_fills_to_limit(self) -> None:
        anchors = [
            _hit(10, "a0"),
            _hit(10, "a1"),
            _hit(10, "a2"),
            *[_hit(20 + i, f"u{i}") for i in range(15)],
        ]
        picked = _take_unique_messages(anchors, 12)
        self.assertEqual(len(picked), 12)
        self.assertEqual(len({h.message_id for h in picked}), 12)
        self.assertEqual(picked[0].message_id, 10)
        self.assertEqual(picked[0].content, "a0")


class TimezoneTests(unittest.TestCase):
    def test_date_label_uses_configured_timezone(self) -> None:
        # 2024-01-01 00:30 Asia/Shanghai == 2023-12-31 16:30 UTC
        ts = int(datetime(2024, 1, 1, 0, 30, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())
        with patch("product_app.app.memory.context.formatter.mem_cfg") as cfg:
            cfg.memory_timezone = "Asia/Shanghai"
            self.assertEqual(_date_label(ts), "2024-01-01")
            cfg.memory_timezone = "UTC"
            self.assertEqual(_date_label(ts), "2023-12-31")

    def test_current_date_label_follows_timezone(self) -> None:
        fixed = datetime(2026, 8, 5, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

        class _FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz is not None else fixed

        with patch("product_app.app.memory.context.formatter.mem_cfg") as cfg:
            cfg.memory_timezone = "Asia/Shanghai"
            with patch("product_app.app.memory.context.formatter.datetime", _FixedDateTime):
                self.assertEqual(current_date_label(), "2026-08-05")


if __name__ == "__main__":
    unittest.main()
