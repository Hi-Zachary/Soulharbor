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
from product_app.app.memory.models import Block, RankedHit, RankedTurn, Span, SpanTurn
from product_app.app.memory.store.lexical_search import LexicalSearcher
from product_app.app.memory.store.merge import merge_windows
from product_app.app.memory.store.rerank import (
    _select_direct,
    _select_split,
    _take_unique_messages,
)


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
        turn_id=mid,
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
        anchor_turn_id=turns[0].turn_id if turns else 0,
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
        turn_id=mid,
        fused_score=fused,
    )


class DefaultLimitTests(unittest.TestCase):
    def test_default_limits_are_ten(self) -> None:
        keys = (
            "MEMORY_ANCHOR_CE_TOP_K",
            "MEMORY_ANCHOR_CE_DIRECT_K",
            "MEMORY_ANCHOR_CE_PER_SUBQUERY_K",
            "MEMORY_WINDOW_TOP_K",
        )
        saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
        try:
            import product_app.app.memory.config as config_mod

            config_mod = importlib.reload(config_mod)
            self.assertEqual(config_mod.mem_cfg.anchor_ce_top_k, 10)
            self.assertEqual(config_mod.mem_cfg.anchor_ce_direct_k, 10)
            self.assertEqual(config_mod.mem_cfg.anchor_ce_per_subquery_k, 3)
            self.assertEqual(config_mod.mem_cfg.bundle_top_k, 10)
            self.assertFalse(hasattr(config_mod.mem_cfg, "anchor_ce_orig_top"))
        finally:
            os.environ.update(saved)
            import product_app.app.memory.config as config_mod

            importlib.reload(config_mod)


class AnchorCeSelectTests(unittest.TestCase):
    def test_select_direct_dedupes_message_id(self) -> None:
        anchors = [
            _hit(1, "a0", fused=0.1),
            _hit(1, "a1", fused=0.2),
            _hit(2, "b", fused=0.3),
            _hit(3, "c", fused=0.4),
        ]
        scores = [0.9, 0.95, 0.8, 0.7]
        picked = _select_direct(anchors, scores, limit=2)
        self.assertEqual([h.message_id for h in picked], [1, 2])
        self.assertEqual(picked[0].content, "a1")
        self.assertAlmostEqual(picked[0].rerank_score, 0.95)

    def test_select_split_union_then_fill_from_original(self) -> None:
        # 10 distinct messages; three subqueries each take top-3 with overlap.
        anchors = [_hit(i, f"m{i}") for i in range(1, 11)]
        # original prefers high ids
        original = [0.10 * i for i in range(1, 11)]
        # A: 1,2,3  B: 2,4,5  C: 6,7,8
        sub_a = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        sub_b = [0.1, 0.85, 0.1, 0.8, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1]
        sub_c = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.8, 0.7, 0.1, 0.1]
        picked = _select_split(
            anchors,
            [original, sub_a, sub_b, sub_c],
            per_subquery=3,
            limit=10,
        )
        mids = {h.message_id for h in picked}
        self.assertEqual(mids, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        self.assertEqual(len(picked), 10)

    def test_select_split_does_not_pad_beyond_available(self) -> None:
        anchors = [_hit(i, f"m{i}") for i in range(1, 7)]
        original = [0.1] * 6
        sub_a = [0.9, 0.8, 0.7, 0.0, 0.0, 0.0]
        sub_b = [0.0, 0.0, 0.0, 0.9, 0.8, 0.7]
        picked = _select_split(
            anchors,
            [original, sub_a, sub_b],
            per_subquery=3,
            limit=10,
        )
        self.assertEqual(len(picked), 6)
        self.assertEqual({h.message_id for h in picked}, set(range(1, 7)))


class BudgetPackTests(unittest.TestCase):
    def test_budget_can_pack_fewer_than_ten(self) -> None:
        windows = _make_windows(10)

        def counter(text: str) -> int:
            cost = 40
            for i in range(10):
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
        self.assertLess(len(packed), 10)
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
    def test_lexical_search_keeps_assistant_candidates(self) -> None:
        class FakeStore:
            def list_active_with_embeddings(self, user_id, limit=5000, role_scope="both"):
                del user_id, limit, role_scope
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
                        turn_id=10,
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
                        turn_id=11,
                        embedding=[0.1],
                    ),
                ]

        hits = LexicalSearcher(FakeStore()).search(user_id=1, query="推免 导师", limit=10)
        self.assertTrue(hits)
        self.assertEqual({h.role for h in hits}, {"assistant", "user"})
        self.assertEqual({h.message_id for h in hits}, {1, 2})


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


class EmptyMemoryBlockTests(unittest.TestCase):
    def test_format_sections_empty_without_evidence(self) -> None:
        from product_app.app.memory.context.builder import build_memory_block
        from product_app.app.memory.context.formatter import format_sections

        self.assertEqual(format_sections(bundles=[], profiles=[], query="x"), [])
        block, packed = build_memory_block(
            bundles=[],
            profiles=[],
            token_budget=1600,
            query="x",
        )
        self.assertEqual(block, "")
        self.assertEqual(packed, 0)


class UpsertStaleChunkTests(unittest.TestCase):
    def test_upsert_drops_stale_higher_index_chunks(self) -> None:
        import tempfile
        from pathlib import Path

        from product_app.app.memory.store.repository import TraceStore

        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "mem.db")
            store.init()
            store.upsert_chunks(
                user_id=1,
                conversation_id=1,
                message_id=42,
                turn_id=42,
                role="user",
                position=1,
                created_at=100,
                retrievable=True,
                visible_to_user=True,
                is_final=True,
                chunks=["chunk-0", "chunk-1", "chunk-2"],
            )
            store.upsert_chunks(
                user_id=1,
                conversation_id=1,
                message_id=42,
                turn_id=42,
                role="user",
                position=1,
                created_at=200,
                retrievable=True,
                visible_to_user=True,
                is_final=True,
                chunks=["new-only"],
            )
            with store._db() as conn:
                rows = conn.execute(
                    "SELECT chunk_index, content FROM memory_blocks "
                    "WHERE message_id=42 ORDER BY chunk_index"
                ).fetchall()
            self.assertEqual([(int(r["chunk_index"]), r["content"]) for r in rows], [(0, "new-only")])


class AnnMessageLevelTests(unittest.TestCase):
    def test_ann_index_build_keeps_message_level_rows(self) -> None:
        from product_app.app.memory.store.ann_index import UserAnnCache

        cache = UserAnnCache()
        rows = [
            Block(
                id=1,
                user_id=1,
                conversation_id=1,
                message_id=1,
                role="assistant",
                position=1,
                chunk_index=0,
                content="assistant",
                created_at=1,
                turn_id=10,
                embedding=[1.0, 0.0],
            ),
            Block(
                id=2,
                user_id=1,
                conversation_id=1,
                message_id=2,
                role="user",
                position=2,
                chunk_index=0,
                content="user",
                created_at=2,
                turn_id=11,
                embedding=[0.0, 1.0],
            ),
        ]
        built = cache.get_or_build(1, (1, 2, 3), rows, kind="message_level")
        if built is None:
            self.skipTest("faiss not installed in this environment")
        self.assertEqual({r.role for r in built.rows}, {"assistant", "user"})
        self.assertEqual(built.kind, "message_level")
        self.assertIs(cache.peek(1, kind="message_level"), built)
        self.assertIsNone(cache.peek(1, kind="legacy_mixed"))


class TurnAwareStitchTests(unittest.TestCase):
    def test_same_turn_helpers_and_stitch_keep_assistant_anchor(self) -> None:
        import tempfile
        from pathlib import Path

        from product_app.app.memory.store.repository import TraceStore
        from product_app.app.memory.store.stitch import SpanStitcher

        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "mem.db")
            store.init()
            store.upsert_chunks(
                user_id=1,
                conversation_id=7,
                message_id=100,
                turn_id=500,
                role="user",
                position=1,
                created_at=100,
                retrievable=True,
                visible_to_user=True,
                is_final=True,
                chunks=["我想申请推免"],
            )
            store.upsert_chunks(
                user_id=1,
                conversation_id=7,
                message_id=101,
                turn_id=500,
                role="assistant",
                position=2,
                created_at=101,
                retrievable=True,
                visible_to_user=True,
                is_final=True,
                chunks=["建议先联系导师并准备材料"],
            )
            store.upsert_chunks(
                user_id=1,
                conversation_id=7,
                message_id=102,
                turn_id=600,
                role="user",
                position=3,
                created_at=102,
                retrievable=True,
                visible_to_user=True,
                is_final=True,
                chunks=["后续我开始整理科研经历"],
            )
            anchor_rows = store.list_turn_messages(user_id=1, conversation_id=7, turn_id=500)
            self.assertEqual([row["message_id"] for row in anchor_rows], [100, 101])

            stitcher = SpanStitcher(store)
            windows = stitcher.stitch(
                user_id=1,
                anchors=[
                    RankedTurn(
                        conversation_id=7,
                        turn_id=500,
                        score=0.9,
                        anchor_message_ids=[101],
                        anchor_roles=["assistant"],
                        hits=[_hit(101, "建议先联系导师并准备材料", role="assistant")],
                    )
                ],
                queries=["导师建议"],
            )
            self.assertEqual(len(windows), 1)
            self.assertEqual([t.role for t in windows[0].messages if t.segment == "anchor"], ["user", "assistant"])
            self.assertEqual([t.role for t in windows[0].messages if t.segment != "anchor"], ["user"])

    def test_hidden_nonfinal_assistant_not_listed(self) -> None:
        import tempfile
        from pathlib import Path

        from product_app.app.memory.store.repository import TraceStore

        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "mem.db")
            store.init()
            store.upsert_chunks(
                user_id=1,
                conversation_id=1,
                message_id=1,
                turn_id=1,
                role="assistant",
                position=1,
                created_at=1,
                retrievable=False,
                visible_to_user=False,
                is_final=False,
                chunks=["内部草稿"],
            )
            self.assertEqual(store.list_active_with_embeddings(1), [])


class FormatterTests(unittest.TestCase):
    def test_noncontiguous_sections_render_without_trace_ids(self) -> None:
        from product_app.app.memory.context.formatter import format_sections

        window = Span(
            bundle_id="w1",
            conversation_id=1,
            anchor_turn_id=10,
            anchor_ids=[2],
            messages=[
                SpanTurn(message_id=1, conversation_id=1, role="user", position=1, content="之前的用户消息", created_at=1, turn_id=9, segment="earlier"),
                SpanTurn(message_id=2, conversation_id=1, role="user", position=2, content="请给我推免建议", created_at=2, turn_id=10, segment="anchor", is_anchor=True, matched_chunk="推免建议"),
                SpanTurn(message_id=3, conversation_id=1, role="assistant", position=3, content="建议先联系导师", created_at=3, turn_id=10, segment="anchor"),
                SpanTurn(message_id=4, conversation_id=1, role="user", position=4, content="我之后会补材料", created_at=4, turn_id=11, segment="later"),
            ],
            fused_score=1.0,
            rerank_score=1.0,
        )
        lines = format_sections(bundles=[window], profiles=[], query="导师建议")
        text = "\n".join(lines)
        self.assertIn("[较早的用户消息]", text)
        self.assertIn("[检索命中的对话轮次]", text)
        self.assertIn("[后续用户消息]", text)
        self.assertIn("助手", text)
        self.assertNotIn("message=", text)


if __name__ == "__main__":
    unittest.main()
