"""Tests for message/segment fragment retrieval."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from product_app.app.memory.config import mem_cfg
from product_app.app.memory.context.fragment_formatter import format_fragment_sections, format_segment_region
from product_app.app.memory.models import RankedHit, RetrievedFragment, Span, Turn
from product_app.app.memory.profile.maintainer import build_maintainer_system
from product_app.app.memory.retrieval.direct import DirectRetriever, retrieval_is_insufficient
from product_app.app.memory.retrieval.router import QueryRouter
from product_app.app.memory.store.expand import expand_message_anchor, expand_segment_anchor
from product_app.app.memory.store.fragments import (
    collapse_parent_anchors,
    trim_fragment_to_budget,
)
from product_app.app.memory.models import RetrievalAnchor
from product_app.app.memory.store.index_units import build_index_units
from product_app.app.memory.store.ingest import TraceIngestor
from product_app.app.memory.store.repository import TraceStore


class IndexUnitTests(unittest.TestCase):
    def test_short_message_is_indexed_as_message(self) -> None:
        has_segments, units, segments = build_index_units(
            message_id=1, role="user", content="短消息"
        )
        self.assertFalse(has_segments)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].unit_type, "message")
        self.assertEqual(segments, [])

    def test_long_message_indexes_only_segments(self) -> None:
        long_text = "。".join(["这是较长内容"] * 80)
        has_segments, units, segments = build_index_units(
            message_id=2, role="assistant", content=long_text
        )
        self.assertTrue(has_segments)
        self.assertGreater(len(units), 1)
        self.assertTrue(all(u.unit_type == "segment" for u in units))
        self.assertEqual(len(units), len(segments))


class PlannerTests(unittest.TestCase):
    def test_planner_user_scope_filters_assistant(self) -> None:
        plan = QueryRouter().plan("我之前说过什么")
        self.assertTrue(plan.subqueries)
        self.assertIn(plan.subqueries[0].role_scope, {"user", "both"})

    def test_low_confidence_role_scope_falls_back_to_both(self) -> None:
        self.assertTrue(retrieval_is_insufficient([]))
        self.assertTrue(
            retrieval_is_insufficient(
                [
                    RankedHit(
                        chunk_id=1,
                        user_id=1,
                        conversation_id=1,
                        message_id=1,
                        role="user",
                        position=1,
                        content="a",
                        created_at=1,
                    )
                ]
            )
        )

        class FakeSemantic:
            def search(self, **kwargs):
                role_scope = kwargs.get("role_scope", "both")
                if role_scope == "user":
                    return []
                return [
                    RankedHit(
                        chunk_id=2,
                        user_id=1,
                        conversation_id=1,
                        message_id=2,
                        role="assistant",
                        position=2,
                        content="助手内容",
                        created_at=2,
                    )
                ]

        class FakeLexical:
            def search(self, **kwargs):
                return FakeSemantic().search(**kwargs)

        retriever = DirectRetriever(FakeSemantic(), FakeLexical())  # type: ignore[arg-type]
        hits, _, _ = retriever.retrieve(user_id=1, query="测试", role_scope="user")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].role, "assistant")


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class ExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(Path(self.tmp.name) / "mem.db")
        self.store.init()
        self.embedder = _FakeEmbedder()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_conversation(self) -> None:
        ing = TraceIngestor(self.store, embedder=self.embedder)
        ing.ingest_message(
            Turn(
                user_id=1,
                conversation_id=1,
                message_id=1,
                role="user",
                content="较早的用户问题",
                position=1,
                created_at=1,
            )
        )
        ing.ingest_message(
            Turn(
                user_id=1,
                conversation_id=1,
                message_id=2,
                role="assistant",
                content="可以，先按第二种做。",
                position=2,
                created_at=2,
                reply_to_message_id=1,
            )
        )
        ing.ingest_message(
            Turn(
                user_id=1,
                conversation_id=1,
                message_id=3,
                role="user",
                content="后续用户跟进",
                position=3,
                created_at=3,
            )
        )

    def test_message_anchor_expands_only_user_messages(self) -> None:
        self._seed_conversation()
        with patch(
            "product_app.app.memory.store.expand._similarity",
            side_effect=lambda q, t, e: 0.9 if "用户" in t else 0.0,
        ):
            anchor = RetrievalAnchor(
                unit_type="message",
                unit_id=2,
                parent_message_id=2,
                role="assistant",
                content="可以，先按第二种做。",
                score=0.8,
                source_query="q",
            )
            earlier_ids, _, later_ids, _, reply_id, _ = expand_message_anchor(
                self.store,
                self.embedder,
                user_id=1,
                anchor=anchor,
                query="跟进",
            )
        self.assertEqual(reply_id, 1)
        self.assertTrue(all(isinstance(x, int) for x in earlier_ids + later_ids))

    def test_segment_anchor_expands_only_adjacent_segments(self) -> None:
        long_text = "第一段内容。" + "第二段核心。" + "第三段内容。"
        ing = TraceIngestor(self.store, embedder=self.embedder)
        ing.ingest_message(
            Turn(
                user_id=1,
                conversation_id=1,
                message_id=10,
                role="user",
                content=long_text * 40,
                position=1,
                created_at=1,
            )
        )
        segments = self.store.list_message_segments(user_id=1, parent_message_id=10)
        self.assertGreaterEqual(len(segments), 2)
        core = segments[1]
        with patch(
            "product_app.app.memory.store.expand._similarity",
            side_effect=lambda q, t, e: 0.9 if "第一段" in t or "第三段" in t else 0.0,
        ):
            region = expand_segment_anchor(
                self.store,
                self.embedder,
                user_id=1,
                parent_message_id=10,
                core_segment_ids=[int(core["id"])],
                query="核心",
            )
        self.assertLessEqual(len(region.before_segment_ids), 1)
        self.assertLessEqual(len(region.after_segment_ids), 1)

    def test_segment_expansion_stops_below_threshold(self) -> None:
        long_text = "第一段。" + "第二段核心。" + "第三段。"
        ing = TraceIngestor(self.store, embedder=self.embedder)
        ing.ingest_message(
            Turn(
                user_id=1,
                conversation_id=1,
                message_id=11,
                role="user",
                content=long_text * 40,
                position=1,
                created_at=1,
            )
        )
        segments = self.store.list_message_segments(user_id=1, parent_message_id=11)
        core = segments[1]
        with patch("product_app.app.memory.store.expand._similarity", return_value=0.0):
            region = expand_segment_anchor(
                self.store,
                self.embedder,
                user_id=1,
                parent_message_id=11,
                core_segment_ids=[int(core["id"])],
                query="核心",
            )
        self.assertEqual(region.before_segment_ids, [])
        self.assertEqual(region.after_segment_ids, [])

    def test_assistant_segment_includes_reply_context(self) -> None:
        self._seed_conversation()
        anchor = RetrievalAnchor(
            unit_type="message",
            unit_id=2,
            parent_message_id=2,
            role="assistant",
            content="可以，先按第二种做。",
            score=0.8,
            source_query="q",
        )
        with patch("product_app.app.memory.store.expand._similarity", return_value=0.0):
            _, _, _, _, reply_id, reply_content = expand_message_anchor(
                self.store,
                self.embedder,
                user_id=1,
                anchor=anchor,
                query="第二种",
            )
        self.assertEqual(reply_id, 1)
        self.assertIn("较早的用户问题", reply_content)

    def test_reply_context_does_not_trigger_expansion(self) -> None:
        self._seed_conversation()
        anchor = RetrievalAnchor(
            unit_type="message",
            unit_id=2,
            parent_message_id=2,
            role="assistant",
            content="可以，先按第二种做。",
            score=0.8,
            source_query="q",
        )
        with patch("product_app.app.memory.store.expand._similarity", return_value=0.0):
            earlier_ids, _, later_ids, _, reply_id, _ = expand_message_anchor(
                self.store,
                self.embedder,
                user_id=1,
                anchor=anchor,
                query="第二种",
            )
        self.assertEqual(reply_id, 1)
        self.assertEqual(earlier_ids, [])
        self.assertEqual(later_ids, [])


class FormatterTests(unittest.TestCase):
    def test_core_segment_is_explicitly_marked(self) -> None:
        body = format_segment_region(
            core_contents=["核心片段"],
            expanded_before=["补充前"],
            expanded_after=[],
            omitted_before=True,
            omitted_after=False,
        )
        self.assertIn("【核心命中】", body)
        self.assertIn("【相邻补充】", body)

    def test_expanded_segment_is_not_marked_as_core(self) -> None:
        body = format_segment_region(
            core_contents=["核心"],
            expanded_before=["扩展"],
            expanded_after=[],
            omitted_before=False,
            omitted_after=False,
        )
        self.assertIn("【相邻补充】\n扩展", body)
        self.assertIn("【核心命中】\n核心", body)

    def test_ellipsis_only_when_content_is_omitted(self) -> None:
        with_ellipsis = format_segment_region(
            core_contents=["核心"],
            expanded_before=[],
            expanded_after=[],
            omitted_before=True,
            omitted_after=False,
        )
        without = format_segment_region(
            core_contents=["核心"],
            expanded_before=[],
            expanded_after=[],
            omitted_before=False,
            omitted_after=False,
        )
        self.assertIn("……（前文省略）", with_ellipsis)
        self.assertNotIn("……", without)

    def test_internal_scores_and_ids_are_not_rendered(self) -> None:
        frag = RetrievedFragment(
            fragment_type="message",
            anchor_role="user",
            parent_message_id=10,
            score=0.99,
            core_unit_ids=[10],
            expanded_unit_ids=[],
            reply_context_message_id=None,
            earlier_user_message_ids=[],
            later_user_message_ids=[],
            omitted_before=False,
            omitted_after=False,
            token_count=10,
            core_message_content="用户原话",
            core_contents=["用户原话"],
        )
        lines = format_fragment_sections(
            bundles=[
                Span(
                    bundle_id="f1",
                    conversation_id=1,
                    anchor_ids=[10],
                    messages=[],
                    fused_score=0.9,
                    fragment=frag,
                )
            ],
            profiles=[],
        )
        text = "\n".join(lines)
        self.assertNotIn("0.99", text)
        self.assertNotIn("message_id", text)
        self.assertNotIn("segment_id", text)


class CollapseTests(unittest.TestCase):
    def test_same_parent_adjacent_core_segments_are_merged(self) -> None:
        anchors = collapse_parent_anchors(
            [
                RetrievalAnchor(
                    unit_type="segment",
                    unit_id=3,
                    parent_message_id=100,
                    role="user",
                    content="s3",
                    score=0.8,
                    source_query="q",
                    segment_id=3,
                    segment_index=2,
                ),
                RetrievalAnchor(
                    unit_type="segment",
                    unit_id=4,
                    parent_message_id=100,
                    role="user",
                    content="s4",
                    score=0.7,
                    source_query="q",
                    segment_id=4,
                    segment_index=3,
                ),
            ]
        )
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].core_unit_ids, [3, 4])


class BudgetTests(unittest.TestCase):
    def test_core_content_survives_token_budget(self) -> None:
        frag = RetrievedFragment(
            fragment_type="segment",
            anchor_role="user",
            parent_message_id=1,
            score=0.9,
            core_unit_ids=[1],
            expanded_unit_ids=[2, 3],
            reply_context_message_id=None,
            earlier_user_message_ids=[],
            later_user_message_ids=[],
            omitted_before=False,
            omitted_after=False,
            token_count=999,
            core_contents=["核心内容必须保留"],
            expanded_contents=["扩展一" * 50, "扩展二" * 50],
        )
        trimmed = trim_fragment_to_budget(frag)
        self.assertIn("核心内容必须保留", trimmed.core_contents[0])
        self.assertLess(trimmed.token_count, frag.token_count)


class IngestTests(unittest.TestCase):
    def test_hidden_nonfinal_assistant_not_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "mem.db")
            store.init()
            ing = TraceIngestor(store, embedder=_FakeEmbedder())
            ing.ingest_message(
                Turn(
                    user_id=1,
                    conversation_id=1,
                    message_id=1,
                    role="assistant",
                    content="内部草稿",
                    position=1,
                    created_at=1,
                    retrievable=False,
                    visible_to_user=False,
                    is_final=False,
                )
            )
            self.assertEqual(store.list_active_with_embeddings(1), [])


class ProfileMaintainerTests(unittest.TestCase):
    def test_profile_maintainer_does_not_use_assistant_memory(self) -> None:
        prompt = build_maintainer_system(target_chars=40, max_operations=3, max_active=20)
        self.assertIn("助手的建议", prompt)
        self.assertIn("除非用户明确表示将长期采用", prompt)


class PipelineHardeningTests(unittest.TestCase):
    def test_assistant_anchor_reaches_formatter(self) -> None:
        frag = RetrievedFragment(
            fragment_type="message",
            anchor_role="assistant",
            parent_message_id=2,
            score=0.8,
            core_unit_ids=[2],
            expanded_unit_ids=[],
            reply_context_message_id=1,
            earlier_user_message_ids=[],
            later_user_message_ids=[],
            omitted_before=False,
            omitted_after=False,
            token_count=20,
            core_message_content="建议先联系导师",
            core_contents=["建议先联系导师"],
            reply_context_content="我想申请推免",
        )
        text = "\n".join(
            format_fragment_sections(
                bundles=[
                    Span(
                        bundle_id="f1",
                        conversation_id=1,
                        anchor_ids=[2],
                        messages=[],
                        fused_score=0.8,
                        fragment=frag,
                    )
                ],
                profiles=[],
            )
        )
        self.assertIn("role=\"assistant\"", text)
        self.assertIn("助手：建议先联系导师", text)
        self.assertIn("我想申请推免", text)
        self.assertNotIn("用户曾说", text)

    def test_segment_anchor_does_not_restore_full_parent(self) -> None:
        frag = RetrievedFragment(
            fragment_type="segment",
            anchor_role="assistant",
            parent_message_id=10,
            score=0.9,
            core_unit_ids=[3],
            expanded_unit_ids=[],
            reply_context_message_id=None,
            earlier_user_message_ids=[],
            later_user_message_ids=[],
            omitted_before=True,
            omitted_after=True,
            token_count=12,
            core_contents=["中间核心片段"],
        )
        text = "\n".join(
            format_fragment_sections(
                bundles=[
                    Span(
                        bundle_id="f1",
                        conversation_id=1,
                        anchor_ids=[3],
                        messages=[],
                        fused_score=0.9,
                        fragment=frag,
                    )
                ],
                profiles=[],
            )
        )
        self.assertIn("type=\"segment\"", text)
        self.assertIn("【核心命中】", text)
        self.assertIn("中间核心片段", text)
        self.assertIn("……（前文省略）", text)
        self.assertIn("……（后文省略）", text)
        self.assertNotIn("完整父消息", text)

    def test_first_and_last_segment_ellipsis_rules(self) -> None:
        first = format_segment_region(
            core_contents=["开头"],
            expanded_before=[],
            expanded_after=[],
            omitted_before=False,
            omitted_after=True,
        )
        last = format_segment_region(
            core_contents=["结尾"],
            expanded_before=[],
            expanded_after=[],
            omitted_before=True,
            omitted_after=False,
        )
        self.assertNotIn("前文省略", first)
        self.assertIn("后文省略", first)
        self.assertIn("前文省略", last)
        self.assertNotIn("后文省略", last)

    def test_excluded_core_drops_fragment(self) -> None:
        from product_app.app.memory.store.fragments import FragmentBuilder

        store = MagicMock()
        builder = FragmentBuilder(store, embedder=_FakeEmbedder())
        hits = [
            RankedHit(
                chunk_id=1,
                user_id=1,
                conversation_id=1,
                message_id=99,
                role="user",
                position=1,
                content="live",
                created_at=1,
                unit_type="message",
                unit_id=99,
                parent_message_id=99,
                rerank_score=0.9,
            )
        ]
        frags = builder.build(
            user_id=1,
            hits=hits,
            query="q",
            exclude_message_ids={99},
        )
        self.assertEqual(frags, [])

    def test_ce_score_preferred_over_fused(self) -> None:
        from product_app.app.memory.store.fragments import hit_to_anchor

        hit = RankedHit(
            chunk_id=1,
            user_id=1,
            conversation_id=1,
            message_id=1,
            role="user",
            position=1,
            content="x",
            created_at=1,
            fused_score=0.2,
            rerank_score=0.95,
        )
        anchor = hit_to_anchor(hit)
        self.assertAlmostEqual(anchor.score, 0.95)

    def test_retrieval_exception_is_visible_in_test_mode(self) -> None:
        from dataclasses import replace

        from product_app.app.memory.retrieval.pipeline import RetrievalPipeline

        class BoomRouter:
            def plan(self, query: str):
                raise RuntimeError("planner boom")

            def set_llm(self, llm):
                return None

        pipe = RetrievalPipeline(store=MagicMock(), profile=MagicMock())
        pipe._router = BoomRouter()  # type: ignore[assignment]
        with patch(
            "product_app.app.memory.retrieval.pipeline.mem_cfg",
            replace(mem_cfg, raise_retrieval_errors=True),
        ):
            with self.assertRaises(RuntimeError):
                pipe.run(user_id=1, query="q")
        with patch(
            "product_app.app.memory.retrieval.pipeline.mem_cfg",
            replace(mem_cfg, raise_retrieval_errors=False),
        ):
            windows, trace = pipe.run(user_id=1, query="q")
        self.assertEqual(windows, [])
        self.assertTrue(trace.fallback)
        self.assertEqual(trace.extra.get("error_type"), "RuntimeError")


if __name__ == "__main__":
    unittest.main()
