"""Unit tests for strict profile LLM extract (no network, no keyword gates)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from product_app.app.memory.profile.extractor import (
    _parse_commands,
    evidence_supported,
)
from product_app.app.memory.profile.schema import normalize_fact
from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.store.repository import TraceStore


class StubLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def generate_structured(self, messages, *, max_new_tokens=256, system_text=""):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


class ProfileExtractTests(unittest.TestCase):
    def test_parse_filters_non_allowlisted(self):
        raw = json.dumps(
            {
                "commands": [
                    {
                        "command": "add",
                        "tag": "identity",
                        "feature": "name",
                        "value": "小王",
                        "evidence": "我叫小王",
                    },
                    {
                        "command": "add",
                        "tag": "diagnosis",
                        "feature": "label",
                        "value": "抑郁症",
                        "evidence": "抑郁症",
                    },
                ]
            },
            ensure_ascii=False,
        )
        cmds = _parse_commands(raw, max_items=5)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0]["feature"], "name")

    def test_evidence_supported(self):
        cmd = {
            "command": "add",
            "tag": "identity",
            "feature": "name",
            "value": "小王",
            "evidence": "我叫小王",
        }
        self.assertTrue(evidence_supported(cmd, ["你好，我叫小王"]))
        self.assertFalse(evidence_supported(cmd, ["今天食堂什么菜"]))

    def test_normalize_fact_reject_unknown(self):
        self.assertIsNone(
            normalize_fact(tag="psychology", feature="trait", value="高焦虑")
        )
        self.assertIsNotNone(
            normalize_fact(tag="identity", feature="name", value="小王")
        )

    def test_llm_extract_writes_active_directly(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            llm = StubLLM(
                {
                    "commands": [
                        {
                            "command": "add",
                            "tag": "identity",
                            "feature": "name",
                            "value": "小王",
                            "evidence": "我叫小王",
                        },
                        {
                            "command": "add",
                            "tag": "identity",
                            "feature": "mood",
                            "value": "今天很难过",
                            "evidence": "很难过",
                        },
                    ]
                }
            )
            out = svc.maybe_llm_extract(
                user_id=1,
                llm=llm,
                recent_turns=[
                    {"role": "user", "content": "我叫小王，大三"},
                    {"role": "assistant", "content": "你好小王。"},
                ],
                source_message_id=2,
                force=True,
                max_items=3,
            )
            self.assertIsNotNone(out)
            actives = svc.list_active(1)
            self.assertEqual(len(actives), 1)
            self.assertIn("小王", actives[0].content)
            self.assertEqual(actives[0].origin, "extracted")

    def test_empty_when_model_returns_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            llm = StubLLM({"commands": []})
            out = svc.maybe_llm_extract(
                user_id=1,
                llm=llm,
                recent_turns=[
                    {"role": "user", "content": "今天食堂什么菜"},
                    {"role": "assistant", "content": "有红烧肉。"},
                ],
                source_message_id=2,
                force=True,
                max_items=3,
            )
            self.assertIsNone(out)
            self.assertEqual(llm.calls, 1)
            self.assertEqual(len(svc.list_active(1)), 0)

    def test_llm_command_remember(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            out = svc.maybe_handle_user_command(
                user_id=1,
                user_text="请记住我叫小李",
                source_message_id=1,
                parsed={
                    "action": "remember",
                    "tag": "identity",
                    "feature": "name",
                    "value": "小李",
                    "query": "",
                },
            )
            self.assertTrue(out and out.startswith("profile_saved:"))
            self.assertEqual(len(svc.list_active(1)), 1)

    def test_batch_trigger_after_n_messages(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            llm = StubLLM(
                {
                    "commands": [
                        {
                            "command": "add",
                            "tag": "education",
                            "feature": "major",
                            "value": "计算机",
                            "evidence": "我是计算机专业",
                        }
                    ]
                }
            )
            for _ in range(2):
                svc.note_message_for_llm_propose(1)
            out = svc.maybe_llm_extract(
                user_id=1,
                llm=llm,
                recent_turns=[
                    {"role": "user", "content": "我是计算机专业的"},
                    {"role": "assistant", "content": "好。"},
                ],
                source_message_id=10,
                trigger_messages=3,
                trigger_age_sec=99999,
                max_items=3,
            )
            self.assertIsNone(out)
            self.assertEqual(llm.calls, 0)
            svc.note_message_for_llm_propose(1)
            out = svc.maybe_llm_extract(
                user_id=1,
                llm=llm,
                recent_turns=[
                    {"role": "user", "content": "我是计算机专业的"},
                    {"role": "assistant", "content": "好。"},
                ],
                source_message_id=11,
                trigger_messages=3,
                trigger_age_sec=99999,
                max_items=3,
            )
            self.assertIsNotNone(out)
            self.assertEqual(llm.calls, 1)
            self.assertEqual(len(svc.list_active(1)), 1)


if __name__ == "__main__":
    unittest.main()
