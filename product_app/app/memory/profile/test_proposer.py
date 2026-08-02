"""Unit tests for LLM profile proposer + consent (no network)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from product_app.app.memory.profile.proposer import (
    _parse,
    evidence_supported,
    roughly_same,
)
from product_app.app.memory.profile.service import ProfileService
from product_app.app.memory.store.repository import TraceStore


class StubLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def generate_structured(self, messages, *, max_new_tokens=256, system_text=""):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


class ProfileProposerTests(unittest.TestCase):
    def test_parse_max_one(self):
        raw = (
            '{"proposals":['
            '{"content":"希望先被倾听","evidence":"希望先被倾听"},'
            '{"content":"第二条","evidence":"y"}'
            "]}"
        )
        items = _parse(raw, max_items=1)
        self.assertEqual(len(items), 1)

    def test_evidence_supported(self):
        prop = {
            "content": "希望先被倾听",
            "evidence": "希望你先听我说",
        }
        self.assertTrue(evidence_supported(prop, ["今天好累，希望你先听我说"]))
        self.assertFalse(evidence_supported(prop, ["今天食堂什么菜"]))

    def test_roughly_same(self):
        self.assertTrue(roughly_same("希望先被倾听", "希望先被倾听一下"))
        self.assertFalse(roughly_same("希望先被倾听", "喜欢跑步"))

    def test_llm_propose_to_pending_then_confirm(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            llm = StubLLM(
                {
                    "proposals": [
                        {
                            "content": "希望情绪强烈时先被倾听",
                            "evidence": "希望情绪强烈时先被倾听",
                        },
                        {
                            "content": "诊断为抑郁症患者",
                            "evidence": "抑郁症",
                        },
                    ]
                }
            )
            with patch("product_app.app.memory.config.mem_cfg") as cfg:
                cfg.profile_llm_propose_max = 1
                cfg.profile_llm_skip_if_pending = True
                cfg.profile_llm_trigger_messages = 5
                cfg.profile_llm_trigger_age_sec = 300
                out = svc.maybe_llm_propose(
                    user_id=1,
                    llm=llm,
                    recent_turns=[
                        {"role": "user", "content": "我希望情绪强烈时先被倾听"},
                        {"role": "assistant", "content": "好的，我记下了这个偏好候选。"},
                    ],
                    source_message_id=2,
                    force=True,
                )
            self.assertIsNotNone(out)
            self.assertEqual(svc._db.count_pending(1), 1)
            self.assertEqual(len(svc.list_active(1)), 0)
            svc.maybe_handle_user_command(
                user_id=1, user_text="可以", source_message_id=3
            )
            actives = svc.list_active(1)
            self.assertEqual(len(actives), 1)
            self.assertIn("倾听", actives[0].content)

    def test_empty_when_model_returns_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            llm = StubLLM({"proposals": []})
            with patch("product_app.app.memory.config.mem_cfg") as cfg:
                cfg.profile_llm_propose_max = 1
                cfg.profile_llm_skip_if_pending = True
                cfg.profile_llm_trigger_messages = 5
                cfg.profile_llm_trigger_age_sec = 300
                out = svc.maybe_llm_propose(
                    user_id=1,
                    llm=llm,
                    recent_turns=[
                        {"role": "user", "content": "今天食堂什么菜"},
                        {"role": "assistant", "content": "有红烧肉。"},
                    ],
                    source_message_id=2,
                    force=True,
                )
            self.assertIsNone(out)
            self.assertEqual(llm.calls, 1)
            self.assertEqual(svc._db.count_pending(1), 0)

    def test_skip_if_pending_already(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            svc.propose(
                user_id=1,
                content="已有待确认偏好希望少打断",
                source_message_ids=[1],
                source_messages=["希望你少打断我"],
            )
            self.assertEqual(svc._db.count_pending(1), 1)
            llm = StubLLM(
                {
                    "proposals": [
                        {
                            "content": "希望先被倾听",
                            "evidence": "希望先被倾听",
                        }
                    ]
                }
            )
            with patch("product_app.app.memory.config.mem_cfg") as cfg:
                cfg.profile_llm_propose_max = 1
                cfg.profile_llm_skip_if_pending = True
                cfg.profile_llm_trigger_messages = 5
                cfg.profile_llm_trigger_age_sec = 300
                out = svc.maybe_llm_propose(
                    user_id=1,
                    llm=llm,
                    recent_turns=[
                        {"role": "user", "content": "我希望先被倾听"},
                        {"role": "assistant", "content": "嗯。"},
                    ],
                    source_message_id=3,
                    force=True,
                )
            self.assertIsNone(out)
            self.assertEqual(llm.calls, 0)
            self.assertEqual(svc._db.count_pending(1), 1)


    def test_batch_trigger_after_n_messages(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            TraceStore(db).init()
            svc = ProfileService(db)
            llm = StubLLM(
                {
                    "proposals": [
                        {
                            "content": "希望先被倾听",
                            "evidence": "希望先被倾听",
                        }
                    ]
                }
            )
            with patch("product_app.app.memory.config.mem_cfg") as cfg:
                cfg.profile_llm_propose_max = 1
                cfg.profile_llm_skip_if_pending = True
                cfg.profile_llm_trigger_messages = 3
                cfg.profile_llm_trigger_age_sec = 99999
                for _ in range(2):
                    svc.note_message_for_llm_propose(1)
                out = svc.maybe_llm_propose(
                    user_id=1,
                    llm=llm,
                    recent_turns=[
                        {"role": "user", "content": "我希望先被倾听"},
                        {"role": "assistant", "content": "好。"},
                    ],
                    source_message_id=10,
                )
                self.assertIsNone(out)
                self.assertEqual(llm.calls, 0)
                svc.note_message_for_llm_propose(1)
                out = svc.maybe_llm_propose(
                    user_id=1,
                    llm=llm,
                    recent_turns=[
                        {"role": "user", "content": "我希望先被倾听"},
                        {"role": "assistant", "content": "好。"},
                    ],
                    source_message_id=11,
                )
                self.assertIsNotNone(out)
                self.assertEqual(llm.calls, 1)
                self.assertEqual(svc._db.count_pending(1), 1)


if __name__ == "__main__":
    unittest.main()
