"""Unit tests for profile maintainer (no evidence from LLM; code owns safety)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from product_app.app.memory.context.builder import build_memory_block
from product_app.app.memory.context.profile_formatter import render_user_profile
from product_app.app.memory.models import ProfileItem
from product_app.app.memory.profile.maintainer import parse_profile_decision
from product_app.app.memory.profile.operations import (
    ProfileOperation,
    filter_profile_operations,
    normalize_profile_content,
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


def _seed_user_msg(store: TraceStore, *, user_id: int, message_id: int, content: str) -> None:
    store.upsert_chunks(
        user_id=user_id,
        conversation_id=1,
        message_id=message_id,
        role="user",
        position=message_id,
        created_at=1_700_000_000 + message_id,
        chunks=[content],
    )


class ProfileMaintainTests(unittest.TestCase):
    def test_parse_decision(self):
        d = parse_profile_decision(
            {
                "operations": [
                    {"op": "add", "target_id": "", "content": "用户偏好简洁沟通"}
                ]
            }
        )
        self.assertEqual(len(d.operations), 1)
        self.assertEqual(d.operations[0].op, "add")

    def test_normalize_adds_period(self):
        self.assertEqual(normalize_profile_content("用户喜欢跑步"), "用户喜欢跑步。")

    def test_reject_too_long_content(self):
        long = "字" * 80
        ops = filter_profile_operations(
            [ProfileOperation(op="add", target_id="", content=long)],
            active_profiles={},
            token_counter=None,
            max_operations=3,
            max_chars=64,
            max_tokens=48,
        )
        self.assertEqual(ops, [])

    def test_maintain_add_update_delete(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            store = TraceStore(db)
            store.init()
            svc = ProfileService(db)
            _seed_user_msg(store, user_id=1, message_id=2, content="我叫小王，喜欢直接沟通")

            llm = StubLLM(
                {
                    "operations": [
                        {
                            "op": "add",
                            "target_id": "",
                            "content": "用户希望被称为小王",
                        },
                        {
                            "op": "add",
                            "target_id": "",
                            "content": "用户偏好直接简洁的沟通方式",
                        },
                    ]
                }
            )
            changes = svc.maintain_from_user_turn(
                user_id=1,
                current_user_message_id=2,
                recent_turns=[
                    {
                        "message_id": 2,
                        "role": "user",
                        "content": "我叫小王，喜欢直接沟通",
                    }
                ],
                llm=llm,
            )
            self.assertEqual(len(changes.added_ids), 2)
            actives = svc.list_all_for_context(user_id=1)
            self.assertEqual(len(actives), 2)

            pid = actives[0].id
            _seed_user_msg(store, user_id=1, message_id=3, content="其实叫我小夏就好")
            llm2 = StubLLM(
                {
                    "operations": [
                        {
                            "op": "update",
                            "target_id": pid,
                            "content": "用户希望被称为小夏",
                        }
                    ]
                }
            )
            changes2 = svc.maintain_from_user_turn(
                user_id=1,
                current_user_message_id=3,
                recent_turns=[
                    {
                        "message_id": 2,
                        "role": "user",
                        "content": "我叫小王，喜欢直接沟通",
                    },
                    {
                        "message_id": 3,
                        "role": "user",
                        "content": "其实叫我小夏就好",
                    },
                ],
                llm=llm2,
            )
            self.assertEqual(changes2.updated_ids, [pid])
            updated = {p.id: p for p in svc.list_all_for_context(user_id=1)}
            self.assertIn("小夏", updated[pid].content)

    def test_foreign_profile_id_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            store = TraceStore(db)
            store.init()
            svc = ProfileService(db)
            _seed_user_msg(store, user_id=1, message_id=1, content="我叫小王")
            svc.maintain_from_user_turn(
                user_id=1,
                current_user_message_id=1,
                recent_turns=[
                    {"message_id": 1, "role": "user", "content": "我叫小王"}
                ],
                llm=StubLLM(
                    {
                        "operations": [
                            {
                                "op": "add",
                                "target_id": "",
                                "content": "用户希望被称为小王",
                            }
                        ]
                    }
                ),
            )
            real_id = svc.list_all_for_context(user_id=1)[0].id
            _seed_user_msg(store, user_id=2, message_id=10, content="改成小李")
            changes = svc.maintain_from_user_turn(
                user_id=2,
                current_user_message_id=10,
                recent_turns=[
                    {"message_id": 10, "role": "user", "content": "改成小李"}
                ],
                llm=StubLLM(
                    {
                        "operations": [
                            {
                                "op": "update",
                                "target_id": real_id,
                                "content": "用户希望被称为小李",
                            }
                        ]
                    }
                ),
            )
            self.assertEqual(changes.updated_ids, [])
            self.assertIn("小王", svc.list_all_for_context(user_id=1)[0].content)

    def test_enforce_count_and_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            store = TraceStore(db)
            store.init()
            svc = ProfileService(db)
            for i in range(20):
                mid = 100 + i
                text = f"旧画像内容编号{i}"
                _seed_user_msg(store, user_id=1, message_id=mid, content=text)
                svc._db.apply_profile_operations(
                    user_id=1,
                    current_user_message_id=mid,
                    operations=[
                        ProfileOperation(op="add", target_id="", content=text)
                    ],
                    max_active=20,
                    max_block_tokens=640,
                    max_operations=1,
                    max_chars=64,
                    max_tokens=48,
                )
            self.assertEqual(len(svc.list_all_for_context(user_id=1)), 20)
            mid = 999
            text = "新的重要画像条目"
            _seed_user_msg(store, user_id=1, message_id=mid, content=text)
            changes = svc._db.apply_profile_operations(
                user_id=1,
                current_user_message_id=mid,
                operations=[ProfileOperation(op="add", target_id="", content=text)],
                max_active=20,
                max_block_tokens=640,
                max_operations=1,
                max_chars=64,
                max_tokens=48,
            )
            self.assertEqual(len(changes.added_ids), 1)
            self.assertEqual(len(changes.pruned_ids), 1)
            actives = svc.list_all_for_context(user_id=1)
            self.assertEqual(len(actives), 20)
            self.assertTrue(any("新的重要画像" in p.content for p in actives))

    def test_render_user_profile(self):
        items = [
            ProfileItem("a", 1, "用户喜欢跑步", "maintained", [], "active", 1, 1),
            ProfileItem("b", 1, "用户就读计算机。", "maintained", [], "active", 2, 2),
        ]
        block = render_user_profile(items)
        self.assertTrue(block.startswith("<user_profile>"))
        self.assertIn("用户喜欢跑步。", block)
        self.assertIn("用户就读计算机。", block)

    def test_builder_separates_profile_budget(self):
        profiles = [
            ProfileItem("a", 1, "用户偏好简洁沟通", "maintained", [], "active", 1, 1)
        ]
        block, packed = build_memory_block(
            bundles=[],
            profiles=profiles,
            token_budget=1600,
            query="x",
        )
        self.assertIn("<user_profile>", block)
        self.assertNotIn("<memory>", block)
        self.assertEqual(packed, 0)

    def test_empty_operations(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "p.db"
            store = TraceStore(db)
            store.init()
            _seed_user_msg(store, user_id=1, message_id=1, content="今天天气不错")
            svc = ProfileService(db)
            llm = StubLLM({"operations": []})
            changes = svc.maintain_from_user_turn(
                user_id=1,
                current_user_message_id=1,
                recent_turns=[
                    {"message_id": 1, "role": "user", "content": "今天天气不错"}
                ],
                llm=llm,
            )
            self.assertEqual(llm.calls, 1)
            self.assertIsNone(changes.summary())
            self.assertEqual(len(svc.list_all_for_context(user_id=1)), 0)


if __name__ == "__main__":
    unittest.main()
