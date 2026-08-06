"""Load and ingest SoulHarbor-MH-LongMemEval-30 instances."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_LME_DATE_FMT = "%Y/%m/%d (%a) %H:%M"
_SOURCE_SESSION_RE = re.compile(r"^(.*-s\d+)-[ua]\d+$")


def parse_lme_date(value: str) -> int:
    return int(datetime.strptime(str(value).strip(), _LME_DATE_FMT).timestamp())


def is_lme_instance(row: Dict[str, Any]) -> bool:
    return bool(row.get("haystack_sessions") and row.get("question_id"))


def load_lme_instances(
    path: Path,
    *,
    limit: int = 0,
    categories: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = [rows]
    if categories:
        want = set(categories)
        rows = [r for r in rows if r.get("category") in want]
    if capabilities:
        want = set(capabilities)
        rows = [r for r in rows if r.get("capability") in want]
    if limit > 0:
        return rows[:limit]
    return rows


def source_message_to_session(source_message_id: str) -> str:
    m = _SOURCE_SESSION_RE.match(str(source_message_id or "").strip())
    if m:
        return m.group(1)
    return str(source_message_id or "")


class LMETraceEngine:
    """Fresh ER backend per question instance; real session timestamps."""

    def __init__(self, *, work_dir: Path, llm: Any) -> None:
        from product_app.app.memory.engine import MemoryEngine

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = work_dir / "trace.db"
        self._llm = llm
        self._engine = MemoryEngine(self._db_path, llm=llm)
        self._user_id = 1
        self._next_message_id = 1
        self._sid_to_cid: Dict[str, int] = {}
        self._source_to_db: Dict[str, int] = {}
        self._db_to_source: Dict[int, str] = {}
        self._last_retrieved_db_ids: Set[int] = set()

    @property
    def source_to_db(self) -> Dict[str, int]:
        return dict(self._source_to_db)

    @property
    def db_to_source(self) -> Dict[int, str]:
        return dict(self._db_to_source)

    def ingest_instance(self, item: Dict[str, Any]) -> None:
        session_ids = list(item.get("haystack_session_ids") or [])
        session_dates = list(item.get("haystack_dates") or [])
        sessions = list(item.get("haystack_sessions") or [])
        if len(session_ids) != len(sessions):
            raise ValueError(
                f"session id/count mismatch for {item.get('question_id')}: "
                f"{len(session_ids)} ids vs {len(sessions)} sessions"
            )
        for idx, (sid, session) in enumerate(zip(session_ids, sessions)):
            if sid not in self._sid_to_cid:
                self._sid_to_cid[sid] = self._next_conversation_id()
            cid = self._sid_to_cid[sid]
            created = parse_lme_date(session_dates[idx] if idx < len(session_dates) else session_dates[-1])
            current_turn_id = 0
            for pos, turn in enumerate(session or [], start=1):
                role = str(turn.get("role") or "")
                content = str(turn.get("content") or "")
                if role not in ("user", "assistant") or not content.strip():
                    continue
                source_mid = str(turn.get("message_id") or "")
                db_mid = self._next_message_id
                self._next_message_id += 1
                if role == "user" or current_turn_id <= 0:
                    current_turn_id = db_mid
                if source_mid:
                    self._source_to_db[source_mid] = db_mid
                    self._db_to_source[db_mid] = source_mid
                self._engine.ingest_message(
                    user_id=self._user_id,
                    conversation_id=cid,
                    message_id=db_mid,
                    turn_id=current_turn_id,
                    role=role,
                    content=content,
                    position=pos,
                    created_at=created + pos,
                )

    def _next_conversation_id(self) -> int:
        cid = len(self._sid_to_cid) + 1
        return cid

    def retrieve(self, query: str) -> str:
        block, _ = self.retrieve_with_details(query)
        return block

    def retrieve_with_details(self, query: str) -> Tuple[str, Dict[str, Any]]:
        from product_app.app.memory.config import mem_cfg
        from product_app.app.memory.context.builder import build_memory_block
        from product_app.app.memory.models import RetrievalTrace
        from product_app.app.memory.retrieval.sufficiency import is_enough
        from product_app.app.memory.token_utils import fallback_token_count

        query = (query or "").strip()
        if not query:
            self._last_retrieved_db_ids = set()
            return "", {"retrieved_db_message_ids": [], "retrieved_source_message_ids": [], "retrieved_session_ids": []}

        try:
            windows, trace = self._engine._pipeline.run(
                user_id=self._user_id,
                query=query,
                exclude_message_ids=set(),
            )
            counter = getattr(self._llm, "count_tokens", None) if self._llm else None
            profiles = (
                self._engine._profile.list_all_for_context(
                    user_id=self._user_id, token_counter=counter
                )
                if mem_cfg.profile_enabled
                else []
            )
            trace.profile_hits = len(profiles)
            trace.enough = is_enough(windows, profiles)
            budget = int(mem_cfg.context_token_budget)
            block, packed_count = build_memory_block(
                bundles=windows,
                profiles=profiles,
                token_budget=budget,
                token_counter=counter,
                query=query,
            )
            trace.selected_bundles = packed_count
            trace.extra["packed_window_count"] = packed_count
            trace.extra["topk_before_budget"] = len(windows)
            if counter and block:
                trace.memory_tokens = int(counter(block))
            else:
                trace.memory_tokens = fallback_token_count(block) if block else 0
            self._engine._last_trace = trace

            anchor_db_ids: Set[int] = set()
            included_db_ids: Set[int] = set()
            anchor_turn_ids: Set[int] = set()
            for window in windows[: max(0, int(packed_count))]:
                anchor_db_ids.update(int(x) for x in window.anchor_ids)
                anchor_turn_ids.add(int(window.anchor_turn_id))
                for turn in window.messages:
                    included_db_ids.add(int(turn.message_id))
                    if turn.is_anchor:
                        anchor_db_ids.add(int(turn.message_id))
            self._last_retrieved_db_ids = included_db_ids
            anchor_source_ids = [
                self._db_to_source[mid]
                for mid in sorted(anchor_db_ids)
                if mid in self._db_to_source
            ]
            included_source_ids = [
                self._db_to_source[mid]
                for mid in sorted(included_db_ids)
                if mid in self._db_to_source
            ]
            session_ids = sorted({source_message_to_session(mid) for mid in included_source_ids if mid})
            details = {
                "retrieved_db_message_ids": sorted(included_db_ids),
                "retrieved_source_message_ids": included_source_ids,
                "retrieved_session_ids": session_ids,
                "retrieved_anchor_db_message_ids": sorted(anchor_db_ids),
                "retrieved_anchor_source_message_ids": anchor_source_ids,
                "retrieved_included_db_message_ids": sorted(included_db_ids),
                "retrieved_included_source_message_ids": included_source_ids,
                "retrieved_anchor_turn_ids": sorted(anchor_turn_ids),
                "retrieval_trace": trace.to_log_dict(),
                "memory_block": block,
            }
            return block, details
        except Exception:
            self._engine._last_trace = RetrievalTrace(fallback=True)
            self._last_retrieved_db_ids = set()
            return "", {
                "retrieved_db_message_ids": [],
                "retrieved_source_message_ids": [],
                "retrieved_session_ids": [],
                "retrieval_trace": self._engine._last_trace.to_log_dict(),
                "memory_block": "",
            }

    def list_active_profiles(self) -> List[str]:
        info = self._engine.inspect(self._user_id)
        return [
            str(p.get("content") or "").strip()
            for p in (info.get("support_preferences") or [])
            if str(p.get("content") or "").strip()
        ]

    def store_snapshot(self) -> Dict[str, Any]:
        info = self._engine.inspect(self._user_id)
        stats = self._engine._store.index_stats(self._user_id)
        return {
            "backend": "er",
            "trace_blocks": info.get("trace_blocks"),
            "support_preferences": info.get("support_preferences"),
            "index": stats,
            "active_profiles": self.list_active_profiles(),
            "last_trace": (self._engine.last_trace.to_log_dict() if self._engine.last_trace else None),
        }
