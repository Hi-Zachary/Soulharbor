# SoulHarbor long-term memory

Modules under this package implement **ER** (Experience Rebuild):

- `store/` — Trace archive: save raw turns as Blocks, search, Stitch spans, link sessions, select evidence
- `profile/` — strict LLM long-term user facts (allowlisted; no keyword gates)
- `retrieval/` — end-to-end recall pipeline
- `context/` — format the `<memory>` block for the chat model

Entry points: `MemoryService` (chat) and `MemoryEngine` (ingest / recall).

Terminology: **Trace / Turn / Block / Anchor / Stitch / Span** (Profile unchanged).

Default read path: Planner → Hybrid → Adaptive Expansion → Merge → Decay-aware MMR → Chronological Injection → Usage Reinforcement.

See `项目文档/05_记忆系统_ER经历重建_最终方案.md`.
