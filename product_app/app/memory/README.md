# SoulHarbor long-term memory

Modules under this package implement **AER** (adaptive experience reconstruction):

- `store/` — Trace archive: save raw turns as Blocks, search, Stitch spans, link sessions, select evidence
- `profile/` — user-confirmed preferences (consent)
- `retrieval/` — end-to-end recall pipeline
- `context/` — format the `<memory>` block for the chat model

Entry points: `MemoryService` (chat) and `MemoryEngine` (ingest / recall).

Terminology: **Trace / Turn / Block / Focus / Stitch / Span** (Profile unchanged).

See `项目文档/05_记忆系统_AER自适应经历重建_最终方案.md`.
