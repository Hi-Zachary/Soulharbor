# SoulHarbor long-term memory

Modules under this package implement **AER** (adaptive experience reconstruction):

- `store/` — save raw turns, search, expand windows, link sessions, select evidence
- `profile/` — user-confirmed preferences (consent)
- `retrieval/` — end-to-end recall pipeline
- `context/` — format the `<memory>` block for the chat model

Entry points: `MemoryService` (chat) and `MemoryEngine` (ingest / recall).

See `项目文档/05_记忆系统_AER自适应经历重建_最终方案.md`.
