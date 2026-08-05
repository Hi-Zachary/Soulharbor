# SoulHarbor long-term memory

Modules under this package implement **ER** (Experience Rebuild):

- `store/` — Trace archive: save raw turns as Blocks, search, Stitch spans, select evidence
- `profile/` — strict LLM long-term user facts (allowlisted; no keyword gates)
- `retrieval/` — end-to-end recall pipeline
- `context/` — format the `<memory>` block for the chat model

Entry points: `MemoryService` (chat) and `MemoryEngine` (ingest / recall).

Terminology: **Trace / Turn / Block / Anchor / Stitch / Span** (Profile unchanged).

Default read path:

```text
Query Router
→ Dense + BM25 → RRF
→ Multi-query Coverage CrossEncoder (≤ MEMORY_ANCHOR_CE_TOP_K, default 12)
→ Collapse same-message chunk anchors
→ Adaptive Stitch (probe = hit chunk; inject = full message)
→ Merge overlapping windows
→ CE Top-k (≤ MEMORY_WINDOW_TOP_K, default 12)
→ Pack by CE into token budget (skip oversized; continue with later windows)
→ Sort by record time for display
→ Inject
```

`MEMORY_ANCHOR_CE_TOP_K` / `MEMORY_WINDOW_TOP_K` are candidate caps, not injection
guarantees. Actual prompt size is controlled by `MEMORY_CONTEXT_TOKEN_BUDGET`.

See `项目文档/03_详尽技术文档.md` §7–§8.
