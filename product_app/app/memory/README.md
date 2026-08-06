# SoulHarbor long-term memory

Modules under this package implement **ER** (Experience Rebuild):

- `store/` — Trace archive: save raw turns as Blocks, search, Stitch spans, select evidence
- `profile/` — unified Maintainer (add/update/delete); code owns user_id, sources, ≤20 / ≤640 tokens
- `retrieval/` — episodic recall only (profiles not retrieved by query)
- `context/` — `<user_profile>` + `<memory>` with separate token budgets

Entry points: `MemoryService` (chat) and `MemoryEngine` (ingest / recall).

Terminology: **Trace / Turn / Block / Anchor / Stitch / Span** (Profile unchanged).

Default read path:

```text
Query Router
→ Dense + BM25（仅 user）→ RRF
→ Anchor CE
   · Direct：原始 query CE Top-10（message_id 去重）
   · Split：每个子问题 CE Top-3 并集；不足 10 时按原始 query CE 补齐（Planner ≤3 子问题）
→ Collapse same-message chunk anchors
→ Adaptive Stitch（probe = hit chunk；仅余弦 ≥ 0.40）
→ Merge overlapping windows（连通分量；硬保留锚点）
→ CE Top-k（≤ MEMORY_WINDOW_TOP_K，默认 10）
→ Pack by CE into token budget（超限 continue）
→ Sort by record time for display（含当前日期）
→ Inject
```

`MEMORY_ANCHOR_CE_TOP_K` / `MEMORY_ANCHOR_CE_DIRECT_K` / `MEMORY_WINDOW_TOP_K` 是候选上限，不是注入保证。实际提示词长度由 `MEMORY_CONTEXT_TOKEN_BUDGET` 控制。

正式评测：`evaluate/memory/runs/qa_er_all50/`（all_50：QA 90.8% / stale 6.9% / F1 0.602）。

See `项目文档/03_详尽技术文档.md` §7–§8.
