# SoulHarbor 记忆系统：自适应经历重建（AER）详解

> 与当前源码一致：`product_app/app/memory/`，`MEMORY_BACKEND=aer`。  
> **产品故事与动机**：`00_SoulHarbor故事与终稿叙事.md`。  
> 对比实验与数字见 `06_记忆系统_对比实验设计.md`。

---

## 0. 一句话与设计立场

**SoulHarbor 的长期记忆：Episode 不在写入时"理解"用户；Profile 仅经 consent 落库；查询时再把原文编年重建成可用的经历证据。**

心理陪伴场景里，错误归纳的代价远高于漏召回：一次考试焦虑、一次冷战、一句气话，如果被系统写成“稳定特质”，会在后续对话里反复误导。因此：

1. **Episode 写路径零 LLM**：消息按规则分块、BGE-M3 向量化后原样入库（`store/`）；不抽事实、不合并、不概括。
2. **读路径做检索智能**：混合召回 → 锚点扩窗（AER）→ 跨会话串链 → 覆盖度选择 → 渲染 `<memory>`；可选 LLM 仅用于查询规划（及可选重排开关下的特征重排仍由代码完成）。
3. **唯一允许落库的“理解”**：Profile（关怀画像）旁路。默认在助理回合后可由中文 LLM（`profile/proposer.py`）提议 **最多 1** 条候选进 **pending**（须对上用户原文证据；已有 pending 则不再抽）；用户确认后才写 active。显式「记住…」仍为正则快路径；政策层硬挡诊断词、人格标签、瞬时情绪。

在 30 条长程咨询弧、固定 reader（MiniMax-M3）同协议对照 Mem0 时：**QA ≈91–93% vs 64%，内容 F1 ≈0.79–0.89 vs 0.48**。

**这个立场是被证据逼出来的**：早期写时抽取 / 分层调和无法从结构上消除"写错标签"；后来转向存原文，再删掉写时调和器，补上自适应扩窗、覆盖度选择与跨会话经历链，才形成现在的 AER。中间态快照在 `archive/`，答辩可对照；线上文档只描述 AER 终态。

---

## 1. 术语表（源码命名）

| 概念 | 代码名 | 含义 |
|---|---|---|
| AER | Adaptive Experience Reconstruction | 查询时把命中锚点扩成一段经历、再选证注入 |
| Store | `memory/store/` | 原文库：分块、向量、检索、扩窗、串链、覆盖选择 |
| Seed（锚点） | `is_seed` / `seed_ids` / `seed_top_k` | hybrid 召回命中的核心消息，扩窗以此为中心 |
| EpisodeWindow | `EpisodeWindow` | 一段重建后的经历窗口（若干 `WindowTurn`） |
| WindowTurn | `WindowTurn` | 窗口内的一条消息证据 |
| Profile | `memory/profile/` | consent 画像旁路；条目类型 `ProfileItem` |
| Memory block | `<memory>` | 注入到对话 system 侧的长期记忆块标签 |
| Engine | `MemoryEngine` | 写/读/管理命令的统一编排 |
| Facade | `MemoryService` | 对接聊天：滑动窗 + 会话摘要 + AER |

---

## 2. 目录结构

```
product_app/app/memory/
├── service.py              # 聊天门面：build_chat_context / run_post_turn / 摘要
├── engine.py               # MemoryEngine
├── config.py               # mem_cfg（MemorySettings）
├── models.py               # EpisodeMessage / RankedHit / EpisodeWindow / ProfileItem / RetrievalTrace
├── embeddings.py           # BGE-M3 单例
├── inject.py / extract.py  # 会话摘要块与摘要 prompt（短期工作记忆，非 AER）
├── session_context.py      # 消息切片给分类器 / 聊天
├── store/              # 原文编年 + AER 核心
│   ├── ingest.py           # EpisodeIngestor：分块写入 + 向量
│   ├── repository.py       # EpisodeStore：SQLite
│   ├── chunker.py          # 确定性中文分块
│   ├── semantic.py     # SemanticSearcher（FAISS / cosine 回退）
│   ├── ann_index.py        # 按用户 FAISS 缓存
│   ├── lexical_search.py   # LexicalSearcher（字级 BM25）
│   ├── fusion.py           # RRF
│   ├── expand.py      # WindowExpander：自适应 / 固定扩窗 ★
│   ├── merge.py       # 同会话重叠窗合并
│   ├── rerank.py        # 特征重排（可选 LLM）
│   ├── link.py       # 跨会话时序证据链
│   ├── select.py         # 覆盖度贪心选择 ★
│   └── text_sim.py         # cosine / Jaccard / 实体粗抽
├── retrieval/
│   ├── pipeline.py         # 端到端编排
│   ├── router.py           # direct / split
│   ├── direct.py           # 单查询 hybrid
│   ├── split_query.py      # 多子查询 RRF 合并
│   └── sufficiency.py      # 是否“够用”（观测用）
├── profile/                   # Profile
│   ├── service.py / repository.py / detector.py / policy.py
├── context/
│   ├── builder.py          # 包 <memory>
│   ├── formatter.py        # 只注入 user 原文 + 查询相关裁剪
│   └── token_budget.py
└── commands/               # remember / forget / inspect / correct
```

运维脚本：`scripts/memory/`（`backfill_episodes.py` / `verify_index.py` / `migrate_profiles.py` / `compare_backends.py`）。

评测：`evaluate/memory/data/all_30.jsonl`（构建见 §9）+ `episodic_eval/` + `runs/`。

---

## 3. 架构总览

```
对话轮次
├─ 写（post_turn，异步/后台）
│    MemoryService.run_post_turn
│      → MemoryEngine.ingest_message
│           EpisodeIngestor：chunker → BGE-M3 → EpisodeStore
│           ProfileService：显式记住 / LLM提议pending / 确认 / 忘掉
│      → （独立）滚动会话摘要：裸基座（disable_adapter）
│
└─ 读（本轮生成前）
     MemoryService.build_chat_context
       ① 会话摘要块 <session-so-far>（若有）
       ② 若用户开启长期记忆：
            MemoryEngine.build_context
              RetrievalPipeline.run
                router → hybrid seeds → WindowExpander
                → merge → rerank → chain_link → coverage
                → Profile 全量注入
              → format → <memory> + _MEMORY_SYSTEM_PREFIX
       ③ 自上次摘要以来的原文（verbatim 窗）
```

设计要点：

- **助理话轮也入库**：用于语义邻接与扩窗连续性，但 **注入块默认只放用户原文**（`formatter._pick_user_turns`：每窗最多 6 条 user 话轮、优先 seed 与查询相关句），避免把咨询师复述当成用户事实、也节省预算。
- **本轮已在 prompt 里的原文 message_id 会从召回排除**，避免与 verbatim 窗重复。
- **链路异常只打日志、返回空块**，对话永不因记忆挂死。

---

## 4. 存储层（`store/repository.py`）

SQLite（与主库同文件或同路径策略由引擎构造时传入），逻辑表：

| 表 | 作用 |
|---|---|
| `memory_episode_chunks` | 分块原文：`user_id, conversation_id, message_id, role, position, chunk_index, content, created_at, is_deleted`；`UNIQUE(message_id, chunk_index)` |
| `memory_episode_embeddings` | 每块向量 JSON；`chunk_id` 关联 |
| `memory_embed_retry` | 嵌入失败重试（`attempts` 上限见 `MEMORY_EMBED_RETRY_MAX`） |
| `support_profile_items` | ProfileItem：`origin ∈ {explicit, confirmed}`，`status ∈ {active, deleted}` |
| `support_profile_sources` | 画像 ↔ 来源 message_id |
| `support_profile_pending` | 助理提议队列（每用户最多 8 条；确认时按先进先出全部落库） |
| `support_profile_llm_state` | LLM 提议攒批游标（未抽消息数 / 批次开始时间 / 上次尝试），按用户一行 |
| `memory_backfill_checkpoint` | 历史回填断点 |

**幂等**：同 `(message_id, chunk_index)` 重写不产生重复块。  
**删除**：软删 chunk（`is_deleted=1`）并清 embedding；画像软删。  
表名保留 `memory_episode_*` / `support_profile_*`，与 Python 层 `EpisodeStore` / `ProfileService` 一一对应。

---

## 5. 写路径（`store/ingest.py`）

1. `ingest_message(EpisodeMessage)`：仅 `user` / `assistant`。
2. **分块**（`chunker.py`，无 LLM）  
   - 长度 ≤ `MEMORY_CHUNK_SOFT_LIMIT`（默认 500）→ 整段一块；  
   - 否则按 `。！？；\n` 切开，再按 `TARGET_MIN/MAX`（150–350）合并。
3. `upsert_chunks` → 对每块 `MemoryEmbedder.embed`；失败入 `memory_embed_retry`，后续 `process_embed_retries` 补跑。
4. **Profile 旁路**（仅在开关打开时，`profile/service.py`）  
   - 用户「记住/别忘了…」→ 规则快路径 `create_explicit`；  
   - **助理回合后**（`MEMORY_PROFILE_LLM_PROPOSE=1`，默认开）：借鉴 MemMachine **攒批再抽**——每条 user/assistant 消息计入 `support_profile_llm_state`；当未抽消息数 ≥ `MEMORY_PROFILE_LLM_TRIGGER_MESSAGES`（默认 **5**）或批次年龄 ≥ `MEMORY_PROFILE_LLM_TRIGGER_AGE_SEC`（默认 **300s**）时，才调用中文 LLM（`profile/proposer.py`）。**pending 为空**才抽、每次最多 `MEMORY_PROFILE_LLM_PROPOSE_MAX`（默认 **1**）条、须能对上用户原文证据 → `propose` 入 **pending**（**绝不直接写 active**）；再过 `policy`；闲聊由模型自行返回空列表，**不用关键词门控**；无论是否抽到候选，尝试后都会重置批次计数；  
   - 助理文本里「要不要我记住…」正则 offer 仍可作为弱补充；  
   - 用户整句「可以/对/好的…」→ `pop_all_pending`，**一次确认全部**为 `confirmed`；  
   - 「A 改成 B」/「我现在不要 A」→ `correct`；「忘掉/删掉…」→ 软删；「查看记忆」→ 仅已确认偏好；  
   - **政策拒绝**：诊断词、人格标签、瞬时情绪前缀等。  
   - 与 MemMachine 差异：MemMachine 英文 LLM **攒批抽取后即写入画像**；SoulHarbor **攒批 + 单条候选 + 确认后才是记忆**。LongMemEval 评测已归档，不混入产品默认。
5. **会话摘要**（`MemoryService`，非 AER 长期记忆）：未摘要 token 超阈值时用裸基座生成滚动摘要，写入 `conversations.summary`，属于短期工作记忆。

写路径对 Episode **不做**：fact 抽取、合并覆盖、时间衰减关闭、层级晋升。Profile 旁路是唯一可选用 LLM 的写入理解（只进 pending，确认后才 active）。历史块默认长期可召回，由读路径预算与相关性决定是否进入上下文。

---

## 6. 读路径：RetrievalPipeline（逐步）

入口：`MemoryEngine.build_context` → `RetrievalPipeline.run(user_id, query, exclude_message_ids)`。

### 6.1 查询路由（`retrieval/router.py`）

- **无关键词 / 正则门控**。`MEMORY_SPLIT_QUERY_ENABLED=0` 时整句 `direct`。  
- 有 LLM 时：用 `_PLANNER_SYSTEM` 规划器输出一行 JSON（`mode` + `queries`）；`split` 最多 3 条互不重复的子查询；解析失败或不足 2 条 → `direct`。  
- 无 LLM / 调用失败：整句 `direct`（hybrid 已能覆盖多数单目标问题）。

### 6.2 Hybrid 召回 → Seeds

- **SemanticSearcher**：默认 FAISS `IndexFlatIP`（归一化向量上的内积 ≈ cosine），按 `user_id` 进程内缓存；`active_embedding_fingerprint` 变化时重建；`MEMORY_ANN_ENABLED=0` 或 faiss 不可用时回退 Python 循环。assistant 块仍 × `MEMORY_ASSISTANT_WEIGHT`。  
- **LexicalSearcher**：字级 unigram + bigram BM25（中文友好，不依赖分词器）；仍扫该用户活跃文本。  
- **RRF**（`fusion.reciprocal_rank`，k=60）融合，取 `MEMORY_SEED_TOP_K`（默认 8）作为 seeds。  
- split 模式：各子查询各自召回后再 RRF 合并（`split_query.py`）。

### 6.3 AER 扩窗（`store/expand.py`）★

对每个 seed 生成一个 `EpisodeWindow`：

**adaptive（默认）**  
从 seed 所在会话位置向左/右游走。候选是否纳入看连续性得分：

\[
\begin{aligned}
s =&\; 0.40\cdot\mathrm{cos}(c,\mathrm{pivot}) + 0.22\cdot\mathrm{cos}(c,\mathrm{span}) \\
&+ 0.18\cdot\mathrm{Jaccard}_{\mathrm{ent}} + 0.10\cdot\mathrm{pos} + 0.10\cdot\mathrm{Jaccard}_{\mathrm{query}}
\end{aligned}
\]

- 默认阈值 `MEMORY_CONTINUITY_THRESHOLD=0.28`；连续失败 2 次停止；  
- 跨度 ≤ `MEMORY_EXPAND_MAX_SPAN`（12），消息数 ≤ `MEMORY_WINDOW_MAX_MESSAGES`（10）；  
- **逃生口**：embedding 弱但强共指实体——候选与 seed 实体 Jaccard ≥0.35 且得分 ≥ 0.75×阈值仍纳入（实体共指比语义更可靠地标识"同一件事"）；  
- adaptive 扩完后保留 `max(2×MEMORY_WINDOW_TOP_K, MEMORY_SEED_TOP_K)` = 12 个候选供下游合并/串链/选择。

**fixed**  
退化为邻居窗：`MEMORY_NEIGHBOR_BEFORE/AFTER`（默认 2/2），再按会话合并重叠窗。

产出的是「一段经历」，而不是孤立句子——这对 context / integration 类问题至关重要。

### 6.4 合并、重排、串链、覆盖选择

1. **span_merge**：同会话**位置重叠或相邻**（`min(cur_pos) ≤ max(prev_pos)+1`）的窗合并，控制消息上限（超限按 seed 居中裁剪）。  
2. **feature rerank**（无 LLM）：`fused×2.0 + 查询词重叠×1.5 + 含 user 话轮×0.3 − 消息数长度税`，pool 取 `max(2×window_top_k, 8)`。  
3. **chain_link**：跨会话贪心串链（实体 0.30 / 语义 0.30 / 时间序 0.20 / 查询增益 0.15 / 同会话 0.1），链内按时间排序，打上 `chain_id`（E1/E2…）。时间分按间隔**指数衰减**：`time = max(0.05, 0.35·exp(−days/90))`（同天 ≈0.35，90 天 ≈0.12，约 180 天到 0.05 下限）；倒序但 3 天内给 0.15，更早的倒序给 0。**比链尾早 >14 天的事件跳过不串**——"太远的旧事不该被硬拉进当前话题"；`E{no}` 标签在"多链或链内多窗"时给出。  
4. **coverage**（默认）：贪心选 `MEMORY_WINDOW_TOP_K`（6），得分 ≈  
   `min(相关,2.2) + 1.6×新增信息覆盖 + 1.0×查询词增益 + 时间桶首现(0.9)/重复(−0.15) − 1.8×与已选冗余`，**剩余最佳分 <0.05 提前终止**。  
   也可切 `EVIDENCE_SELECTION_MODE=topk` 做消融。

### 6.5 Profile 注入

active 支持偏好数量小且粘性高——**不按查询过滤**，`list_for_inject` 全量注入（默认上限 30），与经历窗一并交给 formatter。

### 6.6 渲染与注入

- `context/formatter.py`：按链分组；**仅 user 话轮**（每窗注入上限 6 条，优先 seed 与查询相关句）；默认尽量整段注入（seed ≤1000 / 普通 ≤800 字）。仅极端超长才用 **BGE 句子级语义相似度** 定位相关局部（批量 embed，失败则保尾部）；总长仍由 `<memory>` token 预算整行裁剪。  
- `context/builder.py`：包成 `<memory>…</memory>`，默认 token 预算 1600；`trim_lines_to_budget` 到首个超预算行截断，builder 再兜底丢尾部整行。  
- `service.py` 在 system 侧加 `_MEMORY_SYSTEM_PREFIX`：说明块是历史证据、可部分采用、冲突以当前话语为准、禁止编造。

`RetrievalTrace` 记录 mode、pivots、bundles、链条数、时延等，便于日志与评测对照。

---

## 7. 与聊天主路径的衔接

| 步骤 | 行为 |
|---|---|
| 意图分类 | `session_context.messages_for_classifier`：短窗，不含 memory |
| 咨询路由 | MacBERT 决定咨询/闲聊强度；记忆开关由用户 `memory_enabled` 与 `mem_cfg` 共同决定 |
| 生成 | 单 chat LoRA（按咨询/闲聊调 scale）；记忆块已在 system；流式输出 |
| post_turn | 落库消息 → ingest → 可能触发摘要 |

关闭长期记忆时：仍有会话摘要 + 近期原文，行为退化为普通多轮助手。

---

## 8. 配置一览（`memory/config.py`）

| 变量 | 默认 | 含义 |
|---|---|---|
| `MEMORY_BACKEND` | `aer` | 后端标识（仅 `aer` 走本实现） |
| `MEMORY_STORE_ENABLED` | 1 | Episode（原文库）读写 |
| `MEMORY_PROFILE_ENABLED` | 1 | Profile |
| `MEMORY_PROFILE_LLM_PROPOSE` | 1 | 助理回合后中文 LLM 提议 pending（仍需用户确认） |
| `MEMORY_PROFILE_LLM_PROPOSE_MAX` | 1 | 每次最多提议条数 |
| `MEMORY_PROFILE_LLM_SKIP_IF_PENDING` | 1 | 已有待确认时不再抽新候选 |
| `MEMORY_PROFILE_LLM_TRIGGER_MESSAGES` | 5 | 未抽消息数达到此值才尝试 LLM 提议（MemMachine 式攒批） |
| `MEMORY_PROFILE_LLM_TRIGGER_AGE_SEC` | 300 | 批次年龄（秒）达到此值也触发尝试；`0` 关闭年龄触发 |
| `MEMORY_SPLIT_QUERY_ENABLED` | 1 | LLM 查询规划（`direct` / `split`；关掉则整句 direct） |
| `MEMORY_BUNDLE_RERANK_ENABLED` | 1 | 特征重排（无 LLM） |
| `MEMORY_EXPAND_MODE` | `adaptive` | `adaptive` \| `fixed` |
| `MEMORY_CONTINUITY_THRESHOLD` | 0.28 | 扩窗阈值 |
| `MEMORY_EXPAND_MAX_SPAN` | 12 | 最大位置跨度 |
| `EVIDENCE_SELECTION_MODE` | `coverage` | `coverage` \| `topk` |
| `MEMORY_CROSS_SESSION_LINKING` / `MEMORY_LINK_THRESHOLD` | 1 / 0.22 | 跨会话链 |
| `MEMORY_CONTEXT_TOKEN_BUDGET` | 1600 | 注入预算 |
| `MEMORY_SEMANTIC_TOP_K` / `MEMORY_LEXICAL_TOP_K` / `MEMORY_SEED_TOP_K` / `MEMORY_WINDOW_TOP_K` | 30 / 30 / 8 / 6 | 各阶段宽度 |
| `MEMORY_WINDOW_MAX_MESSAGES` | 10 | 单窗消息上限 |
| `MEMORY_NEIGHBOR_BEFORE` / `MEMORY_NEIGHBOR_AFTER` | 2 / 2 | `fixed` 扩窗邻域 |
| `MEMORY_CHUNK_SOFT_LIMIT` / `MEMORY_CHUNK_TARGET_MIN` / `MEMORY_CHUNK_TARGET_MAX` | 500 / 150 / 350 | 分块阈值 |
| `MEMORY_ASSISTANT_WEIGHT` | 0.55 | 语义召回时 assistant 块降权 |
| `MEMORY_EMBED_RETRY_MAX` | 5 | 嵌入重试上限 |
| `MEMORY_ANN_ENABLED` | 1 | 语义侧 FAISS 缓存；0 则 Python 循环 |
| `MEMORY_OBSERVABILITY` | 1 | trace 日志 |

`product_app/start.sh` 默认 `export MEMORY_BACKEND=aer`。

---

## 9. 评测数据如何构建（`evaluate/memory/data/all_30.jsonl`）

> 规范原文曾写在 `SPEC.md`（现已随分册 JSONL 归档到 `archive/eval_memory_data_splits/`）。  
> **仓库只保留完整集** `evaluate/memory/data/all_30.jsonl`（30 行，每行一条 case）。  
> 实验协议与数字见第 10 节与 `06_记忆系统_对比实验设计.md`。

### 9.1 目标与原则

为 AER（原文库 + 查询时重建）构造**像真实多轮心理咨询**的长程 QA，用来对照 flat 抽取式（Mem0）：

| 原则 | 含义 |
|---|---|
| 真实咨询弧，不窄触发 | 对话不要写成「为某个规则/关键词定制」；系统强项应在细节丰富的自然弧里自然发挥 |
| 原文可检索 | 人名 / 课程 / 地点 / 数字 / 事件要具体，hybrid 能召回；Mem0 扁平抽取易丢/合并 |
| 情境会演进 | 部分问题持续、部分后来好转——自然带出「当前状态 vs 已解决」 |
| gold 无歧义 | 每题答案由对话唯一确定；干扰项含 stale / 过泛化 / 合并错误 |
| 无记忆元操作泄漏 | 禁「系统记住 / 长期模式 / 更新记忆 / 固化 / 概括」等表述 |

### 9.2 构建流程（MiniMax 合成 + 人工验收）

整套 30 条由 **MiniMax 大模型按规范生成**，再人工校对与扫描，大致流水线：

```
① 定规范（6 能力维 × schema × 真实性硬规则）
      ↓
② 按类出种子：学生人设 + 周历大纲 + 本类要埋的 gold 事实
      ↓
③ MiniMax 生成：5–8 个会话的咨询对话（user/assistant）
      + 3 道测试题（2 MCQ + 1 judge）+ gold_facts
      ↓
④ 自动/半自动校验
      · schema / 周数 / 会话数
      · 禁词扫描
      · gold 是否可由对话唯一推出
      · 会话间是否有「上次…」类回指
      ↓
⑤ 人工改写：补细节、修歧义选项、压 flat 易错的干扰项
      ↓
⑥ 合并为 all_30.jsonl（现行唯一评测入口）
```

说明：

- **生成模型**：MiniMax（与评测 reader 同属 MiniMax 系；数据生成与答题是两套用途——生成偏创作温度，评测答题用固定 reader、低温）。  
- **assistant 口吻**：共情、提问、反映，2–4 句；不替用户下结论，不提记忆系统。  
- **分册产物**：曾按类导出 `detail_recall.jsonl` 等 + 各类 README；整理后只留完整集，分册与 `SPEC.md` 进 archive。

### 9.3 Schema（每行一个 JSON）

```json
{
  "case_id": "detail_recall_001",
  "category": "detail_recall",
  "user": {"username": "u_detail_recall_001"},
  "conversations": [
    {
      "sid": "detail_recall_001-s01",
      "checkpoint": "s1",
      "week": 1,
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
  ],
  "test_questions": [
    {
      "qid": "q1",
      "type": "mcq",
      "probes": "recall",
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ..."],
      "gold": "B"
    },
    {
      "qid": "q2",
      "type": "mcq",
      "probes": "current",
      "question": "...",
      "options": ["A. ...", "B. ..."],
      "gold": "A"
    },
    {
      "qid": "q3",
      "type": "judge",
      "probes": "context",
      "question": "...",
      "gold": "参考答案（开放题判分依据）"
    }
  ],
  "gold_facts": {
    "should_remember": ["应长期可召回的具体事实…"],
    "should_forget": ["flat 系统易误存/过泛化的内容…"]
  }
}
```

字段要点：

- `conversations[].week`：弧上的相对周次，评测 ingest 时用作时间锚。  
- `checkpoint`：会话在弧中的阶段标记（`s1`…），便于对照。  
- `test_questions[].probes`：能力标签（`recall` / `context` / `preference` / `current` / `integration` / `temporal` 等），用于分 probe 统计。  
- `gold_facts.should_forget`：不是「对话里说过要忘掉」，而是**对照基线不该固化成稳定事实**的内容（一次性事件、旧状态、人格标签等）。

### 9.4 六类能力维（每类 5 条 = 30）

对话都按同一真实弧标准写，**只是测试题侧重不同**：

| category | 问题侧重 | 相对 Mem0 的优势来源 |
|---|---|---|
| `detail_recall` | 人名 / 课程 / 数字 / 事件 | 原文 + 词法召回 |
| `context_recall` | 「那件事里还提到什么 / 当时情况」 | AER 扩窗、富窗口 |
| `preference_recall` | 偏好 / 需求 / 自我照顾方式 | Profile（consent） |
| `current_state` | 某处境的**最新**状态 | 时序 + 相关选择抗 stale |
| `cross_session_integration` | 需拼多个会话才能答 | 跨会话证据链 + 覆盖选择 |
| `temporal_sequence` | 何时 / 先后 | 原文时间结构 + 链内排序 |

实测规模（完整集统计）：每案平均约 **6–7 个会话、跨 5–7 周**；每案固定 **3 题**（2 MCQ + 1 judge）。`cross_session_integration` 会话消息更密（约 24 条/案），其它类约 12–14 条/案。

### 9.5 真实性硬规则（生成与验收共用）

1. 每条 = **一个学生连续数周**的咨询弧；会话间有自然回指（「上次说的那个室友」「后来那个项目」），禁止互不相干 filler 拼接。  
2. **多话题交织**：学业 / 人际 / 家庭 / 求职 / 情感 / 自我成长在同一弧演进。  
3. **细节具体**：人名（室友小王、导师张老师）、课程名、地点、数字自然嵌入。  
4. **情境演进**：有的问题持续、有的后来好转；用自然事件表达，不硬塞 CLOSE/UPDATE 触发词。  
5. **assistant = 咨询师风格**；禁止记忆系统元操作。  
6. **gold 唯一**；MCQ 干扰含 flat 易错项（stale 旧状态、合并细节、过泛化人格）。

### 9.6 测试题怎么出

- **q1（MCQ）**：对准该类主能力；选项 = 正确答案 + 合理干扰（常含 Mem0 易错的 stale/合并项）。  
- **q2（MCQ）**：同弧另一自然记忆点（常是 detail 或 current）。  
- **q3（judge）**：开放回忆（来龙去脉 / 感受 / 偏好总结）；`gold` 给参考答案，评测时用同一 reader 做 reference-based 判定。  
- **`should_remember`**：2–4 条应能长期召回的具体事实。  
- **`should_forget`**：1–2 条「一次性事件被写成特质 / 旧状态当现状」——用于内容 F1 的负向对照。

### 9.7 数据样例（摘自 `preference_recall_001`）

下面是真实评测集中的一条（对话略缩；完整原文见 `all_30.jsonl`）。弧跨 7 周：失眠与室友作息 → 帮闺蜜分析受伤 → 选女咨询师 → 不满「一上来给建议」→ 换周老师、发现夜路独处有用 → 作息缓和、固定夜走 → 回访说清「先被听到再一起想」。

**会话片段（第 1 / 4 / 7 周）：**

```text
[week 1] user: 最近又开始失眠……和室友小李作息对不上……提过一次她说「没办法嘛赶时间」……
         assistant: 你试过一次被挡回来了……那之后你一般怎么让自己撑过去？

[week 4] user: ……赵老师一上来就讲「你可以试试」……我一直在想「我还没说完呢」……
               好像更喜欢先被听一会儿，再讨论怎么办。
         assistant: 你想先被完整听见，再一起想办法。这个顺序对你挺重要的。

[week 7] user: 周老师问什么对我有用……我比较希望先被听到，再一起想……
               学生会学术部我放弃了，把时间留给自己走那段路。
         assistant: 你能清楚说出自己的需要，又愿意为它做选择……
```

**测试题与 gold：**

```json
{
  "test_questions": [
    {
      "qid": "q1",
      "type": "mcq",
      "probes": "preference",
      "question": "用户在咨询中比较希望咨询师先做什么？",
      "options": [
        "A. 直接给她几个可操作的方法",
        "B. 先完整倾听她说完，再一起讨论",
        "C. 帮她逐条分析每个问题",
        "D. 给她布置具体的作业和练习"
      ],
      "gold": "B"
    },
    {
      "qid": "q2",
      "type": "mcq",
      "probes": "recall",
      "question": "用户在选择学校心理咨询师时的性别偏好是？",
      "options": [
        "A. 男老师，因为更理性",
        "B. 无所谓",
        "C. 女老师，因为更放松",
        "D. 年轻老师更好沟通"
      ],
      "gold": "C"
    },
    {
      "qid": "q3",
      "type": "judge",
      "probes": "preference",
      "question": "用户在自己情绪不好时，倾向于什么样的自我照顾方式？",
      "gold": "独处；晚上绕图书馆后小路走半小时左右，听慢节奏音乐；咨询中希望先被完整倾听再一起讨论。"
    }
  ],
  "gold_facts": {
    "should_remember": [
      "希望咨询师先完整倾听再一起讨论；自己想出来的办法更愿意执行",
      "选学校咨询师时偏好女老师",
      "夜走校园小路 + 慢节奏音乐舒缓情绪"
    ],
    "should_forget": [
      "用户是「冷淡/回避型人格」（单次偏好升格为人格标签）",
      "与室友小李关系彻底破裂（后来双方调整作息有所缓和）"
    ]
  }
}
```

**细节召回类一句对照**（`detail_recall_001`）：室友小周（电气）深夜语音 → 导师张老师鄱阳湖湿地碳汇 → 南门外螺蛳粉和解约定 11 点后戴耳机 → 字节二面英文 presentation 卡在模型选型页。题面直接问专业名 / 研究方向 / 面试经过——测的是「具体词是否还在库里」。

### 9.8 入口与归档

| 路径 | 内容 |
|---|---|
| `evaluate/memory/data/all_30.jsonl` | **唯一正式评测数据**（30 条完整弧） |
| `evaluate/memory/episodic_eval/` | AER / Mem0 QA harness |
| `archive/eval_memory_data_splits/` | 历史分册 JSONL、`SPEC.md`、各类 README（不参与默认评测） |

---

## 10. 评测协议与结果（指针）

- 协议：同一对话流逐轮 ingest → **固定 reader（MiniMax-M3）** 只看「该方法记忆块 + 测试题」作答；MCQ 精确匹配，judge 参考 `gold`。  
- 脚本：`evaluate/memory/episodic_eval/run_qa_episodic.py`（AER）、`run_qa_mem0.py`（同协议 Mem0）。  
- 正式结果：`evaluate/memory/runs/qa_episodic_*`、`qa_mem0_*`。  
- 解读与主表：`06_记忆系统_对比实验设计.md`。

---

## 11. 设计取舍与局限

**取舍**  
- 存原文换抗固化，代价是注入块更长、依赖扩窗与覆盖选择质量。  
- 不做写时图谱 / Neo4j，单机 SQLite + 向量 JSON，部署简单。  
- 助理话轮检索可用、注入不用：偏「用户事实」安全。

**局限**  
- temporal 类题相对弱（约 60–73%），可补显式时间锚。  
- stale 主要靠读路径相关选择与结束启发式，无写时生命周期状态机。  
- 评测集 30 条偏小；外部 LongMemEval 等可后续扩展。  

---

## 12. 本地校验

单元测试在本地目录 `product_app/tests/`（**不入库**），需要时自行运行：

```bash
cd /root/autodl-tmp/SoulHarbor
/root/autodl-tmp/CondaEnv/soulhar/bin/python -m unittest \
  product_app.tests.memory.test_aer \
  product_app.tests.memory.test_episodic_core -v
```

覆盖：自适应扩窗、覆盖度多样性、跨会话时序、formatter 语义截取、分块/幂等/隔离/画像政策等。
