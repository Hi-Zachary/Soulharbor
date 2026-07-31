# SoulHarbor 评测（evaluate）

```
evaluate/
├── README.md
├── dialogue/              # 主对话评测（SoulChat / SMILE）
│   ├── data/              # 抽样评测集 JSONL
│   ├── runs/              # 已跑结果
│   ├── cache/             # 抽样缓存
│   ├── sampling/          # 抽评测集
│   ├── runners/           # 生成与打分
│   ├── lib/               # 公共库
│   └── docs/
└── memory/                # 长期记忆 QA 评测（AER vs Mem0）
    ├── data/              # all_30.jsonl（完整 30 条）
    ├── runs/              # 正式结果：qa_episodic_* / qa_mem0_*
    └── episodic_eval/     # 评测协议脚本
```

## 主对话 — `dialogue/`

- 入口：`dialogue/run_soulchat_eval_pipeline.sh`
- 说明：`dialogue/README.md`、`dialogue/docs/`

## 长期记忆 — `memory/`

- **数据**：`memory/data/all_30.jsonl`（30 条）
- **协议**：`episodic_eval/run_qa_episodic.py`（AER）与 `run_qa_mem0.py`（Mem0，同协议对照）
- **结果**：`memory/runs/`（`summary.json` + `results.jsonl`）
  - AER：`qa_episodic_20260731_145921`（QA ≈91%，content F1 ≈0.89）；另有 `qa_episodic_20260731_134643`（QA ≈93%，content F1 ≈0.79）
  - Mem0：`qa_mem0_20260731_135743`（QA ≈64%，content F1 ≈0.48）
- 方案与结果解读：`项目文档/05_记忆系统_AER自适应经历重建_最终方案.md`、`项目文档/06_记忆系统_对比实验设计.md`

## 已有结果速查

| 任务 | 路径 |
|------|------|
| SoulChat 指标 | `dialogue/runs/soulchat_corpus_eval_20260426_092741/` |
| SMILE 指标 | `dialogue/runs/smile_zeroshot_eval_20260427_055722/` |
| 记忆 QA（AER） | `memory/runs/qa_episodic_20260731_145921/` |
| 记忆 QA（Mem0） | `memory/runs/qa_mem0_20260731_135743/` |
