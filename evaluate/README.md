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
└── memory/                # 长期记忆 QA 评测（ER vs Mem0）
    ├── data/              # all_30.jsonl（完整 30 条）
    ├── runs/              # 正式结果：qa_er_* / qa_mem0_*
    └── er_eval/     # 评测协议脚本
```

## 主对话 — `dialogue/`

- 入口：`dialogue/run_soulchat_eval_pipeline.sh`
- 说明：`dialogue/README.md`、`dialogue/docs/`

## 长期记忆 — `memory/`

- **数据**：`memory/data/all_30.jsonl`（30 条）
- **协议**：`er_eval/run_qa_er.py`（ER）与 `run_qa_mem0.py`（Mem0，同协议对照）
- **结果**：`memory/runs/`（`summary.json` + `results.jsonl`）
  - ER 终版：`qa_er_20260802_210915`（QA ≈92%，staleness ≈14%，content F1 ≈0.86）；历史跑次见 `archive/eval_memory_runs_pre_final_20260802/`
  - Mem0：`archive/eval_memory_runs_pre_final_20260802/qa_mem0_20260731_135743`（QA ≈64%，content F1 ≈0.48）
- 方案与结果解读：`项目文档/05_记忆系统_ER经历重建_最终方案.md`、`项目文档/06_记忆系统_对比实验设计.md`

## 已有结果速查

| 任务 | 路径 |
|------|------|
| SoulChat 指标 | `dialogue/runs/soulchat_corpus_eval_20260426_092741/` |
| SMILE 指标 | `dialogue/runs/smile_zeroshot_eval_20260427_055722/` |
| 记忆 QA（ER） | `memory/runs/qa_er_20260802_210915/` |
| 记忆 QA（Mem0） | `memory/runs/qa_mem0_20260731_135743/` |
