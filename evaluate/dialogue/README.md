# 主对话评测 `evaluate/dialogue`

```
dialogue/
├── README.md
├── run_soulchat_eval_pipeline.sh   # SoulChat 主入口（seen/unseen × base/sft/dpo）
├── run_smile_after_soulchat.sh     # SMILE 零样本（可选）
├── data/                           # 评测集 JSONL
├── runs/                           # 已跑结果
├── cache/                          # 抽样缓存
├── docs/                           # 说明与启动文档
├── sampling/                       # 抽评测集
├── runners/                        # 加载模型、生成、打分、汇总
└── lib/                            # 公共库（指标、instruction、加载器）
```

## 日常怎么用

```bash
cd /root/autodl-tmp/SoulHarbor
bash evaluate/dialogue/run_soulchat_eval_pipeline.sh
```

说明：`docs/主对话评测说明.md`、`docs/评测启动.md`

## 子目录职责

| 目录 | 放什么 |
|------|--------|
| `data/` | `soulchat_seen_1k.jsonl`、`soulchat_unseen_1k.jsonl`、`smile_1k.jsonl` 等 |
| `runs/` | 生成与指标结果目录 |
| `sampling/` | `sample_soulchat_eval_sets.py`、`sample_smile_eval_set.py` |
| `runners/` | `run_soulchat_paper_eval.py`、`run_single_dataset_paper_eval.py`、`summarize_*`、`compute_metrics_from_predictions.py` |
| `lib/` | `soulchat_paper_metrics.py`、`soulchat_model.py`、`*_to_instruction.py`、`metric.py` 等 |

入口脚本会把 `lib/`、`sampling/`、`runners/` 加入 `PYTHONPATH`。
