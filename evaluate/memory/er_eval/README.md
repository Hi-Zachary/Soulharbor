# Trace memory long-horizon QA（API）

评测 **SoulHarbor 经历重建（ER）长期记忆**，并可同协议对照官方 Mem0。

## Datasets

| 数据 | 脚本 | 说明 |
|------|------|------|
| `../data/50cases.jsonl` | `run_qa_er.py` | 旧 case-centric（50×5 题，含 MCQ / staleness / content F1） |
| `../data/soulharbor_mh_longmemeval_30_bundle/soulharbor_mh_longmemeval_30.jsonl` | `run_qa_lme.py` | 新 LongMemEval 风格（30 题，每题独立长历史） |

新基准 **完整 QA + 检索评测只需主 JSONL**；`oracle.json` 与 `profile_gold.jsonl` 分别用于诊断和画像评测（可选）。

## Setup

```bash
cp config.example.json config.json   # 填入 MiniMax api_key
```

Embedding 默认本地 `models/encoders/bge-m3`；对话/答题/Judge 走 MiniMax API。  
`config.json` 含密钥，不入库（见根目录 `.gitignore`）。

## Run

```bash
cd /root/autodl-tmp/SoulHarbor/evaluate/memory/er_eval

# 旧协议 ER
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_er.py --selftest
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_er.py \
  --data ../data/50cases.jsonl --workers 4 --qa-workers 3

# 新协议 ER（LongMemEval-30）
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_lme.py --selftest
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_lme.py --workers 2
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_lme.py --oracle

# Mem0（旧协议对照）
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_mem0.py \
  --data ../data/50cases.jsonl --workers 4 --qa-workers 3
```

结果写入 `../runs/qa_er_*/`、`../runs/qa_lme_*/`、`../runs/qa_mem0_*/`（`summary.json` + `results.jsonl`）。

## Metrics

### 旧协议（`run_qa_er.py`）

- `qa_accuracy`（MCQ + judge）
- `overgeneralization_rate` / `staleness_rate`
- `content_f1`（应记/应忘）
- `subclass_qa` / `probe_qa`

### 新协议（`run_qa_lme.py`）

- `qa.overall_accuracy` / `by_capability` / `by_evaluation_type`
- `qa.abstention_accuracy` / `structured_field_accuracy`
- `retrieval.message_recall_at_k` / `session_recall_at_k` / `all_evidence_recall` / `stale_only_rate`
