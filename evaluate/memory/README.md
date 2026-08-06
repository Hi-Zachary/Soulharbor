# Trace memory long-horizon QA

## Datasets

| 文件 | 协议 | 说明 |
|------|------|------|
| `data/50cases.jsonl` | case-centric（50 案 × 5 题） | 旧 SoulHarbor 结构化评测 |
| `data/soulharbor_mh_longmemeval_30_bundle/soulharbor_mh_longmemeval_30.jsonl` | question-centric LongMemEval | **新正式基准**（30 题，每题独立长历史） |

新基准配套文件见 bundle 内 `soulharbor_mh_longmemeval_30_README.md` 与 `soulharbor_mh_longmemeval_30_EVALUATION_GUIDE.md`。

## Runners

- **旧协议 ER**：`er_eval/run_qa_er.py --data ../data/50cases.jsonl`
- **新协议 ER**：`er_eval/run_qa_lme.py --data ../data/soulharbor_mh_longmemeval_30_bundle/soulharbor_mh_longmemeval_30.jsonl`
- **Mem0 对照**（旧协议）：`er_eval/run_qa_mem0.py`

## Baselines (50cases)

- **终版 ER**（CE Top-k 读路径）：`runs/qa_er_20260806_143259/` — QA **97.2%** / Stale **2.6%** / F1 **0.467**
- **Mem0**：`runs/qa_mem0_all50/` — QA **68.7%** / Stale **46.6%** / F1 **0.400**

协议细节：`er_eval/README.md`；结果解读：`../../项目文档/03_详尽技术文档.md` §8
