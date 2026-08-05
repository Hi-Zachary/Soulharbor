# Trace memory long-horizon QA

- **唯一正式数据**：`data/all_50.jsonl`（50 案 / 250 题）
- **终版 ER**（CE Top-k 读路径）：`runs/qa_er_all50/`
  - QA **90.8%** / micro **92.0%**（230/250）/ Stale **6.9%** / F1 **0.602**
  - 源跑次：`runs/qa_er_20260805_223327/`
- **Mem0**：`runs/qa_mem0_all50/`
  - case-avg QA **68.7%** / micro **71.2%**（178/250）/ Stale **46.6%** / F1 **0.400**
- 历史跑次：`../../archive/memory_pre_ce_topk_20260805/`（含旧 coverage CE `qa_er_all50_pre_final`）
- 协议：`er_eval/`；结果解读：`../../项目文档/03_详尽技术文档.md` §8
