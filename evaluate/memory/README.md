# Trace memory long-horizon QA

- **唯一正式数据**：`data/all_50.jsonl`（50 案 / 250 题）
- **终版 ER**（多查询 coverage Anchor CE）：`runs/qa_er_all50/`
  - QA **90.5%** / micro **90.8%**（227/250）/ Stale **8.6%** / F1 **0.783**
- **Mem0**：`runs/qa_mem0_all50/`
  - case-avg QA **68.7%** / micro **71.2%**（178/250）/ Stale **46.6%** / F1 **0.400**
- 历史分集数据与分跑次：`../../archive/memory_pre_final_coverage_ce_20260804/`
- 协议：`er_eval/`；结果解读：`../../项目文档/03_详尽技术文档.md` §8
