# SoulHarbor Product App（FastAPI + Jinja + Tailwind）

生产 Web 应用：用户对话 + 管理端 + 意图分流 + 双 LoRA 生成 + 长期记忆。

## 启动

```bash
cd /root/autodl-tmp/SoulHarbor
conda activate /root/autodl-tmp/CondaEnv/soulhar
bash product_app/start.sh
```

`start.sh` 会自动选择最新的 `dpo_synth_*`、`extraction_sft` 与 `bge-m3`。

- 用户端：`http://<host>:8000/app`
- 管理端：`http://<host>:8000/admin/login`（默认 `soulharbor_admin`）

完整说明：仓库根目录 `启动指令.md`、`项目文档/`。

## 本目录结构

```
product_app/
├── app/           # FastAPI 与记忆子系统
├── templates/     # 用户端 / 管理端页面
├── data/          # soulharbor.db（运行时库）
└── start.sh
```

训练、评测、烟测脚本已迁至 `archive/offline/`。
