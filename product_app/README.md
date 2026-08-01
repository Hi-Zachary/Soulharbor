# SoulHarbor Product App（FastAPI + Vue）

生产 Web 应用：用户对话 + 管理端 + 意图分流 + chat LoRA 生成 + AER 长期记忆。

## 启动

```bash
cd /root/autodl-tmp/SoulHarbor
conda activate /root/autodl-tmp/CondaEnv/soulhar
bash product_app/start.sh
```

`start.sh` 会自动选择最新的 `dpo_synth_*` 与 `bge-m3`，并默认 `MEMORY_BACKEND=aer`。

- 用户端：`http://<host>:8000/app`
- 管理端：`http://<host>:8000/admin/login`（默认 `soulharbor_admin`）

完整说明：仓库根目录 `启动指令.md`、`项目文档/`。

## 本目录结构

```
product_app/
├── app/           # FastAPI 与记忆子系统（memory/ = AER）
├── frontend/      # Vue 3 SPA 源码
├── static/spa/    # 构建产物
├── templates/     # 管理端页面
├── data/          # soulharbor.db（运行时库）
└── start.sh
```
