# SoulHarbor Frontend

Vue 3 + Vite 用户端 SPA。

## 开发

```bash
conda activate soulhar
cd product_app/frontend
pnpm install
pnpm dev          # http://127.0.0.1:5173 ，API 代理到 :8000
```

## 构建（产物供 FastAPI 托管）

```bash
pnpm build        # → product_app/static/spa/
```

FastAPI 检测到 `static/spa/index.html` 后，对 `/app` `/login` `/register` `/admin/login` 返回 SPA；`/admin` 仍为 Jinja 管理端。
