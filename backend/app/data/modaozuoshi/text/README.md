# Embedding 配置

## 推荐：DeepSeek 创作 + 本地 BGE（免费）

DeepSeek 官方**暂无 Embedding 接口**，向量化用本地 BGE 即可，创作仍用 DeepSeek：

```env
EMBEDDING_PROVIDER=local_bge
```

## 豆包 Embedding API

1. 登录 [火山方舟控制台](https://console.volcengine.com/ark)
2. 开通 Embedding 模型，创建接入点，获得 `ep-xxx` 格式 ID
3. 配置 `.env`：

```env
EMBEDDING_PROVIDER=doubao
EMBEDDING_API_KEY=你的火山方舟Key
EMBEDDING_MODEL=ep-xxxxxxxx
```

4. 重新入库：`venv\Scripts\python scripts\setup_rag.py`
