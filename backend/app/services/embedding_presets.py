"""Embedding 服务商预设配置。"""

PROVIDER_PRESETS: dict[str, dict] = {
    "local_bge": {
        "mode": "local_bge",
        "description": "本地 BGE 中文模型，完全免费",
    },
    "doubao": {
        "mode": "api",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-embedding-large-text-250515",
        "description": "火山方舟豆包 Embedding（推荐 doubao-embedding-large）",
        "requires_api_key": True,
        "requires_model": True,
    },
    "deepseek": {
        "mode": "api",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-embedding-v2",
        "description": "DeepSeek Embedding（官方暂未开放，会自动回退 local_bge）",
        "fallback": "local_bge",
    },
    "custom": {
        "mode": "api",
        "description": "自定义 OpenAI 兼容 Embedding API",
        "requires_api_key": True,
        "requires_model": True,
    },
}
