from pydantic_settings import BaseSettings

from app.services.embedding_presets import PROVIDER_PRESETS

_PLACEHOLDER_KEYS = {"", "your_api_key_here", "sk-xxx", "sk-your-key-here"}


class Settings(BaseSettings):
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_review_model: str = "gpt-4o"
    data_dir: str = "app/data"

    # RAG 配置
    rag_enabled: bool = True
    rag_collection: str = "modaozuoshi"
    rag_persist_dir: str = "app/data/chroma"
    rag_source_subdir: str = "modaozuoshi/text"
    rag_novel_source: str = ""
    rag_top_k: int = 5
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 100

    # Embedding: local_bge | doubao | deepseek | custom
    embedding_provider: str = "local_bge"
    embedding_local_model: str = "BAAI/bge-large-zh-v1.5"
    hf_endpoint: str = "https://hf-mirror.com"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def effective_embedding_api_key(self) -> str:
        key = (self.embedding_api_key or self.llm_api_key).strip()
        return "" if key in _PLACEHOLDER_KEYS else key

    @property
    def resolved_embedding(self) -> dict:
        provider = self.embedding_provider.lower()
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["local_bge"])

        if preset["mode"] == "local_bge":
            return {
                "provider": provider,
                "mode": "local_bge",
                "model": self.embedding_local_model,
                "base_url": "",
                "api_key": "",
                "fallback": None,
                "description": preset.get("description", ""),
            }

        base_url = self.embedding_base_url or preset.get("base_url", "")
        model = self.embedding_model or preset.get("default_model", "")

        return {
            "provider": provider,
            "mode": "api",
            "model": model,
            "base_url": base_url,
            "api_key": self.effective_embedding_api_key,
            "fallback": preset.get("fallback"),
            "description": preset.get("description", ""),
        }

    @property
    def use_api_embedding(self) -> bool:
        resolved = self.resolved_embedding
        return resolved["mode"] == "api" and bool(
            resolved["api_key"] and resolved["model"] and resolved["base_url"]
        )


settings = Settings()

# 尽早注入 HF 镜像，避免库默认连 huggingface.co 超时卡住
import os as _os

if settings.hf_endpoint:
    _os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
