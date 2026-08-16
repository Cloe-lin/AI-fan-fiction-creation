import logging
import os
from pathlib import Path

import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.config import settings

logger = logging.getLogger(__name__)

_deepseek_embedding_available: bool | None = None
_embedding_fn_cache: EmbeddingFunction | None = None


def _apply_hf_env() -> None:
    """尽早设置 HuggingFace 镜像，避免默认连 huggingface.co 卡住。"""
    endpoint = (settings.hf_endpoint or "").strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        # 部分库仍读 HF_HUB_BASE
        os.environ.setdefault("HUGGINGFACE_HUB_BASE_URL", endpoint)


def _model_is_cached(model_name: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    folder = "models--" + model_name.replace("/", "--")
    snapshots = hub / folder / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


class BgeEmbeddingFunction(EmbeddingFunction[Documents]):
    """本地 BGE Embedding，small 用 fastembed，large 用 sentence-transformers。"""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._backend = None
        self._model = None

        fastembed_models = {
            m["model"]
            for m in __import__(
                "fastembed", fromlist=["TextEmbedding"]
            ).TextEmbedding.list_supported_models()
        }

        if model_name in fastembed_models:
            from fastembed import TextEmbedding

            logger.info("加载 fastembed 模型: %s", model_name)
            self._backend = "fastembed"
            self._model = TextEmbedding(model_name=model_name)
            return

        _apply_hf_env()
        from sentence_transformers import SentenceTransformer

        cached = _model_is_cached(model_name)
        logger.info(
            "加载 sentence-transformers: %s（缓存=%s，镜像=%s）",
            model_name,
            cached,
            settings.hf_endpoint or "huggingface.co",
        )
        self._backend = "sentence_transformers"

        if cached:
            try:
                self._model = SentenceTransformer(model_name, local_files_only=True)
                logger.info("已从本地缓存离线加载模型")
                return
            except Exception as e:
                logger.warning("离线加载失败，尝试联网: %s", e)

        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        if self._backend == "fastembed":
            return [embedding.tolist() for embedding in self._model.embed(input)]

        vectors = self._model.encode(
            list(input),
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.tolist()


class DoubaoEmbeddingFunction(EmbeddingFunction[Documents]):
    """火山方舟豆包 Embedding，兼容文本 /embeddings 与多模态 /embeddings/multimodal。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=60.0)
        self._use_multimodal: bool | None = None
        logger.info("使用豆包 Embedding: %s @ %s", model, self.base_url)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _parse_vectors(self, payload: dict, expected: int) -> list[list[float]]:
        data = payload.get("data")

        # 多模态接口：data 是 {embedding: [...]} 对象
        if isinstance(data, dict) and "embedding" in data:
            emb = data["embedding"]
            if emb and isinstance(emb[0], list):
                emb = emb[0]
            if expected != 1:
                raise RuntimeError("多模态接口单次仅支持 1 条文本")
            return [emb]

        if isinstance(data, list):
            data_sorted = sorted(
                data,
                key=lambda x: x.get("index", 0) if isinstance(x, dict) else 0,
            )
            vectors: list[list[float]] = []
            for item in data_sorted:
                if not isinstance(item, dict):
                    continue
                emb = item.get("embedding")
                if emb is None:
                    continue
                if emb and isinstance(emb[0], list):
                    emb = emb[0]
                vectors.append(emb)
            if len(vectors) != expected:
                raise RuntimeError(
                    f"豆包返回向量数不匹配: expect={expected} got={len(vectors)}"
                )
            return vectors

        raise RuntimeError(f"无法解析豆包 Embedding 响应: {str(payload)[:200]}")

    def _embed_text_api(self, batch: list[str]) -> httpx.Response:
        return self._client.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json={"model": self.model, "input": batch, "encoding_format": "float"},
        )

    def _embed_multimodal_api(self, text: str) -> httpx.Response:
        # vision 多模态接入点：一次只能嵌一条文本
        return self._client.post(
            f"{self.base_url}/embeddings/multimodal",
            headers=self._headers(),
            json={"model": self.model, "input": [{"type": "text", "text": text}]},
        )

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self._use_multimodal is True:
            vectors: list[list[float]] = []
            for text in batch:
                resp = self._embed_multimodal_api(text)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"豆包 Embedding 调用失败 ({resp.status_code}): {resp.text[:300]}"
                    )
                vectors.extend(self._parse_vectors(resp.json(), 1))
            return vectors

        if self._use_multimodal is False:
            resp = self._embed_text_api(batch)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"豆包 Embedding 调用失败 ({resp.status_code}): {resp.text[:300]}"
                )
            return self._parse_vectors(resp.json(), len(batch))

        # 首次探测：先试文本接口，不行再切多模态
        resp = self._embed_text_api(batch)
        if resp.status_code == 200:
            self._use_multimodal = False
            return self._parse_vectors(resp.json(), len(batch))

        body = resp.text
        if "multimodal" in body or "vision" in body or "does not support this api" in body:
            logger.info("检测到多模态 Embedding 接入点，切换 /embeddings/multimodal")
            self._use_multimodal = True
            return self._embed_batch(batch)

        raise RuntimeError(
            f"豆包 Embedding 调用失败 ({resp.status_code}): {body[:300]}"
        )

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        if not texts:
            return []

        # 首次探测模式
        if self._use_multimodal is None:
            probe = self._embed_batch(texts[:1])
            if len(texts) == 1:
                return probe
            rest = texts[1:]
        else:
            probe = []
            rest = texts

        if self._use_multimodal:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            vectors: list[list[float] | None] = [None] * len(rest)
            workers = min(8, max(1, len(rest)))

            def _one(idx_text: tuple[int, str]) -> tuple[int, list[float]]:
                idx, text = idx_text
                resp = self._embed_multimodal_api(text)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"豆包 Embedding 调用失败 ({resp.status_code}): {resp.text[:300]}"
                    )
                return idx, self._parse_vectors(resp.json(), 1)[0]

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_one, (i, t)) for i, t in enumerate(rest)]
                for fut in as_completed(futures):
                    idx, vec = fut.result()
                    vectors[idx] = vec
            return probe + [v for v in vectors if v is not None]

        batch_size = 32
        all_vectors = list(probe)
        for start in range(0, len(rest), batch_size):
            batch = rest[start : start + batch_size]
            all_vectors.extend(self._embed_batch(batch))
        return all_vectors


def clear_embedding_cache() -> None:
    global _embedding_fn_cache
    _embedding_fn_cache = None


def get_embedding_function():
    global _embedding_fn_cache
    if _embedding_fn_cache is not None:
        return _embedding_fn_cache

    resolved = settings.resolved_embedding

    if resolved["mode"] == "local_bge":
        _embedding_fn_cache = BgeEmbeddingFunction(settings.embedding_local_model)
        return _embedding_fn_cache

    if resolved["provider"] == "deepseek" and not _check_deepseek_embedding():
        _embedding_fn_cache = BgeEmbeddingFunction(settings.embedding_local_model)
        return _embedding_fn_cache

    if not settings.use_api_embedding:
        logger.warning("Embedding API 配置不完整，回退到 local_bge")
        _embedding_fn_cache = BgeEmbeddingFunction(settings.embedding_local_model)
        return _embedding_fn_cache

    if resolved["provider"] == "doubao":
        _embedding_fn_cache = DoubaoEmbeddingFunction(
            api_key=resolved["api_key"],
            base_url=resolved["base_url"],
            model=resolved["model"],
        )
        return _embedding_fn_cache

    from chromadb.utils import embedding_functions

    logger.info(
        "使用 %s Embedding API: %s @ %s",
        resolved["provider"],
        resolved["model"],
        resolved["base_url"],
    )
    _embedding_fn_cache = embedding_functions.OpenAIEmbeddingFunction(
        api_key=resolved["api_key"],
        api_base=resolved["base_url"],
        model_name=resolved["model"],
    )
    return _embedding_fn_cache


def _check_deepseek_embedding() -> bool:
    global _deepseek_embedding_available
    if _deepseek_embedding_available is not None:
        return _deepseek_embedding_available

    resolved = settings.resolved_embedding
    if not resolved["api_key"]:
        _deepseek_embedding_available = False
        return False

    try:
        resp = httpx.post(
            f"{resolved['base_url'].rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {resolved['api_key']}"},
            json={"model": resolved["model"], "input": "ping"},
            timeout=8,
        )
        _deepseek_embedding_available = resp.status_code == 200
    except Exception:
        _deepseek_embedding_available = False

    if not _deepseek_embedding_available:
        logger.warning(
            "DeepSeek Embedding API 暂不可用，已自动回退到 local_bge。"
        )
    return _deepseek_embedding_available


def get_embedding_info() -> dict:
    resolved = settings.resolved_embedding

    if resolved["mode"] == "local_bge":
        return {
            "provider": resolved["provider"],
            "mode": "local_bge",
            "model": resolved["model"],
            "base_url": "",
            "ready": True,
            "note": "",
        }

    if resolved["provider"] == "deepseek":
        # 不主动探测 API，避免阻塞启动；已知官方无 Embedding
        return {
            "provider": "deepseek",
            "mode": "local_bge",
            "model": settings.embedding_local_model,
            "base_url": "",
            "ready": True,
            "note": "DeepSeek 暂无 Embedding 接口，已自动使用本地 BGE",
        }

    if settings.use_api_embedding:
        return {
            "provider": resolved["provider"],
            "mode": "api",
            "model": resolved["model"],
            "base_url": resolved["base_url"],
            "ready": True,
            "note": "",
        }

    note = ""
    if resolved["provider"] == "doubao" and not resolved["model"]:
        note = "请在 .env 设置 EMBEDDING_MODEL=你的火山方舟接入点ID（ep-xxx）"
    elif not resolved["api_key"]:
        note = "请设置 EMBEDDING_API_KEY"

    return {
        "provider": resolved["provider"],
        "mode": "local_bge",
        "model": settings.embedding_local_model,
        "base_url": "",
        "ready": False,
        "note": note or "API 配置不完整，当前使用 local_bge",
    }


def get_collection_suffix() -> str:
    info = get_embedding_info()
    safe_model = info["model"].replace("/", "_").replace(".", "_").replace("-", "_")
    if info["mode"] == "api":
        return f"{info['provider']}_{safe_model}"
    return f"{info['mode']}_{safe_model}"
