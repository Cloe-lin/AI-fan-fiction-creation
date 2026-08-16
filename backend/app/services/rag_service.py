import hashlib
import logging
from pathlib import Path

import chromadb

from app.config import settings
from app.models.rag import CanonChunk, RagIngestResponse, RagStatusResponse
from app.services.character_service import character_service
from app.services.embedding import get_collection_suffix, get_embedding_function, get_embedding_info
from app.services.novel_registry import novel_registry
from app.services.text_chunker import chunk_text

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self):
        self._client: chromadb.ClientAPI | None = None
        self._collections: dict[str, object] = {}

    @property
    def persist_dir(self) -> Path:
        return Path(settings.rag_persist_dir)

    def _get_novel(self, novel_id: str):
        return novel_registry.require(novel_id)

    def _text_dir(self, novel_id: str) -> Path:
        return self._get_novel(novel_id).text_dir

    def _collection_name(self, novel_id: str) -> str:
        novel = self._get_novel(novel_id)
        return f"{novel.collection}_{get_collection_suffix()}"

    def _get_embedding_function(self):
        return get_embedding_function()

    def get_embedding_info(self) -> dict:
        return get_embedding_info()

    def _get_client(self):
        if self._client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def _ensure_collection(self, novel_id: str):
        if novel_id not in self._collections:
            client = self._get_client()
            name = self._collection_name(novel_id)
            print(f"[RAG] 加载 Embedding 模型（首次可能需 10-30 秒）...", flush=True)
            self._collections[novel_id] = client.get_or_create_collection(
                name=name,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine", "novel_id": novel_id},
            )
            print(f"[RAG] Embedding 就绪，集合: {name}", flush=True)
        return self._collections[novel_id]

    def _count_chunks_lightweight(self, novel_id: str) -> int:
        try:
            client = self._get_client()
            name = self._collection_name(novel_id)
            for col in client.list_collections():
                if col.name == name:
                    return client.get_collection(name=name).count()
        except Exception as e:
            logger.warning("轻量读取索引数量失败 (%s): %s", novel_id, e)
        return 0

    def _all_character_names(self, novel_id: str) -> list[str]:
        names: set[str] = set()
        for profile in character_service.list_all(novel_id=novel_id):
            full = character_service.get(profile.id, novel_id=novel_id)
            if full:
                names.add(full.name)
                names.update(full.aliases)
        return sorted(names, key=len, reverse=True)

    def _resolve_character_names(
        self, novel_id: str, character_ids: list[str]
    ) -> list[str]:
        names: set[str] = set()
        for cid in character_ids:
            profile = character_service.get(cid, novel_id=novel_id) or character_service.get_by_name(
                cid, novel_id=novel_id
            )
            if profile:
                names.add(profile.name)
                names.update(profile.aliases)
            else:
                names.add(cid)
        return list(names)

    def get_status(self, novel_id: str) -> RagStatusResponse:
        novel = self._get_novel(novel_id)
        text_dir = self._text_dir(novel_id)
        text_files = sorted(
            str(f.name) for f in text_dir.glob("*.txt") if f.is_file()
        ) if text_dir.exists() else []

        total_chunks = 0
        if settings.rag_enabled:
            try:
                if novel_id in self._collections:
                    total_chunks = self._collections[novel_id].count()
                else:
                    total_chunks = self._count_chunks_lightweight(novel_id)
            except Exception as e:
                logger.warning("读取 RAG 索引状态失败 (%s): %s", novel_id, e)

        embedding = self.get_embedding_info()
        hints: list[str] = []

        if embedding.get("note"):
            hints.append(embedding["note"])

        if not text_files:
            hints.append(
                f"请将《{novel.title}》全文 txt 放入 {text_dir}，然后运行: python scripts/setup_rag.py --novel {novel_id}"
            )
        elif settings.embedding_provider.lower() == "doubao" and not settings.use_api_embedding:
            hints.append(
                "豆包 Embedding 需配置 EMBEDDING_API_KEY 和 EMBEDDING_MODEL（火山方舟接入点 ep-xxx）"
            )

        return RagStatusResponse(
            enabled=settings.rag_enabled,
            novel_id=novel_id,
            novel_title=novel.title,
            collection_name=self._collection_name(novel_id),
            total_chunks=total_chunks,
            text_dir=str(text_dir),
            text_files=text_files,
            persist_dir=str(self.persist_dir),
            embedding_mode=embedding["mode"],
            embedding_provider=embedding.get("provider", settings.embedding_provider),
            embedding_model=embedding["model"],
            embedding_ready=embedding["ready"],
            hint=" ".join(hints),
        )

    def ingest(self, novel_id: str, force: bool = False) -> RagIngestResponse:
        novel = self._get_novel(novel_id)
        text_dir = self._text_dir(novel_id)

        if not text_dir.exists():
            text_dir.mkdir(parents=True, exist_ok=True)
            return RagIngestResponse(
                success=False,
                novel_id=novel_id,
                files_processed=0,
                chunks_indexed=0,
                message=f"文本目录不存在，已创建: {text_dir}。请将原著 txt 文件放入该目录后重试。",
            )

        text_files = list(text_dir.glob("*.txt"))
        if not text_files:
            return RagIngestResponse(
                success=False,
                novel_id=novel_id,
                files_processed=0,
                chunks_indexed=0,
                message=f"未找到 txt 文件。请将《{novel.title}》原著文本放入: {text_dir}",
            )

        collection = self._ensure_collection(novel_id)
        if force and collection.count() > 0:
            name = self._collection_name(novel_id)
            self._client.delete_collection(name)
            self._collections[novel_id] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine", "novel_id": novel_id},
            )
            collection = self._collections[novel_id]

        character_names = self._all_character_names(novel_id)
        total_chunks = 0

        for file_path in text_files:
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(
                text=text,
                source_file=file_path.name,
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
                character_names=character_names,
            )

            if not chunks:
                continue

            batch_size = 64
            file_chunks = len(chunks)
            for start in range(0, file_chunks, batch_size):
                end = min(start + batch_size, file_chunks)
                batch = chunks[start:end]
                ids = []
                documents = []
                metadatas = []

                for chunk in batch:
                    chunk_id = hashlib.md5(
                        f"{novel_id}:{chunk.source_file}:{chunk.chapter}:{chunk.chunk_index}:{chunk.content[:50]}".encode()
                    ).hexdigest()
                    ids.append(chunk_id)
                    documents.append(chunk.content)
                    metadatas.append(
                        {
                            "novel_id": novel_id,
                            "source_file": chunk.source_file,
                            "chapter": chunk.chapter,
                            "chapter_title": chunk.chapter_title,
                            "characters": ",".join(chunk.characters),
                            "chunk_index": chunk.chunk_index,
                        }
                    )

                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_chunks += len(batch)
                print(
                    f"  入库进度 [{novel.title}]: {file_path.name} {end}/{file_chunks} ({100 * end // file_chunks}%)",
                    flush=True,
                )

            logger.info("已入库 %s (%s): %d 块", file_path.name, novel_id, file_chunks)

        return RagIngestResponse(
            success=True,
            novel_id=novel_id,
            files_processed=len(text_files),
            chunks_indexed=total_chunks,
            message=f"《{novel.title}》成功入库 {len(text_files)} 个文件，共 {total_chunks} 个文本块",
        )

    def search(
        self,
        novel_id: str,
        query: str,
        character_ids: list[str] | None = None,
        time_period: str = "",
        top_k: int | None = None,
    ) -> list[CanonChunk]:
        if not settings.rag_enabled:
            return []

        collection = self._ensure_collection(novel_id)
        if collection.count() == 0:
            return []

        top_k = top_k or settings.rag_top_k
        character_names = self._resolve_character_names(novel_id, character_ids or [])

        queries = self._build_queries(query, character_names, time_period)
        seen_ids: set[str] = set()
        results: list[CanonChunk] = []

        for q in queries:
            query_params: dict = {
                "query_texts": [q],
                "n_results": top_k * 2,
                "include": ["documents", "metadatas", "distances"],
            }

            try:
                raw = collection.query(**query_params)
                self._append_results(raw, seen_ids, results, character_names)
            except Exception as e:
                logger.warning("RAG 检索失败 (%s): %s", novel_id, e)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def retrieve_for_story(
        self,
        novel_id: str,
        scenario: str,
        character_ids: list[str],
        time_period: str = "",
    ) -> list[CanonChunk]:
        return self.search(
            novel_id=novel_id,
            query=scenario,
            character_ids=character_ids,
            time_period=time_period,
            top_k=settings.rag_top_k,
        )

    def build_canon_context(self, chunks: list[CanonChunk], novel_title: str = "") -> str:
        if not chunks:
            return ""

        title_hint = f"《{novel_title}》" if novel_title else "原著"
        lines = [
            f"【{title_hint}参考片段（仅供把握人物与氛围）】",
            f"以下片段用于理解人物言行与后续起点的上下文，请据此写出原著之后的后续剧情。",
            "注意：参考风格和细节，不要大段照搬原文，不要写成夹在原著中间的补篇。",
            "",
        ]

        for i, chunk in enumerate(chunks, 1):
            header_parts = [f"片段{i}"]
            if chunk.chapter:
                header_parts.append(chunk.chapter)
            if chunk.chapter_title:
                header_parts.append(chunk.chapter_title)
            if chunk.source_file:
                header_parts.append(f"（{chunk.source_file}）")
            if chunk.characters:
                header_parts.append(f"[涉及: {', '.join(chunk.characters)}]")

            lines.append(f"--- {' · '.join(header_parts)} ---")
            lines.append(chunk.content)
            lines.append("")

        return "\n".join(lines)

    def _build_queries(
        self, scenario: str, character_names: list[str], time_period: str
    ) -> list[str]:
        queries = []
        base = scenario.strip()
        if base:
            queries.append(base)

        if character_names:
            queries.append(f"{' '.join(character_names)} {base}")
            for name in character_names[:3]:
                queries.append(f"{name} {base}")

        if time_period:
            queries.append(f"{time_period} {' '.join(character_names)} {base}")

        seen = set()
        unique = []
        for q in queries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def _append_results(
        self,
        raw: dict,
        seen_ids: set[str],
        results: list[CanonChunk],
        character_names: list[str] | None = None,
    ):
        if not raw or not raw.get("ids"):
            return

        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            meta = metadatas[idx] or {}
            distance = distances[idx] if idx < len(distances) else 1.0
            score = max(0.0, 1.0 - distance)

            chars = meta.get("characters", "")
            char_list = [c for c in chars.split(",") if c]

            if character_names:
                overlap = set(character_names) & set(char_list)
                if overlap:
                    score = min(1.0, score + 0.1 * len(overlap))
                for name in character_names:
                    if name in documents[idx]:
                        score = min(1.0, score + 0.05)

            results.append(
                CanonChunk(
                    content=documents[idx],
                    source_file=meta.get("source_file", ""),
                    chapter=meta.get("chapter", ""),
                    chapter_title=meta.get("chapter_title", ""),
                    characters=char_list,
                    score=round(score, 4),
                    chunk_index=meta.get("chunk_index", 0),
                )
            )


rag_service = RagService()
