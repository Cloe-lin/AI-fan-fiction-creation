from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.models.character import CharacterProfile, CharacterSummary
from app.models.rag import (
    RagIngestResponse,
    RagSearchRequest,
    RagSearchResponse,
    RagStatusResponse,
)
from app.models.story import (
    StoryContinueRequest,
    StoryCreateRequest,
    StoryCreateResponse,
    StorySeries,
    StorySeriesSummary,
)
from app.services.character_service import character_service
from app.services.novel_registry import novel_registry
from app.services.rag_service import rag_service
from app.services.story_generator import story_generator
from app.services.story_store import story_store


class NovelSummary(BaseModel):
    id: str
    title: str
    default_after_ending: bool = True
    ending_summary: str = ""
    ending_state: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.rag_enabled:
        for novel in novel_registry.list_all():
            status = rag_service.get_status(novel.id)
            print(
                f"[RAG] {novel.title}: 索引块数={status.total_chunks} 集合={status.collection_name}",
                flush=True,
            )
    yield


app = FastAPI(
    title="AI 同人文平台",
    description="基于原著人物深度档案的后续同人文创作平台：默认接结局、可存档连载、记住原著与用户走向",
    version="0.4.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    novels = []
    for novel in novel_registry.list_all():
        status = rag_service.get_status(novel.id)
        novels.append(
            {
                "id": novel.id,
                "title": novel.title,
                "total_chunks": status.total_chunks,
            }
        )
    return {
        "status": "ok",
        "service": "ai-tongrenwen",
        "rag_enabled": settings.rag_enabled,
        "novels": novels,
    }


@app.get("/api/novels", response_model=list[NovelSummary])
async def list_novels():
    return [
        NovelSummary(
            id=n.id,
            title=n.title,
            default_after_ending=n.default_after_ending,
            ending_summary=n.ending_summary,
            ending_state=n.ending_state,
        )
        for n in novel_registry.list_all()
    ]


@app.get("/api/characters", response_model=list[CharacterSummary])
async def list_characters(novel_id: str = Query(..., description="作品 ID")):
    if not novel_registry.get(novel_id):
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    return character_service.list_all(novel_id=novel_id)


@app.get("/api/characters/{character_id}", response_model=CharacterProfile)
async def get_character(
    character_id: str,
    novel_id: str = Query(..., description="作品 ID"),
):
    if not novel_registry.get(novel_id):
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    profile = character_service.get(character_id, novel_id=novel_id)
    if not profile:
        profile = character_service.get_by_name(character_id, novel_id=novel_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"人物 '{character_id}' 不存在")
    return profile


@app.post("/api/stories/create", response_model=StoryCreateResponse)
async def create_story(request: StoryCreateRequest):
    if not novel_registry.get(request.novel_id):
        raise HTTPException(status_code=404, detail=f"作品 '{request.novel_id}' 不存在")
    try:
        return await story_generator.generate(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创作失败: {str(e)}")


@app.get("/api/stories", response_model=list[StorySeriesSummary])
async def list_stories(novel_id: str = Query(..., description="作品 ID")):
    if not novel_registry.get(novel_id):
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    return story_store.list_by_novel(novel_id)


@app.get("/api/stories/{series_id}", response_model=StorySeries)
async def get_story(series_id: str):
    series = story_store.get(series_id)
    if not series:
        raise HTTPException(status_code=404, detail=f"存档 '{series_id}' 不存在")
    return series


@app.post("/api/stories/{series_id}/continue", response_model=StoryCreateResponse)
async def continue_story(series_id: str, request: StoryContinueRequest):
    series = story_store.get(series_id)
    if not series:
        raise HTTPException(status_code=404, detail=f"存档 '{series_id}' 不存在")

    create_req = StoryCreateRequest(
        novel_id=series.novel_id,
        title=request.title or f"{series.title} · 第{series.chapter_count + 1}章",
        characters=series.characters,
        scenario=request.scenario,
        tone=series.tone,
        perspective=series.perspective,
        length=request.length or series.length,
        additional_notes=request.additional_notes,
        series_id=series.id,
        auto_save=True,
    )
    try:
        return await story_generator.generate(create_req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"续写失败: {str(e)}")


@app.delete("/api/stories/{series_id}")
async def delete_story(series_id: str):
    if not story_store.delete(series_id):
        raise HTTPException(status_code=404, detail=f"存档 '{series_id}' 不存在")
    return {"success": True, "id": series_id}


@app.get("/api/rag/status", response_model=RagStatusResponse)
async def rag_status(novel_id: str = Query(..., description="作品 ID")):
    if not novel_registry.get(novel_id):
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    return rag_service.get_status(novel_id)


@app.post("/api/rag/ingest", response_model=RagIngestResponse)
async def rag_ingest(
    novel_id: str = Query(..., description="作品 ID"),
    force: bool = Query(False, description="是否清空重建索引"),
):
    if not novel_registry.get(novel_id):
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    try:
        return rag_service.ingest(novel_id=novel_id, force=force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败: {str(e)}")


@app.post("/api/rag/search", response_model=RagSearchResponse)
async def rag_search(request: RagSearchRequest):
    if not novel_registry.get(request.novel_id):
        raise HTTPException(status_code=404, detail=f"作品 '{request.novel_id}' 不存在")
    status = rag_service.get_status(request.novel_id)
    results = rag_service.search(
        novel_id=request.novel_id,
        query=request.query,
        character_ids=request.characters,
        time_period=request.time_period,
        top_k=request.top_k,
    )
    return RagSearchResponse(
        query=request.query,
        results=results,
        total_in_index=status.total_chunks,
    )
