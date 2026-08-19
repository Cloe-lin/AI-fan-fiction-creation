from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.models.auth import LoginRequest, LoginResponse, UserPublic
from app.models.character import CharacterProfile, CharacterSummary
from app.models.prep import (
    PrepJobView,
    PrepStartRequest,
    PrepUpgradeRequest,
    PrepUpgradeResponse,
    PrepUploadResponse,
)
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
from app.services.access import (
    Actor,
    assert_can_access_job,
    assert_can_access_novel,
    assert_can_access_story,
    assert_can_edit_novel,
    get_actor,
    require_admin_user,
    require_login,
    require_uploader_dep,
)
from app.services.admin_auth import admin_token_configured
from app.services.auth_store import (
    authenticate,
    count_users,
    create_access_token,
    init_db,
    seed_preset_accounts,
)
from app.services.character_service import character_service
from app.services.novel_registry import novel_registry
from app.services.prep_service import prep_service
from app.services.rag_service import rag_service
from app.services.story_generator import story_generator
from app.services.story_store import story_store


class NovelSummary(BaseModel):
    id: str
    title: str
    default_after_ending: bool = True
    ending_summary: str = ""
    ending_state: list[str] = Field(default_factory=list)
    visibility: str = "public"
    owner_id: str = ""
    is_mine: bool = False


class AdminStatusResponse(BaseModel):
    admin_configured: bool
    message: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if count_users() == 0:
        print("[Auth] 用户库为空，自动预置 admin + user01..user20 …", flush=True)
        creds = seed_preset_accounts(force_reset_passwords=True)
        cred_path = Path(settings.data_dir) / "credentials_initial.txt"
        lines = [
            "# 自动首次预置账号（含明文密码，请妥善保管）",
            f"{'username':<12} {'password':<14} role",
            "-" * 40,
        ]
        for c in creds:
            lines.append(f"{c['username']:<12} {c['password']:<14} {c['role']}")
        cred_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[Auth] 凭据已写入 {cred_path}", flush=True)

    if settings.rag_enabled:
        for novel in novel_registry.list_all():
            status = rag_service.get_status(novel.id)
            print(
                f"[RAG] {novel.title}: 索引块数={status.total_chunks} 集合={status.collection_name}",
                flush=True,
            )
    prep_service.resume_interrupted_jobs()
    yield


app = FastAPI(
    title="AI 同人文平台",
    description="基于原著人物深度档案的后续同人文创作平台：默认接结局、可存档连载、记住原著与用户走向",
    version="0.7.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端：用 http://127.0.0.1:8010/ 打开，避免直接双击 html（file://）导致登录无响应
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
_FRONTEND_INDEX = _FRONTEND_DIR / "index.html"


@app.get("/")
async def frontend_index():
    if not _FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html 未找到")
    return FileResponse(_FRONTEND_INDEX, media_type="text/html; charset=utf-8")


if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend_static")


@app.get("/api/health")
async def health(actor: Actor = Depends(get_actor)):
    novels = []
    for novel in novel_registry.list_visible(user_id=actor.user_id, is_admin=actor.is_admin):
        status = rag_service.get_status(novel.id)
        novels.append(
            {
                "id": novel.id,
                "title": novel.title,
                "visibility": novel.visibility or "public",
                "total_chunks": status.total_chunks,
            }
        )
    return {
        "status": "ok",
        "service": "ai-tongrenwen",
        "rag_enabled": settings.rag_enabled,
        "novels": novels,
    }


@app.get("/api/admin/status", response_model=AdminStatusResponse)
async def admin_status():
    ok = admin_token_configured()
    return AdminStatusResponse(
        admin_configured=ok,
        message="已配置旧版 ADMIN_TOKEN（可选）；推荐使用 admin 账号登录"
        if ok
        else "请使用预置账号登录（admin / user01..user20）",
    )


@app.post("/api/auth/login", response_model=LoginResponse)
async def auth_login(request: LoginRequest):
    user = authenticate(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(user)
    return LoginResponse(
        access_token=token,
        user=UserPublic(
            id=user.id,
            username=user.username,
            role=user.role,
            display_name=user.display_name,
        ),
    )


@app.get("/api/auth/me", response_model=UserPublic)
async def auth_me(actor: Actor = Depends(require_login)):
    if not actor.username and actor.is_admin:
        return UserPublic(id=actor.user_id or "admin", username="admin", role="admin")
    from app.services.auth_store import get_user_by_id

    user = get_user_by_id(actor.user_id) if actor.user_id else None
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在或未登录")
    return UserPublic(
        id=user.id,
        username=user.username,
        role=user.role,
        display_name=user.display_name,
    )


@app.get("/api/novels", response_model=list[NovelSummary])
async def list_novels(actor: Actor = Depends(require_login)):
    """公共书库 + 当前用户私人书库（需登录）。"""
    novels = novel_registry.list_visible(user_id=actor.user_id, is_admin=actor.is_admin)
    return [
        NovelSummary(
            id=n.id,
            title=n.title,
            default_after_ending=n.default_after_ending,
            ending_summary=n.ending_summary,
            ending_state=n.ending_state,
            visibility=n.visibility or "public",
            owner_id=n.owner_id or "",
            is_mine=bool(
                n.is_private and actor.user_id and n.owner_id == actor.user_id
            ),
        )
        for n in novels
    ]


@app.post("/api/prep/upload", response_model=PrepUploadResponse)
async def prep_upload(
    file: UploadFile = File(..., description="小说全文 txt"),
    title: str = Form("", description="作品标题，可空则用文件名"),
    visibility: str = Form("private", description="public=公共书库，private=私人书库"),
    actor: Actor = Depends(require_uploader_dep),
):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="仅支持上传 .txt 文件")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")

    vis = (visibility or "private").strip().lower()
    if vis == "public" and not actor.is_admin:
        raise HTTPException(status_code=403, detail="只有管理员可以上传到公共书库")
    if vis == "private" and not actor.user_id and not actor.is_admin:
        raise HTTPException(status_code=401, detail="私人书库需要书库钥匙（X-User-Id）")
    # 管理员也可上传私人书：需提供 user_id；否则默认公共
    if actor.is_admin and vis == "private" and not actor.user_id:
        raise HTTPException(
            status_code=400,
            detail="管理员上传私人书库时请同时提供 X-User-Id，或改选公共书库",
        )
    owner_id = actor.user_id if vis == "private" else ""

    try:
        return await prep_service.create_from_upload(
            filename=file.filename,
            content=content,
            title=title or None,
            visibility=vis,
            owner_id=owner_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


def _prep_job_or_404(job_id: str, actor: Actor):
    job = prep_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 '{job_id}' 不存在")
    assert_can_access_job(job, actor)
    return job


@app.get("/api/prep/jobs/{job_id}", response_model=PrepJobView)
async def prep_job_status(job_id: str, actor: Actor = Depends(require_uploader_dep)):
    _prep_job_or_404(job_id, actor)
    view = prep_service.get_job_view(job_id)
    if not view:
        raise HTTPException(status_code=404, detail=f"任务 '{job_id}' 不存在")
    return view


@app.post("/api/prep/jobs/{job_id}/retry-extract", response_model=PrepJobView)
async def prep_job_retry_extract(
    job_id: str, actor: Actor = Depends(require_uploader_dep)
):
    _prep_job_or_404(job_id, actor)
    try:
        return await prep_service.retry_extract(job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新抽取失败: {e}")


@app.post("/api/prep/jobs/{job_id}/start", response_model=PrepJobView)
async def prep_job_start(
    job_id: str,
    request: PrepStartRequest,
    actor: Actor = Depends(require_uploader_dep),
):
    _prep_job_or_404(job_id, actor)
    try:
        return await prep_service.start_prep(job_id, request.deep_character_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动准备失败: {e}")


@app.post(
    "/api/prep/novels/{novel_id}/upgrade-profiles",
    response_model=PrepUpgradeResponse,
)
async def prep_upgrade_profiles(
    novel_id: str,
    request: PrepUpgradeRequest,
    actor: Actor = Depends(require_uploader_dep),
):
    """预留：将简档升格为深档。"""
    novel = novel_registry.get(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    assert_can_access_novel(novel, actor)
    try:
        data = await prep_service.upgrade_profiles_stub(novel_id, request.character_ids)
        return PrepUpgradeResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/characters", response_model=list[CharacterSummary])
async def list_characters(
    novel_id: str = Query(..., description="作品 ID"),
    actor: Actor = Depends(require_login),
):
    novel = novel_registry.get(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    assert_can_access_novel(novel, actor)
    return character_service.list_all(novel_id=novel_id)


@app.get("/api/characters/{character_id}", response_model=CharacterProfile)
async def get_character(
    character_id: str,
    novel_id: str = Query(..., description="作品 ID"),
    actor: Actor = Depends(require_login),
):
    novel = novel_registry.get(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    assert_can_access_novel(novel, actor)
    profile = character_service.get(character_id, novel_id=novel_id)
    if not profile:
        profile = character_service.get_by_name(character_id, novel_id=novel_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"人物 '{character_id}' 不存在")
    return profile


@app.put("/api/characters/{character_id}", response_model=CharacterProfile)
async def update_character(
    character_id: str,
    body: CharacterProfile,
    novel_id: str = Query(..., description="作品 ID"),
    actor: Actor = Depends(require_login),
):
    """人工编辑人物深档/简档（写入 YAML，立即影响后续创作注入）。"""
    novel = novel_registry.get(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    assert_can_edit_novel(novel, actor)

    existing = character_service.get(character_id, novel_id=novel_id)
    if not existing:
        existing = character_service.get_by_name(character_id, novel_id=novel_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"人物 '{character_id}' 不存在")

    # 锁定身份字段，避免误改导致引用断裂
    data = body.model_dump(mode="json")
    data["id"] = existing.id
    if not (data.get("source") or "").strip():
        data["source"] = existing.source or novel.title
    depth = (data.get("profile_depth") or existing.profile_depth or "deep").strip().lower()
    if depth not in {"deep", "brief"}:
        depth = "deep"
    data["profile_depth"] = depth

    try:
        updated = CharacterProfile(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"档案格式无效: {e}") from e

    character_service.save_profile(novel_id, updated)
    return updated


@app.post("/api/stories/create", response_model=StoryCreateResponse)
async def create_story(
    request: StoryCreateRequest,
    actor: Actor = Depends(require_login),
):
    novel = novel_registry.get(request.novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail=f"作品 '{request.novel_id}' 不存在")
    assert_can_access_novel(novel, actor)
    if request.series_id:
        existing = story_store.get(request.series_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"存档 '{request.series_id}' 不存在")
        assert_can_access_story(existing, actor)
    try:
        return await story_generator.generate(request, owner_id=actor.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创作失败: {str(e)}")


@app.get("/api/stories", response_model=list[StorySeriesSummary])
async def list_stories(
    novel_id: str = Query(..., description="作品 ID"),
    actor: Actor = Depends(require_login),
):
    novel = novel_registry.get(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    assert_can_access_novel(novel, actor)
    return story_store.list_by_novel(
        novel_id,
        owner_id=actor.user_id,
        is_admin=actor.is_admin,
    )


@app.get("/api/stories/{series_id}", response_model=StorySeries)
async def get_story(series_id: str, actor: Actor = Depends(require_login)):
    series = story_store.get(series_id)
    if not series:
        raise HTTPException(status_code=404, detail=f"存档 '{series_id}' 不存在")
    novel = novel_registry.get(series.novel_id)
    if novel:
        assert_can_access_novel(novel, actor)
    assert_can_access_story(series, actor)
    return series


@app.post("/api/stories/{series_id}/continue", response_model=StoryCreateResponse)
async def continue_story(
    series_id: str,
    request: StoryContinueRequest,
    actor: Actor = Depends(require_login),
):
    series = story_store.get(series_id)
    if not series:
        raise HTTPException(status_code=404, detail=f"存档 '{series_id}' 不存在")
    novel = novel_registry.get(series.novel_id)
    if novel:
        assert_can_access_novel(novel, actor)
    assert_can_access_story(series, actor)

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
        return await story_generator.generate(create_req, owner_id=actor.user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"续写失败: {str(e)}")


@app.delete("/api/stories/{series_id}")
async def delete_story(series_id: str, actor: Actor = Depends(require_login)):
    series = story_store.get(series_id)
    if not series:
        raise HTTPException(status_code=404, detail=f"存档 '{series_id}' 不存在")
    novel = novel_registry.get(series.novel_id)
    if novel:
        assert_can_access_novel(novel, actor)
    assert_can_access_story(series, actor)
    if not story_store.delete(series_id):
        raise HTTPException(status_code=404, detail=f"存档 '{series_id}' 不存在")
    return {"success": True, "id": series_id}


@app.get("/api/rag/status", response_model=RagStatusResponse)
async def rag_status(
    novel_id: str = Query(..., description="作品 ID"),
    actor: Actor = Depends(require_login),
):
    novel = novel_registry.get(novel_id)
    if not novel:
        raise HTTPException(status_code=404, detail=f"作品 '{novel_id}' 不存在")
    assert_can_access_novel(novel, actor)
    return rag_service.get_status(novel_id)


@app.post("/api/rag/ingest", response_model=RagIngestResponse)
async def rag_ingest(
    novel_id: str = Query(..., description="作品 ID"),
    force: bool = Query(False, description="是否清空重建索引"),
    _: bool = Depends(require_admin_user),
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
