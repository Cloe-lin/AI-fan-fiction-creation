"""上传小说后的准备工作流：抽候选 → 用户勾选深档 → 档案/结局/登记/RAG。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models.character import CharacterProfile
from app.models.prep import (
    CharacterCandidate,
    PrepJobStatus,
    PrepJobView,
    PrepUploadResponse,
)
from app.services.character_service import character_service
from app.services.llm_client import llm_client
from app.services.novel_registry import NovelConfig, novel_registry
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

CANDIDATE_COUNT = 6
MAX_DEEP_PROFILES = 6
JOBS_DIR_NAME = "prep_jobs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_id_from_title(title: str) -> str:
    ascii_part = re.sub(r"[^a-z0-9]+", "", title.lower())
    if len(ascii_part) >= 4:
        base = ascii_part[:24]
    else:
        base = "novel_" + hashlib.md5(title.encode("utf-8")).hexdigest()[:10]
    existing = {n.id for n in novel_registry.list_all()}
    # 也避开进行中的 job 目录
    data_dir = Path(settings.data_dir)
    if data_dir.exists():
        existing |= {p.name for p in data_dir.iterdir() if p.is_dir()}
    candidate = base
    i = 2
    while candidate in existing:
        candidate = f"{base}_{i}"
        i += 1
    return candidate


def _safe_char_id(raw: str, novel_id: str, used: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", (raw or "").lower()).strip("_")
    if not slug:
        slug = "char_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    # 避免与他书冲突：加小说前缀（若尚未包含）
    if not slug.startswith(novel_id):
        slug = f"{novel_id}_{slug}"
    base = slug[:48]
    candidate = base
    i = 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def decode_novel_bytes(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc), enc.replace("-sig", "")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


def sample_text_segments(text: str, head: int = 8000, mid: int = 4000, tail: int = 8000) -> str:
    n = len(text)
    if n <= head + mid + tail:
        return text
    mid_start = max(0, (n - mid) // 2)
    parts = [
        "【开头】\n" + text[:head],
        "【中段抽样】\n" + text[mid_start : mid_start + mid],
        "【结尾】\n" + text[-tail:],
    ]
    return "\n\n====\n\n".join(parts)


def excerpts_around_names(text: str, names: list[str], window: int = 350, max_total: int = 14000) -> str:
    hits: list[tuple[int, str]] = []
    for name in names:
        if not name or len(name) < 1:
            continue
        start = 0
        while True:
            idx = text.find(name, start)
            if idx < 0:
                break
            a = max(0, idx - window)
            b = min(len(text), idx + len(name) + window)
            hits.append((idx, text[a:b]))
            start = idx + len(name)
            if len(hits) >= 40:
                break
        if len(hits) >= 40:
            break

    if not hits:
        return sample_text_segments(text, head=8000, mid=4000, tail=8000)

    # 均匀抽样若干片段
    hits.sort(key=lambda x: x[0])
    step = max(1, len(hits) // 12)
    picked = [hits[i][1] for i in range(0, len(hits), step)][:12]
    joined = "\n\n----\n\n".join(picked)
    return joined[:max_total]


def _as_plain_text(value, default: str = "") -> str:
    """把 LLM 可能返回的 dict/list 压成可读字符串。"""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _as_plain_text(item, "")
            if text:
                parts.append(text)
        return "；".join(parts) if parts else default
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            text = _as_plain_text(v, "")
            if text:
                parts.append(f"{k}：{text}")
        return "；".join(parts) if parts else default
    return str(value)


def _sanitize_profile_dict(data: dict) -> dict:
    """尽量把 LLM 输出修正为可解析的 CharacterProfile。"""
    valid_types = {
        "family",
        "friend",
        "rival",
        "lover",
        "mentor",
        "enemy",
        "complex",
        "neutral",
    }

    # 模型偶发把字符串字段写成对象/数组，统一压平
    for str_key in (
        "id",
        "name",
        "source",
        "role",
        "core_essence",
        "inner_conflict",
        "background",
        "appearance",
        "profile_depth",
    ):
        if str_key in data:
            data[str_key] = _as_plain_text(data.get(str_key), "")

    rels = data.get("relationships") or []
    cleaned_rels = []
    for rel in rels:
        if not isinstance(rel, dict):
            continue
        rtype = str(rel.get("type") or "complex").lower()
        if rtype not in valid_types:
            rtype = "complex"
        key_events = rel.get("key_events") or []
        if isinstance(key_events, str):
            key_events = [key_events]
        elif not isinstance(key_events, list):
            key_events = [_as_plain_text(key_events)]
        else:
            key_events = [_as_plain_text(x) for x in key_events if _as_plain_text(x)]
        cleaned_rels.append(
            {
                "target": _as_plain_text(rel.get("target"), "未知"),
                "type": rtype,
                "attitude": _as_plain_text(rel.get("attitude"), "态度不明"),
                "dynamic": _as_plain_text(rel.get("dynamic"), "互动模式不明"),
                "key_events": key_events,
                "speech_toward": _as_plain_text(rel.get("speech_toward"), ""),
            }
        )
    data["relationships"] = cleaned_rels

    arcs = data.get("character_arcs") or []
    cleaned_arcs = []
    for arc in arcs:
        if not isinstance(arc, dict):
            continue
        cleaned_arcs.append(
            {
                "period": _as_plain_text(arc.get("period"), "未知阶段"),
                "state": _as_plain_text(arc.get("state"), "状态不明"),
                "trigger": _as_plain_text(arc.get("trigger"), ""),
                "change": _as_plain_text(arc.get("change"), ""),
            }
        )
    data["character_arcs"] = cleaned_arcs

    for list_key in (
        "core_values",
        "life_experiences",
        "personality_traits",
        "preferences",
        "speech_patterns",
        "behavior_patterns",
        "taboos",
        "canon_facts",
        "abilities",
        "aliases",
    ):
        val = data.get(list_key)
        if val is None:
            data[list_key] = []
        elif isinstance(val, dict):
            data[list_key] = [
                f"{k}：{_as_plain_text(v)}" for k, v in val.items() if _as_plain_text(v)
            ]
        elif not isinstance(val, list):
            data[list_key] = [_as_plain_text(val)]
        else:
            data[list_key] = [_as_plain_text(x) for x in val if _as_plain_text(x)]
    return data


def _build_profile_safe(data: dict, cand: CharacterCandidate, job: "PrepJob", depth: str) -> CharacterProfile:
    data = _sanitize_profile_dict(dict(data))
    data["id"] = cand.id
    data["name"] = cand.name
    data["source"] = job.title
    data["profile_depth"] = depth
    if not data.get("aliases"):
        data["aliases"] = list(cand.aliases or [])
    if not data.get("role"):
        data["role"] = cand.role or ("重要角色" if depth == "deep" else "配角")
    data.setdefault(
        "core_essence",
        cand.importance or (f"{cand.name}的核心性格待补全" if depth == "deep" else f"{cand.name}（简档）"),
    )
    data.setdefault("background", "（根据原文自动生成的背景摘要）" if depth == "deep" else "（简档）")
    try:
        return CharacterProfile(**data)
    except Exception as e:
        logger.warning("人物档案校验失败，使用兜底简档 (%s): %s", cand.name, e)
        return CharacterProfile(
            id=cand.id,
            name=cand.name,
            aliases=list(cand.aliases or []),
            source=job.title,
            role=cand.role or "角色",
            core_essence=_as_plain_text(
                data.get("core_essence"), cand.importance or f"{cand.name}"
            ),
            background=_as_plain_text(data.get("background"), "（自动建档兜底）"),
            appearance=_as_plain_text(data.get("appearance"), ""),
            profile_depth=depth,
            personality_traits=data.get("personality_traits")
            if isinstance(data.get("personality_traits"), list)
            else [],
        )


class PrepJob:
    def __init__(
        self,
        job_id: str,
        novel_id: str,
        title: str,
        text_path: str,
        encoding: str = "utf-8",
        visibility: str = "private",
        owner_id: str = "",
    ):
        self.job_id = job_id
        self.novel_id = novel_id
        self.title = title
        self.text_path = text_path
        self.encoding = encoding
        self.visibility = visibility if visibility in {"public", "private"} else "private"
        self.owner_id = owner_id or ""
        self.status = PrepJobStatus.UPLOADED
        self.progress = 5
        self.message = "已上传原文"
        self.error = ""
        self.candidates: list[CharacterCandidate] = []
        self.deep_character_ids: list[str] = []
        self.brief_character_ids: list[str] = []
        self.steps_done: list[str] = ["upload"]
        self.ending_summary: str = ""
        self.ending_state: list[str] = []
        self.created_at = _utc_now()
        self.updated_at = self.created_at

    def to_view(self) -> PrepJobView:
        return PrepJobView(
            job_id=self.job_id,
            novel_id=self.novel_id,
            title=self.title,
            status=self.status,
            progress=self.progress,
            message=self.message,
            error=self.error,
            candidates=self.candidates,
            deep_character_ids=self.deep_character_ids,
            brief_character_ids=self.brief_character_ids,
            steps_done=list(self.steps_done),
            visibility=self.visibility,
            owner_id=self.owner_id,
        )

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "novel_id": self.novel_id,
            "title": self.title,
            "text_path": self.text_path,
            "encoding": self.encoding,
            "visibility": self.visibility,
            "owner_id": self.owner_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "candidates": [c.model_dump() for c in self.candidates],
            "deep_character_ids": self.deep_character_ids,
            "brief_character_ids": self.brief_character_ids,
            "steps_done": self.steps_done,
            "ending_summary": self.ending_summary,
            "ending_state": self.ending_state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrepJob":
        job = cls(
            job_id=data["job_id"],
            novel_id=data["novel_id"],
            title=data["title"],
            text_path=data["text_path"],
            encoding=data.get("encoding", "utf-8"),
            visibility=data.get("visibility", "private") or "private",
            owner_id=data.get("owner_id", "") or "",
        )
        job.status = PrepJobStatus(data.get("status", PrepJobStatus.UPLOADED.value))
        job.progress = int(data.get("progress", 0))
        job.message = data.get("message", "")
        job.error = data.get("error", "")
        job.candidates = [CharacterCandidate(**c) for c in data.get("candidates", [])]
        job.deep_character_ids = list(data.get("deep_character_ids", []))
        job.brief_character_ids = list(data.get("brief_character_ids", []))
        job.steps_done = list(data.get("steps_done", []))
        job.ending_summary = data.get("ending_summary", "") or ""
        job.ending_state = list(data.get("ending_state", []) or [])
        job.created_at = data.get("created_at", _utc_now())
        job.updated_at = data.get("updated_at", job.created_at)
        return job


class PrepService:
    def __init__(self):
        self._jobs: dict[str, PrepJob] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._extract_running: set[str] = set()
        self._load_jobs()

    @property
    def jobs_dir(self) -> Path:
        path = Path(settings.data_dir) / JOBS_DIR_NAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_jobs(self):
        for path in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = PrepJob.from_dict(data)
                self._jobs[job.job_id] = job
            except Exception as e:
                logger.warning("加载 prep job 失败 %s: %s", path, e)

    def _persist(self, job: PrepJob):
        job.updated_at = _utc_now()
        path = self.jobs_dir / f"{job.job_id}.json"
        path.write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _update(
        self,
        job: PrepJob,
        *,
        status: PrepJobStatus | None = None,
        progress: int | None = None,
        message: str | None = None,
        error: str | None = None,
        step: str | None = None,
    ):
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message
        if error is not None:
            job.error = error
        if step and step not in job.steps_done:
            job.steps_done.append(step)
        self._persist(job)

    def _job_age_seconds(self, job: PrepJob) -> float:
        try:
            updated = datetime.fromisoformat(job.updated_at.replace("Z", "+00:00"))
            return max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
        except Exception:
            return 99999.0

    def _schedule_extract(self, job_id: str, force: bool = False):
        """启动/恢复候选抽取；避免重复并发。"""
        job = self.get_job(job_id)
        if not job:
            return
        if job_id in self._extract_running:
            return
        if job.status == PrepJobStatus.AWAITING_SELECTION and job.candidates and not force:
            return
        if job.status == PrepJobStatus.COMPLETED:
            return

        async def runner():
            self._extract_running.add(job_id)
            try:
                await self._extract_candidates(job_id)
            finally:
                self._extract_running.discard(job_id)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(runner())
        except RuntimeError:
            # 无事件循环时（极少）忽略，等请求触发
            pass

    def resume_interrupted_jobs(self):
        """进程重启后：恢复中断的抽取/准备任务。"""
        for job in list(self._jobs.values()):
            if job.status == PrepJobStatus.EXTRACTING_CANDIDATES:
                logger.info("恢复中断的人物抽取任务: %s", job.job_id)
                self._update(
                    job,
                    message="检测到中断，正在重新抽取候选人物…",
                    progress=12,
                )
                self._schedule_extract(job.job_id, force=True)
            elif job.status in {
                PrepJobStatus.GENERATING_PROFILES,
                PrepJobStatus.GENERATING_ENDING,
                PrepJobStatus.REGISTERING,
                PrepJobStatus.INGESTING_RAG,
            }:
                # 流水线中断：回到可选人状态，避免假进行中
                if job.candidates:
                    logger.info("准备流水线中断，回退到待勾选: %s", job.job_id)
                    self._update(
                        job,
                        status=PrepJobStatus.FAILED,
                        message="准备过程被中断，请重新勾选后点「重试准备」",
                        error="服务重启导致任务中断",
                    )
                else:
                    self._update(
                        job,
                        status=PrepJobStatus.EXTRACTING_CANDIDATES,
                        message="检测到中断，正在重新抽取…",
                    )
                    self._schedule_extract(job.job_id, force=True)

    def get_job(self, job_id: str) -> PrepJob | None:
        return self._jobs.get(job_id)

    def get_job_view(self, job_id: str) -> PrepJobView | None:
        job = self.get_job(job_id)
        if not job:
            return None
        # 抽取卡住超过 45 秒且当前未在跑 → 自动重试
        if (
            job.status == PrepJobStatus.EXTRACTING_CANDIDATES
            and job_id not in self._extract_running
            and self._job_age_seconds(job) >= 45
        ):
            self._update(job, message="抽取超时，正在自动重试…", progress=12)
            self._schedule_extract(job_id, force=True)
        return job.to_view()

    async def create_from_upload(
        self,
        filename: str,
        content: bytes,
        title: str | None = None,
        *,
        visibility: str = "private",
        owner_id: str = "",
    ) -> PrepUploadResponse:
        stem = Path(filename).stem
        novel_title = (title or "").strip() or stem
        if not novel_title:
            raise ValueError("请提供小说标题")

        vis = (visibility or "private").strip().lower()
        if vis not in {"public", "private"}:
            raise ValueError("visibility 只能是 public 或 private")
        if vis == "private" and not owner_id:
            raise ValueError("私人书库需要用户身份（X-User-Id）")

        text, encoding = decode_novel_bytes(content)
        if len(text.strip()) < 500:
            raise ValueError("文本过短，请上传完整小说 txt")

        novel_id = _slug_id_from_title(novel_title)
        text_subdir = f"{novel_id}/text"
        source_file = f"{novel_id}.txt"
        text_dir = Path(settings.data_dir) / text_subdir
        text_dir.mkdir(parents=True, exist_ok=True)
        text_path = text_dir / source_file
        text_path.write_text(text, encoding="utf-8")

        (Path(settings.data_dir) / novel_id / "characters").mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex[:12]
        job = PrepJob(
            job_id=job_id,
            novel_id=novel_id,
            title=novel_title,
            text_path=str(text_path),
            encoding=encoding,
            visibility=vis,
            owner_id=owner_id if vis == "private" else "",
        )
        self._jobs[job_id] = job
        lib_label = "私人书库" if vis == "private" else "公共书库"
        self._update(
            job,
            status=PrepJobStatus.EXTRACTING_CANDIDATES,
            progress=10,
            message=f"已保存到{lib_label}，正在抽取约 6 位候选人物（通常 15–60 秒）…",
        )

        self._schedule_extract(job_id, force=True)

        return PrepUploadResponse(
            job_id=job_id,
            novel_id=novel_id,
            title=novel_title,
            status=PrepJobStatus.EXTRACTING_CANDIDATES,
            message=f"已保存到{lib_label}，正在抽取约 6 位候选人物（通常 15–60 秒）…",
            visibility=vis,
            owner_id=job.owner_id,
        )

    async def retry_extract(self, job_id: str) -> PrepJobView:
        job = self.get_job(job_id)
        if not job:
            raise ValueError("任务不存在")
        if job.status in {
            PrepJobStatus.GENERATING_PROFILES,
            PrepJobStatus.GENERATING_ENDING,
            PrepJobStatus.REGISTERING,
            PrepJobStatus.INGESTING_RAG,
        }:
            raise ValueError("准备工作进行中，请稍候")
        job.candidates = []
        job.error = ""
        self._update(
            job,
            status=PrepJobStatus.EXTRACTING_CANDIDATES,
            progress=10,
            message="正在重新抽取候选人物…",
        )
        self._schedule_extract(job_id, force=True)
        return job.to_view()

    async def _extract_candidates(self, job_id: str):
        job = self.get_job(job_id)
        if not job:
            return
        self._update(
            job,
            status=PrepJobStatus.EXTRACTING_CANDIDATES,
            progress=15,
            message="正在从原文抽取候选人物…",
        )
        try:
            text_path = Path(job.text_path)
            if not text_path.is_file():
                # 兼容相对路径工作目录变化
                alt = Path(settings.data_dir) / job.novel_id / "text" / f"{job.novel_id}.txt"
                if alt.is_file():
                    text_path = alt
                    job.text_path = str(alt)
                else:
                    raise FileNotFoundError(f"找不到原文: {job.text_path}")

            text = text_path.read_text(encoding="utf-8")
            sample = sample_text_segments(text)
            result = await llm_client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是网文/小说分析助手。根据给定原文抽样，找出约 6 位最核心的人物。"
                            "只输出 JSON，格式："
                            '{"candidates":[{"id":"snake_case_en","name":"中文名","aliases":[],'
                            '"role":"主角/重要配角…","importance":"一句话说明"}]}'
                            f"候选人数量尽量接近 {CANDIDATE_COUNT}，不要超过 {CANDIDATE_COUNT}。"
                            "id 用英文蛇形命名（拼音或意译均可）。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"作品标题：《{job.title}》\n\n原文抽样：\n{sample}",
                    },
                ],
                temperature=0.2,
            )
            raw_list = result.get("candidates") or []
            used: set[str] = set()
            candidates: list[CharacterCandidate] = []
            for item in raw_list[:CANDIDATE_COUNT]:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                cid = _safe_char_id(str(item.get("id") or name), job.novel_id, used)
                aliases = item.get("aliases") or []
                if not isinstance(aliases, list):
                    aliases = [str(aliases)]
                candidates.append(
                    CharacterCandidate(
                        id=cid,
                        name=name,
                        aliases=[str(a) for a in aliases if a],
                        role=str(item.get("role") or "").strip(),
                        importance=str(item.get("importance") or "").strip(),
                    )
                )

            if not candidates:
                raise ValueError("未能抽取出候选人物，请检查文本是否完整")

            job.candidates = candidates
            self._update(
                job,
                status=PrepJobStatus.AWAITING_SELECTION,
                progress=30,
                message="抽取完成：请自行勾选深档人物（不预勾选；未勾选将做简档），然后点「开始准备」",
                step="candidates",
                error="",
            )
        except asyncio.TimeoutError:
            logger.exception("抽取候选人物超时")
            self._update(
                job,
                status=PrepJobStatus.FAILED,
                progress=20,
                message="抽取超时，可点「重新抽取」重试",
                error="LLM 请求超时",
            )
        except Exception as e:
            logger.exception("抽取候选人物失败")
            self._update(
                job,
                status=PrepJobStatus.FAILED,
                progress=20,
                message="抽取候选人物失败，可点「重新抽取」重试",
                error=str(e),
            )

    async def start_prep(self, job_id: str, deep_character_ids: list[str]) -> PrepJobView:
        job = self.get_job(job_id)
        if not job:
            raise ValueError("任务不存在")
        if job.status == PrepJobStatus.FAILED and not job.candidates:
            raise ValueError("候选人物抽取失败，请重新上传")
        if job.status not in {
            PrepJobStatus.AWAITING_SELECTION,
            PrepJobStatus.FAILED,
        }:
            if job.status == PrepJobStatus.COMPLETED:
                raise ValueError("该任务已完成")
            if job.status in {
                PrepJobStatus.GENERATING_PROFILES,
                PrepJobStatus.GENERATING_ENDING,
                PrepJobStatus.REGISTERING,
                PrepJobStatus.INGESTING_RAG,
                PrepJobStatus.EXTRACTING_CANDIDATES,
            }:
                raise ValueError("准备工作正在进行中，请稍候")
            raise ValueError(f"当前状态不可开始准备: {job.status.value}")

        cand_ids = {c.id for c in job.candidates}
        deep_ids = []
        for cid in deep_character_ids:
            if cid not in cand_ids:
                raise ValueError(f"未知人物: {cid}")
            if cid not in deep_ids:
                deep_ids.append(cid)
        if not deep_ids:
            raise ValueError("请至少勾选 1 位核心人物做深档")
        if len(deep_ids) > MAX_DEEP_PROFILES:
            raise ValueError(f"深档最多 {MAX_DEEP_PROFILES} 人")

        job.deep_character_ids = deep_ids
        job.brief_character_ids = [c.id for c in job.candidates if c.id not in set(deep_ids)]
        job.error = ""
        job.message = "已确认深档人选，开始生成档案…"
        self._persist(job)

        asyncio.create_task(self._run_prep_pipeline(job_id))
        return job.to_view()

    async def _run_prep_pipeline(self, job_id: str):
        job = self.get_job(job_id)
        if not job:
            return
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        if lock.locked():
            return
        async with lock:
            try:
                await self._generate_all_profiles(job)
                await self._generate_ending(job)
                await self._register_novel(job)
                await self._ingest_rag(job)
                character_service.reload()
                self._update(
                    job,
                    status=PrepJobStatus.COMPLETED,
                    progress=100,
                    message="准备完成，可以开始续写",
                    step="completed",
                )
            except Exception as e:
                logger.exception("准备工作流失败")
                self._update(
                    job,
                    status=PrepJobStatus.FAILED,
                    message="准备工作失败",
                    error=str(e),
                )

    async def _generate_all_profiles(self, job: PrepJob):
        self._update(
            job,
            status=PrepJobStatus.GENERATING_PROFILES,
            progress=40,
            message="正在生成人物档案…",
        )
        text = Path(job.text_path).read_text(encoding="utf-8")
        by_id = {c.id: c for c in job.candidates}
        total = len(job.candidates)
        done = 0

        for cid in job.deep_character_ids:
            cand = by_id[cid]
            profile = await self._llm_deep_profile(job, cand, text)
            character_service.save_profile(job.novel_id, profile)
            done += 1
            pct = 40 + int(35 * done / max(total, 1))
            self._update(
                job,
                progress=pct,
                message=f"已生成深档：{cand.name}（{done}/{total}）",
            )

        for cid in job.brief_character_ids:
            cand = by_id[cid]
            profile = await self._llm_brief_profile(job, cand, text)
            character_service.save_profile(job.novel_id, profile)
            done += 1
            pct = 40 + int(35 * done / max(total, 1))
            self._update(
                job,
                progress=pct,
                message=f"已生成简档：{cand.name}（{done}/{total}）",
            )

        self._update(job, step="profiles", message="人物档案已生成")

    async def _llm_deep_profile(
        self, job: PrepJob, cand: CharacterCandidate, text: str
    ) -> CharacterProfile:
        names = [cand.name, *cand.aliases]
        excerpts = excerpts_around_names(text, names)
        data = await llm_client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是角色档案专家。根据原文摘录，为指定人物生成【完整深度档案】JSON。"
                        "必须包含字段："
                        "id,name,aliases,source,role,core_essence,core_values,inner_conflict,"
                        "background,life_experiences,character_arcs,relationships,"
                        "personality_traits,preferences,speech_patterns,behavior_patterns,"
                        "taboos,canon_facts,appearance,abilities。"
                        "重要：core_essence、inner_conflict、background、appearance、role 必须是字符串，"
                        "禁止写成对象或嵌套 JSON；appearance 用一段话描述外貌（可含身高、穿着、气质）。"
                        "character_arcs 元素含 period,state,trigger,change；"
                        "relationships 元素含 target,type,attitude,dynamic,key_events,speech_toward；"
                        "type 只能是 family/friend/rival/lover/mentor/enemy/complex/neutral。"
                        "只输出 JSON。档案必须忠于原文，勿臆造重大剧情。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"作品：《{job.title}》\n"
                        f"人物 ID：{cand.id}\n"
                        f"姓名：{cand.name}\n"
                        f"别名：{', '.join(cand.aliases) or '无'}\n"
                        f"定位：{cand.role}\n\n"
                        f"原文摘录：\n{excerpts}"
                    ),
                },
            ],
            temperature=0.3,
        )
        return _build_profile_safe(data, cand, job, "deep")

    async def _llm_brief_profile(
        self, job: PrepJob, cand: CharacterCandidate, text: str
    ) -> CharacterProfile:
        names = [cand.name, *cand.aliases]
        excerpts = excerpts_around_names(text, names, window=250, max_total=8000)
        data = await llm_client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是角色档案助手。为指定人物生成【简档】JSON，字段："
                        "id,name,aliases,source,role,core_essence,background,"
                        "personality_traits,relationships。"
                        "core_essence、background、role 必须是字符串，禁止对象。"
                        "relationships 最多 3 条，含 target,type,attitude,dynamic；"
                        "type 只能是 family/friend/rival/lover/mentor/enemy/complex/neutral。"
                        "简档要短：core_essence 2–4 句，background 一小段。只输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"作品：《{job.title}》\n"
                        f"人物 ID：{cand.id}\n姓名：{cand.name}\n"
                        f"别名：{', '.join(cand.aliases) or '无'}\n定位：{cand.role}\n\n"
                        f"原文摘录：\n{excerpts}"
                    ),
                },
            ],
            temperature=0.3,
        )
        return _build_profile_safe(data, cand, job, "brief")

    async def _generate_ending(self, job: PrepJob):
        self._update(
            job,
            status=PrepJobStatus.GENERATING_ENDING,
            progress=80,
            message="正在提炼原著结局记忆…",
        )
        text = Path(job.text_path).read_text(encoding="utf-8")
        tail = text[-20000:] if len(text) > 20000 else text
        names = "、".join(c.name for c in job.candidates)
        result = await llm_client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是原著结局分析助手。根据小说结尾文本，提炼结局摘要与结局时人物状态。"
                        "输出 JSON：{\"ending_summary\":\"一段话\",\"ending_state\":[\"条目\",...]}。"
                        "后续创作将接在此结局之后，请写清收束结果，勿剧透式回写中间篇章细节过多。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"作品：《{job.title}》\n关注人物：{names}\n\n结尾原文：\n{tail}"
                    ),
                },
            ],
            temperature=0.2,
        )
        job.ending_summary = str(result.get("ending_summary") or "").strip()
        job.ending_state = [str(s) for s in (result.get("ending_state") or []) if s]
        self._update(job, step="ending", message="结局记忆已生成")

    async def _register_novel(self, job: PrepJob):
        self._update(
            job,
            status=PrepJobStatus.REGISTERING,
            progress=88,
            message="正在登记作品…",
        )
        novel = NovelConfig(
            id=job.novel_id,
            title=job.title,
            collection=job.novel_id,
            text_subdir=f"{job.novel_id}/text",
            source_file=f"{job.novel_id}.txt",
            novel_source="",
            source_encoding="utf-8",
            default_after_ending=True,
            ending_summary=job.ending_summary,
            ending_state=job.ending_state,
            visibility=job.visibility,
            owner_id=job.owner_id if job.visibility == "private" else "",
        )
        novel_registry.upsert(novel, persist=True)
        lib = "私人书库" if job.visibility == "private" else "公共书库"
        self._update(job, step="register", message=f"已登记到{lib}：{job.title}")

    async def _ingest_rag(self, job: PrepJob):
        self._update(
            job,
            status=PrepJobStatus.INGESTING_RAG,
            progress=92,
            message="正在建立原著向量索引（可能较慢）…",
        )
        # 确保人物已加载，便于 chunk 打角色标签
        character_service.reload()
        # ingest 是同步阻塞，放到线程避免卡住事件循环
        await asyncio.to_thread(rag_service.ingest, job.novel_id, True)
        self._update(job, step="rag", message="RAG 索引已建立")

    async def upgrade_profiles_stub(
        self, novel_id: str, character_ids: list[str]
    ) -> dict:
        """预留：将简档升格为深档（后续实现）。"""
        if not novel_registry.get(novel_id):
            raise ValueError(f"作品 '{novel_id}' 不存在")
        return {
            "success": False,
            "message": "「简档升格深档」功能已预留，稍后版本将支持从简档中补选人物生成完整档案。",
            "novel_id": novel_id,
            "character_ids": character_ids,
        }


prep_service = PrepService()
