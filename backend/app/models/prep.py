from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PrepJobStatus(str, Enum):
    UPLOADED = "uploaded"
    EXTRACTING_CANDIDATES = "extracting_candidates"
    AWAITING_SELECTION = "awaiting_selection"
    GENERATING_PROFILES = "generating_profiles"
    GENERATING_ENDING = "generating_ending"
    REGISTERING = "registering"
    INGESTING_RAG = "ingesting_rag"
    COMPLETED = "completed"
    FAILED = "failed"


class CharacterCandidate(BaseModel):
    id: str = Field(..., description="建议人物 ID（英文蛇形）")
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = Field(default="", description="角色定位简述")
    importance: str = Field(default="", description="为何重要")


class PrepUploadResponse(BaseModel):
    job_id: str
    novel_id: str
    title: str
    status: PrepJobStatus
    message: str = ""
    visibility: str = "private"
    owner_id: str = ""


class PrepStartRequest(BaseModel):
    deep_character_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="勾选做深档的人物 ID，1–6 人",
    )


class PrepUpgradeRequest(BaseModel):
    character_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="要从简档升格为深档的人物 ID",
    )


class PrepJobView(BaseModel):
    job_id: str
    novel_id: str
    title: str
    status: PrepJobStatus
    progress: int = Field(0, ge=0, le=100)
    message: str = ""
    error: str = ""
    candidates: list[CharacterCandidate] = Field(default_factory=list)
    deep_character_ids: list[str] = Field(default_factory=list)
    brief_character_ids: list[str] = Field(default_factory=list)
    steps_done: list[str] = Field(default_factory=list)
    visibility: str = "private"
    owner_id: str = ""


class PrepUpgradeResponse(BaseModel):
    success: bool
    message: str
    novel_id: str
    character_ids: list[str] = Field(default_factory=list)
