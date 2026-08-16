from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class StoryTone(str, Enum):
    WARM = "warm"
    TRAGIC = "tragic"
    HUMOROUS = "humorous"
    TENSE = "tense"
    ROMANTIC = "romantic"
    REFLECTIVE = "reflective"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsistencyIssue(BaseModel):
    character: str
    issue_type: str = Field(
        ...,
        description="问题类型：ooc / canon_violation / tone_mismatch / mid_chapter_fill",
    )
    description: str
    severity: str = Field(default="warning", description="error / warning / info")
    suggestion: str = Field(default="", description="修改建议")


class ConsistencyReport(BaseModel):
    passed: bool
    score: float = Field(..., ge=0, le=100, description="一致性评分 0-100")
    issues: list[ConsistencyIssue] = Field(default_factory=list)
    summary: str = ""


class StoryChapter(BaseModel):
    index: int
    title: str = ""
    scenario: str = ""
    content: str
    summary: str = ""
    plot_state: str = Field(default="", description="本章结束后的剧情现状")
    open_threads: list[str] = Field(default_factory=list, description="未收束的线索")
    user_intent: str = Field(default="", description="本章体现的用户意图")
    consistency_report: Optional[ConsistencyReport] = None
    created_at: str = Field(default_factory=_now)


class StorySeries(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    novel_id: str
    novel_title: str = ""
    title: str
    characters: list[str] = Field(default_factory=list)
    character_names: list[str] = Field(default_factory=list)
    tone: StoryTone = StoryTone.WARM
    perspective: str = ""
    length: str = "medium"
    plot_direction: str = Field(
        default="",
        description="用户整体剧情走向（随续写累积）",
    )
    chapters: list[StoryChapter] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @property
    def chapter_count(self) -> int:
        return len(self.chapters)


class StorySeriesSummary(BaseModel):
    id: str
    novel_id: str
    novel_title: str
    title: str
    characters: list[str]
    character_names: list[str]
    chapter_count: int
    plot_direction: str
    updated_at: str
    last_summary: str = ""


class StoryCreateRequest(BaseModel):
    novel_id: str = Field(..., description="作品 ID")
    title: str = Field(..., description="故事标题")
    characters: list[str] = Field(..., min_length=1, description="参与人物 ID 列表")
    scenario: str = Field(..., description="本章后续剧情设定")
    tone: StoryTone = Field(default=StoryTone.WARM)
    perspective: str = Field(default="")
    length: str = Field(default="medium")
    additional_notes: str = Field(default="")
    series_id: str = Field(default="", description="已有存档 ID；空则新建")
    auto_save: bool = Field(default=True, description="是否自动保存/追加到存档")


class StoryCreateResponse(BaseModel):
    title: str
    content: str
    novel_id: str = ""
    novel_title: str = ""
    characters_used: list[str]
    consistency_report: ConsistencyReport
    canon_references: list[dict] = Field(default_factory=list)
    generation_metadata: dict = Field(default_factory=dict)
    series_id: str = ""
    chapter_index: int = 0
    chapter_summary: str = ""
    plot_direction: str = ""
    saved: bool = False


class StoryContinueRequest(BaseModel):
    scenario: str = Field(..., description="下一章剧情走向/设定")
    title: str = Field(default="", description="本章标题，空则自动生成")
    additional_notes: str = Field(default="")
    length: str = Field(default="", description="空则沿用系列设定")
