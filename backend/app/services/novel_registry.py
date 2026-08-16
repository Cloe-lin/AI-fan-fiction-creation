from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.config import settings


class NovelConfig(BaseModel):
    id: str
    title: str
    collection: str
    text_subdir: str
    source_file: str
    novel_source: str = ""
    source_encoding: str = "utf-8"
    default_after_ending: bool = True
    ending_summary: str = ""
    ending_state: list[str] = Field(default_factory=list)

    @property
    def text_dir(self) -> Path:
        return Path(settings.data_dir) / self.text_subdir


class NovelRegistry:
    def __init__(self):
        self._novels: dict[str, NovelConfig] = {}
        self._load()

    def _load(self):
        registry_path = Path(settings.data_dir) / "novels.yaml"
        if not registry_path.exists():
            return

        with open(registry_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for item in data.get("novels", []):
            novel = NovelConfig(**item)
            self._novels[novel.id] = novel

    def list_all(self) -> list[NovelConfig]:
        return list(self._novels.values())

    def get(self, novel_id: str) -> NovelConfig | None:
        return self._novels.get(novel_id)

    def require(self, novel_id: str) -> NovelConfig:
        novel = self.get(novel_id)
        if not novel:
            raise ValueError(f"未知作品: {novel_id}")
        return novel

    def build_ending_context(self, novel_id: str) -> str:
        novel = self.require(novel_id)
        lines = [
            f"【《{novel.title}》原著结局记忆】",
            "后续创作必须接在原著结局之后，不得回写中间篇章。",
            "",
            novel.ending_summary.strip() if novel.ending_summary else "（暂无详细结局摘要，请严格按原著结局之后续写）",
        ]
        if novel.ending_state:
            lines.append("")
            lines.append("【结局时人物状态】")
            for s in novel.ending_state:
                lines.append(f"- {s}")
        return "\n".join(lines)


novel_registry = NovelRegistry()
