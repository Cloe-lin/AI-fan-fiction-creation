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
    # public=公共书库；private=私人书库（仅 owner 可见）
    visibility: str = "public"
    owner_id: str = ""

    @property
    def text_dir(self) -> Path:
        return Path(settings.data_dir) / self.text_subdir

    @property
    def is_private(self) -> bool:
        return (self.visibility or "public") == "private"


class NovelRegistry:
    def __init__(self):
        self._novels: dict[str, NovelConfig] = {}
        self._load()

    @property
    def registry_path(self) -> Path:
        return Path(settings.data_dir) / "novels.yaml"

    def _load(self):
        registry_path = self.registry_path
        if not registry_path.exists():
            return

        with open(registry_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for item in data.get("novels", []):
            novel = NovelConfig(**item)
            self._novels[novel.id] = novel

    def reload(self):
        self._novels.clear()
        self._load()

    def list_all(self) -> list[NovelConfig]:
        return list(self._novels.values())

    def list_visible(self, *, user_id: str = "", is_admin: bool = False) -> list[NovelConfig]:
        """公共书 + 当前用户私人书；管理员可见全部。"""
        result = []
        for novel in self._novels.values():
            if is_admin:
                result.append(novel)
                continue
            if not novel.is_private:
                result.append(novel)
                continue
            if user_id and novel.owner_id == user_id:
                result.append(novel)
        return result

    def get(self, novel_id: str) -> NovelConfig | None:
        return self._novels.get(novel_id)

    def require(self, novel_id: str) -> NovelConfig:
        novel = self.get(novel_id)
        if not novel:
            raise ValueError(f"未知作品: {novel_id}")
        return novel

    def upsert(self, novel: NovelConfig, persist: bool = True):
        self._novels[novel.id] = novel
        if persist:
            self._persist()

    def _persist(self):
        path = self.registry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "novels": [n.model_dump() for n in self._novels.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                payload,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

    def build_ending_context(self, novel_id: str) -> str:
        novel = self.require(novel_id)
        lines = [
            f"【《{novel.title}》原著结局记忆】",
            "后续创作必须接在原著结局之后，不得回写中间篇章。",
            "",
            novel.ending_summary.strip()
            if novel.ending_summary
            else "（暂无详细结局摘要，请严格按原著结局之后续写）",
        ]
        if novel.ending_state:
            lines.append("")
            lines.append("【结局时人物状态】")
            for s in novel.ending_state:
                lines.append(f"- {s}")
        return "\n".join(lines)


novel_registry = NovelRegistry()
