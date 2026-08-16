from pathlib import Path

import yaml

from app.config import settings
from app.models.character import CharacterProfile, CharacterSummary
from app.services.novel_registry import novel_registry


class CharacterService:
    def __init__(self):
        self._profiles: dict[str, CharacterProfile] = {}
        self._novel_by_id: dict[str, str] = {}
        self._load_all()

    def _load_all(self):
        data_dir = Path(settings.data_dir)
        for yaml_file in data_dir.rglob("characters/*.yaml"):
            novel_id = yaml_file.parent.parent.name
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            profile = CharacterProfile(**data)
            self._profiles[profile.id] = profile
            self._novel_by_id[profile.id] = novel_id

    def get_novel_id(self, character_id: str) -> str | None:
        return self._novel_by_id.get(character_id)

    def get(self, character_id: str, novel_id: str | None = None) -> CharacterProfile | None:
        profile = self._profiles.get(character_id)
        if not profile:
            return None
        if novel_id and self._novel_by_id.get(character_id) != novel_id:
            return None
        return profile

    def get_by_name(self, name: str, novel_id: str | None = None) -> CharacterProfile | None:
        for profile in self._profiles.values():
            if novel_id and self._novel_by_id.get(profile.id) != novel_id:
                continue
            if profile.name == name or name in profile.aliases:
                return profile
        return None

    def list_all(self, novel_id: str | None = None) -> list[CharacterSummary]:
        result = []
        for profile in self._profiles.values():
            cid_novel = self._novel_by_id.get(profile.id, "")
            if novel_id and cid_novel != novel_id:
                continue
            result.append(
                CharacterSummary(
                    id=profile.id,
                    name=profile.name,
                    aliases=profile.aliases,
                    role=profile.role,
                    novel_id=cid_novel,
                    core_essence=profile.core_essence[:100] + "..."
                    if len(profile.core_essence) > 100
                    else profile.core_essence,
                    profile_depth=getattr(profile, "profile_depth", "deep") or "deep",
                )
            )
        return result

    def reload(self):
        self._profiles.clear()
        self._novel_by_id.clear()
        self._load_all()

    def save_profile(self, novel_id: str, profile: CharacterProfile) -> Path:
        chars_dir = Path(settings.data_dir) / novel_id / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)
        path = chars_dir / f"{profile.id}.yaml"
        data = profile.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        self._profiles[profile.id] = profile
        self._novel_by_id[profile.id] = novel_id
        return path

    def get_multiple(
        self, character_ids: list[str], novel_id: str | None = None
    ) -> list[CharacterProfile]:
        profiles = []
        for cid in character_ids:
            profile = self.get(cid, novel_id=novel_id)
            if profile:
                profiles.append(profile)
            else:
                profile = self.get_by_name(cid, novel_id=novel_id)
                if profile:
                    profiles.append(profile)
        return profiles

    def validate_same_novel(self, character_ids: list[str]) -> str:
        novel_ids = {
            self._novel_by_id.get(cid)
            for cid in character_ids
            if self._novel_by_id.get(cid)
        }
        if not novel_ids:
            raise ValueError("未找到指定人物")
        if len(novel_ids) > 1:
            raise ValueError("不能混选不同作品的人物")
        novel_id = novel_ids.pop()
        if novel_id not in {n.id for n in novel_registry.list_all()}:
            raise ValueError(f"未知作品: {novel_id}")
        return novel_id

    def build_character_context(
        self, character_ids: list[str], time_period: str = "", novel_id: str | None = None
    ) -> str:
        profiles = self.get_multiple(character_ids, novel_id=novel_id)
        if not profiles:
            return ""

        sections = []
        for profile in profiles:
            sections.append(profile.to_prompt_context())

        context = "\n\n" + "=" * 50 + "\n\n".join(sections)

        if time_period:
            context += f"\n\n【接续起点】{time_period}\n"
            context += (
                "请按原著结局之后的人物状态来写后续剧情。"
                "记住原著收束结果，不要回写或补写原著中间已发生的篇章。"
            )

        return context

    def get_relationship_context(
        self, character_ids: list[str], novel_id: str | None = None
    ) -> str:
        profiles = self.get_multiple(character_ids, novel_id=novel_id)
        if len(profiles) < 2:
            return ""

        lines = ["【人物关系互动指南】"]
        name_set = {p.name for p in profiles} | {
            alias for p in profiles for alias in p.aliases
        }

        for profile in profiles:
            for rel in profile.relationships:
                if rel.target in name_set:
                    lines.append(
                        f"- {profile.name} → {rel.target}：{rel.attitude}。"
                        f"相处模式：{rel.dynamic}。"
                        f"说话方式：{rel.speech_toward or '默认'}"
                    )

        return "\n".join(lines)


character_service = CharacterService()
