from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    FAMILY = "family"
    FRIEND = "friend"
    RIVAL = "rival"
    LOVER = "lover"
    MENTOR = "mentor"
    ENEMY = "enemy"
    COMPLEX = "complex"
    NEUTRAL = "neutral"


class CharacterRelationship(BaseModel):
    target: str = Field(..., description="关系对象姓名")
    type: RelationshipType
    attitude: str = Field(..., description="对此人的态度与情感定位")
    dynamic: str = Field(..., description="互动模式与相处方式")
    key_events: list[str] = Field(default_factory=list, description="影响关系的关键事件")
    speech_toward: str = Field(default="", description="对此人说话时的特殊语气/称呼")


class CharacterArc(BaseModel):
    period: str = Field(..., description="人生阶段，如'少年期'、'乱葬岗时期'")
    state: str = Field(..., description="该阶段的心理状态与外在表现")
    trigger: str = Field(default="", description="导致此状态的关键事件")
    change: str = Field(default="", description="相比上一阶段的深刻变化")


class CharacterProfile(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    source: str = Field(..., description="原著名称")
    role: str = Field(..., description="在故事中的定位")

    # 人物底色
    core_essence: str = Field(..., description="人物底色：最本质的性格内核")
    core_values: list[str] = Field(default_factory=list, description="核心价值观")
    inner_conflict: str = Field(default="", description="内心矛盾与挣扎")

    # 经历与变化
    background: str = Field(..., description="出身与早年经历")
    life_experiences: list[str] = Field(default_factory=list, description="重要人生经历")
    character_arcs: list[CharacterArc] = Field(default_factory=list, description="人物弧光变化")

    # 人际关系
    relationships: list[CharacterRelationship] = Field(default_factory=list)

    # 言行举止
    personality_traits: list[str] = Field(default_factory=list, description="性格特征")
    preferences: list[str] = Field(default_factory=list, description="喜好与习惯")
    speech_patterns: list[str] = Field(default_factory=list, description="说话方式与口头禅")
    behavior_patterns: list[str] = Field(default_factory=list, description="行为习惯与标志性动作")
    taboos: list[str] = Field(default_factory=list, description="绝对不会做的事/说的话")

    # 原著约束
    canon_facts: list[str] = Field(default_factory=list, description="不可违背的原著事实")
    appearance: str = Field(default="", description="外貌特征")
    abilities: list[str] = Field(default_factory=list, description="能力与特长")

    # deep=完整档案可作主写对象；brief=简档仅适合配角
    profile_depth: str = Field(default="deep", description="deep | brief")

    def to_prompt_context(self) -> str:
        """将人物档案转为 LLM 可用的上下文文本"""
        lines = [
            f"【{self.name}】（{self.role}）",
            f"别名：{', '.join(self.aliases) if self.aliases else '无'}",
            f"\n■ 人物底色\n{self.core_essence}",
            f"核心价值观：{'、'.join(self.core_values)}",
        ]
        if self.inner_conflict:
            lines.append(f"内心矛盾：{self.inner_conflict}")

        lines.append(f"\n■ 出身与经历\n{self.background}")
        if self.life_experiences:
            lines.append("重要经历：")
            for exp in self.life_experiences:
                lines.append(f"  - {exp}")

        if self.character_arcs:
            lines.append("\n■ 人物弧光变化")
            for arc in self.character_arcs:
                lines.append(f"  [{arc.period}] {arc.state}")
                if arc.trigger:
                    lines.append(f"    触发：{arc.trigger}")
                if arc.change:
                    lines.append(f"    变化：{arc.change}")

        if self.relationships:
            lines.append("\n■ 人际关系")
            for rel in self.relationships:
                lines.append(f"  → {rel.target}（{rel.type.value}）")
                lines.append(f"    态度：{rel.attitude}")
                lines.append(f"    相处：{rel.dynamic}")
                if rel.speech_toward:
                    lines.append(f"    对其说话：{rel.speech_toward}")

        lines.append("\n■ 性格与言行")
        lines.append(f"性格：{'、'.join(self.personality_traits)}")
        if self.preferences:
            lines.append(f"喜好：{'、'.join(self.preferences)}")
        if self.speech_patterns:
            lines.append("说话方式：")
            for sp in self.speech_patterns:
                lines.append(f"  - {sp}")
        if self.behavior_patterns:
            lines.append("行为习惯：")
            for bp in self.behavior_patterns:
                lines.append(f"  - {bp}")

        if self.taboos:
            lines.append("\n■ 绝对禁忌（创作中不可违反）")
            for taboo in self.taboos:
                lines.append(f"  ✗ {taboo}")

        if self.canon_facts:
            lines.append("\n■ 原著铁律")
            for fact in self.canon_facts:
                lines.append(f"  • {fact}")

        return "\n".join(lines)


class CharacterSummary(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str
    novel_id: str = ""
    core_essence: str
    profile_depth: str = "deep"
