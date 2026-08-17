from app.models.story import (
    ConsistencyIssue,
    ConsistencyReport,
    StoryChapter,
    StoryCreateRequest,
    StoryCreateResponse,
    StorySeries,
    StoryTone,
)
from app.services.character_service import character_service
from app.services.llm_client import llm_client
from app.services.novel_registry import novel_registry
from app.services.rag_service import rag_service
from app.services.story_store import story_store
from app.config import settings

LENGTH_MAP = {
    "short": "800-1500字",
    "medium": "1500-3000字",
    "long": "3000-5000字",
}

TONE_MAP = {
    StoryTone.WARM: "温暖治愈",
    StoryTone.TRAGIC: "悲情深沉",
    StoryTone.HUMOROUS: "轻松幽默",
    StoryTone.TENSE: "紧张悬疑",
    StoryTone.ROMANTIC: "浪漫细腻",
    StoryTone.REFLECTIVE: "沉思回味",
}

DEFAULT_TIME_PERIOD = "原著结局之后"


def build_system_prompt(novel_title: str) -> str:
    return f"""你是一位精通《{novel_title}》原著的同人文连载创作大师。你的核心使命是：

1. **绝对忠于原著**：人物性格、说话方式、行为习惯必须与原著完全一致，绝不允许 OOC
2. **原著结局记忆**：必须记住原著结局与人物收束状态，后续剧情不得与结局矛盾
3. **只写后续**：默认接在原著结局之后继续向前写；禁止补写原著中间篇章
4. **理解用户走向**：严格按用户给出的剧情设定与整体走向推进，不要擅自改写用户意图
5. **连载连贯**：若提供前文摘要/上一章内容，必须自然承接，保持人物关系与未收束线索一致
6. **深度人物塑造**：基于人物档案展现底色、内心矛盾与关系差异
7. **作品隔离**：仅基于《{novel_title}》，不得混入其他小说设定

创作要求：
- 对话符合人物说话风格
- 外在表现符合性格，可写内心活动
- 文笔流畅，有画面感
- 直接输出故事正文，不要标题、作者注或元信息"""


class StoryGenerator:
    async def generate(
        self, request: StoryCreateRequest, *, owner_id: str = ""
    ) -> StoryCreateResponse:
        novel = novel_registry.require(request.novel_id)

        series: StorySeries | None = None
        if request.series_id:
            series = story_store.get(request.series_id)
            if not series:
                raise ValueError(f"存档不存在: {request.series_id}")
            if series.novel_id != request.novel_id:
                raise ValueError("存档作品与当前作品不一致")
            if series.owner_id and owner_id and series.owner_id != owner_id:
                raise ValueError("无权续写他人存档")
            # 续写时沿用存档人物/基调
            request.characters = series.characters or request.characters
            request.tone = series.tone
            request.perspective = request.perspective or series.perspective
            request.length = request.length or series.length
            if not request.title:
                request.title = f"{series.title} · 第{series.chapter_count + 1}章"

        inferred_novel = character_service.validate_same_novel(request.characters)
        if inferred_novel != request.novel_id:
            raise ValueError(
                f"所选人物属于《{novel_registry.require(inferred_novel).title}》，与当前作品不匹配"
            )

        profiles = character_service.get_multiple(
            request.characters, novel_id=request.novel_id
        )
        if not profiles:
            raise ValueError(f"未找到指定人物: {request.characters}")

        character_context = character_service.build_character_context(
            request.characters, DEFAULT_TIME_PERIOD, novel_id=request.novel_id
        )
        relationship_context = character_service.get_relationship_context(
            request.characters, novel_id=request.novel_id
        )
        ending_context = novel_registry.build_ending_context(request.novel_id)
        memory_context = self._build_series_memory(series) if series else ""

        canon_chunks = []
        canon_context = ""
        if settings.rag_enabled:
            rag_query = request.scenario
            if series and series.plot_direction:
                rag_query = f"{series.plot_direction} {request.scenario}"
            canon_chunks = rag_service.retrieve_for_story(
                novel_id=request.novel_id,
                scenario=rag_query,
                character_ids=request.characters,
                time_period=DEFAULT_TIME_PERIOD,
            )
            canon_context = rag_service.build_canon_context(canon_chunks, novel.title)

        system_prompt = build_system_prompt(novel.title)
        user_prompt = self._build_user_prompt(
            request=request,
            novel_title=novel.title,
            character_context=character_context,
            relationship_context=relationship_context,
            ending_context=ending_context,
            memory_context=memory_context,
            canon_context=canon_context,
            chapter_index=(series.chapter_count + 1) if series else 1,
        )

        content = await llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
        )

        consistency_report = await self._check_consistency(
            content, profiles, request, novel.title
        )

        if not consistency_report.passed and consistency_report.score < 60:
            content = await self._revise(
                content,
                consistency_report,
                character_context,
                ending_context,
                memory_context,
                canon_context,
                system_prompt,
            )
            consistency_report = await self._check_consistency(
                content, profiles, request, novel.title
            )

        memory = await self._extract_chapter_memory(
            content=content,
            scenario=request.scenario,
            prior_direction=series.plot_direction if series else "",
            novel_title=novel.title,
        )

        series_id = ""
        chapter_index = 1
        plot_direction = memory.get("plot_direction", request.scenario)
        saved = False

        if request.auto_save:
            series, chapter = self._persist_chapter(
                request=request,
                novel_title=novel.title,
                profiles=profiles,
                content=content,
                consistency_report=consistency_report,
                memory=memory,
                existing=series,
                owner_id=owner_id,
            )
            series_id = series.id
            chapter_index = chapter.index
            plot_direction = series.plot_direction
            saved = True

        return StoryCreateResponse(
            title=request.title,
            content=content,
            novel_id=request.novel_id,
            novel_title=novel.title,
            characters_used=[p.name for p in profiles],
            consistency_report=consistency_report,
            canon_references=[
                {
                    "chapter": c.chapter,
                    "chapter_title": c.chapter_title,
                    "source_file": c.source_file,
                    "characters": c.characters,
                    "score": c.score,
                    "preview": c.content[:120] + "..." if len(c.content) > 120 else c.content,
                }
                for c in canon_chunks
            ],
            generation_metadata={
                "model": settings.llm_model,
                "novel_id": request.novel_id,
                "novel_title": novel.title,
                "time_period": DEFAULT_TIME_PERIOD,
                "tone": request.tone.value,
                "length": request.length,
                "rag_enabled": settings.rag_enabled,
                "canon_chunks_used": len(canon_chunks),
                "continued": bool(request.series_id),
            },
            series_id=series_id,
            chapter_index=chapter_index,
            chapter_summary=memory.get("summary", ""),
            plot_direction=plot_direction,
            saved=saved,
        )

    def _persist_chapter(
        self,
        request: StoryCreateRequest,
        novel_title: str,
        profiles,
        content: str,
        consistency_report: ConsistencyReport,
        memory: dict,
        existing: StorySeries | None,
        owner_id: str = "",
    ) -> tuple[StorySeries, StoryChapter]:
        if existing:
            series = existing
            if not series.owner_id and owner_id:
                series.owner_id = owner_id
        else:
            series = StorySeries(
                novel_id=request.novel_id,
                novel_title=novel_title,
                title=request.title,
                characters=request.characters,
                character_names=[p.name for p in profiles],
                tone=request.tone,
                perspective=request.perspective,
                length=request.length,
                plot_direction=memory.get("plot_direction", request.scenario),
                owner_id=owner_id,
            )

        chapter = StoryChapter(
            index=series.chapter_count + 1,
            title=request.title if series.chapter_count == 0 else (
                request.title if "第" in request.title else f"第{series.chapter_count + 1}章"
            ),
            scenario=request.scenario,
            content=content,
            summary=memory.get("summary", ""),
            plot_state=memory.get("plot_state", ""),
            open_threads=memory.get("open_threads", []),
            user_intent=memory.get("user_intent", request.scenario),
            consistency_report=consistency_report,
        )
        series.chapters.append(chapter)
        if memory.get("plot_direction"):
            series.plot_direction = memory["plot_direction"]
        story_store.save(series)
        return series, chapter

    def _build_series_memory(self, series: StorySeries) -> str:
        if not series or not series.chapters:
            return ""

        lines = [
            "【已写后续记忆（必须承接）】",
            f"系列标题：{series.title}",
            f"用户整体剧情走向：{series.plot_direction or '（首章建立中）'}",
            "",
            "【各章摘要】",
        ]
        for ch in series.chapters:
            lines.append(f"第{ch.index}章：{ch.summary or ch.scenario[:80]}")
            if ch.plot_state:
                lines.append(f"  现状：{ch.plot_state}")
            if ch.open_threads:
                lines.append(f"  未收束：{'；'.join(ch.open_threads)}")

        last = series.chapters[-1]
        preview = last.content[-1200:] if len(last.content) > 1200 else last.content
        lines.extend(
            [
                "",
                f"【上一章结尾（第{last.index}章，请自然续上）】",
                preview,
            ]
        )
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        request: StoryCreateRequest,
        novel_title: str,
        character_context: str,
        relationship_context: str,
        ending_context: str,
        memory_context: str,
        canon_context: str,
        chapter_index: int,
    ) -> str:
        parts = [
            f"请基于《{novel_title}》创作后续同人文第{chapter_index}章：",
            f"\n标题：{request.title}",
            f"基调：{TONE_MAP.get(request.tone, '温暖')}",
            f"篇幅：{LENGTH_MAP.get(request.length, '1500-3000字')}",
            f"接续起点：{DEFAULT_TIME_PERIOD}（硬性默认）",
        ]

        if request.perspective:
            parts.append(f"叙事视角：{request.perspective}")

        parts.append(f"\n【用户本章剧情设定——必须落实】\n{request.scenario}")

        if request.additional_notes:
            parts.append(f"\n额外要求：\n{request.additional_notes}")

        parts.append(f"\n{ending_context}")

        if memory_context:
            parts.append(f"\n{memory_context}")

        parts.append(f"\n{character_context}")

        if relationship_context:
            parts.append(f"\n{relationship_context}")

        if canon_context:
            parts.append(f"\n{canon_context}")

        parts.append(
            "\n\n【输出要求】\n"
            "1. 只输出本章正文\n"
            "2. 必须接在原著结局之后，并承接已写后续记忆\n"
            "3. 落实用户剧情设定与整体走向，不要偏离用户想要的方向\n"
            "4. 不要复述原著中间情节，不要作者注"
        )
        return "\n".join(parts)

    async def _extract_chapter_memory(
        self,
        content: str,
        scenario: str,
        prior_direction: str,
        novel_title: str,
    ) -> dict:
        prompt = f"""根据本章正文与用户设定，提取连载记忆（JSON）：

【用户设定】
{scenario}

【此前整体走向】
{prior_direction or "无（本章为开篇）"}

【本章正文】
{content[:3500]}

返回：
{{
  "summary": "80字内本章摘要",
  "plot_state": "本章结束后人物与局势现状",
  "open_threads": ["未收束线索1", "线索2"],
  "user_intent": "用户想推动的方向（一句话）",
  "plot_direction": "更新后的整体走向（综合此前走向+本章，150字内）"
}}"""

        try:
            result = await llm_client.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": f"你负责为《{novel_title}》后续连载提取剧情记忆，只返回 JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=settings.llm_review_model,
            )
            return {
                "summary": result.get("summary", "")[:200],
                "plot_state": result.get("plot_state", "")[:300],
                "open_threads": result.get("open_threads", [])[:6],
                "user_intent": result.get("user_intent", scenario)[:200],
                "plot_direction": result.get("plot_direction", scenario)[:400],
            }
        except Exception:
            return {
                "summary": scenario[:80],
                "plot_state": "",
                "open_threads": [],
                "user_intent": scenario[:120],
                "plot_direction": (prior_direction + "；" + scenario).strip("；")[:400]
                if prior_direction
                else scenario[:400],
            }

    async def _check_consistency(
        self, content: str, profiles, request: StoryCreateRequest, novel_title: str
    ) -> ConsistencyReport:
        character_summaries = "\n".join(
            f"- {p.name}: {p.core_essence[:200]}" for p in profiles
        )
        taboos = "\n".join(
            f"- {p.name}: {'; '.join(p.taboos[:3])}" for p in profiles
        )

        review_prompt = f"""请作为《{novel_title}》原著专家，审查以下「原著结局之后」的后续同人文。

【审查重点】
1. 人物是否 OOC
2. 是否与原著结局矛盾
3. 是否写成了中间篇章补白（应判 error）
4. 是否明显偏离用户设定：{request.scenario}

【参与人物】
{character_summaries}

【人物禁忌】
{taboos}

【故事内容】
{content}

返回 JSON：
{{
  "passed": true/false,
  "score": 0-100,
  "issues": [
    {{
      "character": "人物名或结构",
      "issue_type": "ooc/canon_violation/tone_mismatch/mid_chapter_fill/direction_mismatch",
      "description": "问题描述",
      "severity": "error/warning/info",
      "suggestion": "修改建议"
    }}
  ],
  "summary": "总体评价"
}}"""

        result = await llm_client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": f"你是《{novel_title}》原著专家，审查后续同人文一致性与走向。",
                },
                {"role": "user", "content": review_prompt},
            ],
            model=settings.llm_review_model,
        )

        issues = [ConsistencyIssue(**issue) for issue in result.get("issues", [])]
        return ConsistencyReport(
            passed=result.get("passed", False),
            score=result.get("score", 0),
            issues=issues,
            summary=result.get("summary", ""),
        )

    async def _revise(
        self,
        content: str,
        report: ConsistencyReport,
        character_context: str,
        ending_context: str,
        memory_context: str,
        canon_context: str,
        system_prompt: str,
    ) -> str:
        issues_text = "\n".join(
            f"- [{issue.severity}] {issue.character}: {issue.description} → {issue.suggestion}"
            for issue in report.issues
            if issue.severity in ("error", "warning")
        )

        revise_prompt = f"""以下后续同人文需修订。

【原稿】
{content}

【问题清单】
{issues_text}

{ending_context}

{memory_context}

【人物档案】
{character_context}
"""
        if canon_context:
            revise_prompt += f"\n【原著参考】\n{canon_context}\n"

        revise_prompt += (
            "\n请输出修订后的完整正文：修正 OOC/设定问题，保持接在原著结局之后，"
            "并符合用户走向；不要中间篇章补白。"
        )

        return await llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": revise_prompt},
            ],
            temperature=0.6,
        )


story_generator = StoryGenerator()
