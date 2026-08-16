import asyncio
import json

from openai import AsyncOpenAI

from app.config import settings

# 抽取/建档请求超时（秒），避免一直挂起
LLM_TIMEOUT_SECONDS = 120


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=LLM_TIMEOUT_SECONDS,
        )

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        kwargs = {
            "model": model or settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await asyncio.wait_for(
            self.client.chat.completions.create(**kwargs),
            timeout=LLM_TIMEOUT_SECONDS + 10,
        )
        return response.choices[0].message.content or ""

    async def chat_json(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
    ) -> dict:
        content = await self.chat(
            messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        # 兼容偶发的 markdown 围栏
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)


llm_client = LLMClient()
