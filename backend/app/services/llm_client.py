import json
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from app.config import settings


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
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

        response = await self.client.chat.completions.create(**kwargs)
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
        return json.loads(content)


llm_client = LLMClient()
