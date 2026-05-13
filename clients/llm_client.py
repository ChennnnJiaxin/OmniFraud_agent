from __future__ import annotations

import base64
from typing import Any

from openai import OpenAI

from infra.config import AppConfig, load_app_config


class LlmClient:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_app_config()

    def create_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )

    def complete(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Any:
        client = self.create_client()
        return client.chat.completions.create(
            model=model or self.config.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def stream_chat(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Any:
        client = self.create_client()
        return client.chat.completions.create(
            model=model or self.config.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

    def verify_chat_model(self, model: str | None = None) -> None:
        self.complete(
            model=model or self.config.openai_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )

    def extract_text_from_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_base64}"
        prompt = (
            "请提取图片中的完整中文文本内容（聊天截图或邮件正文），保持原始语义和重要数字、链接、账号、金额信息。"
            "不要总结，只输出提取后的纯文本。"
        )
        response = self.complete(
            model=self.config.openai_vl_model or self.config.openai_model,
            messages=[
                {"role": "system", "content": "你是一个OCR与信息提取助手。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=1500,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join(part for part in text_parts if part).strip()
        return (content or "").strip()
