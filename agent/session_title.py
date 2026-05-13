from __future__ import annotations

import re
from typing import Iterable

from agent.session_models import AgentMessage
from clients.llm_client import LlmClient

TITLE_MIN_LENGTH = 8
TITLE_MAX_LENGTH = 18

_PUNCTUATION_PATTERN = re.compile(r"[`~!@#$%^&*()_\-+=\[\]{}\\|;:'\",.<>/?，。！？；：、“”‘’（）【】《》·…\s]+")


def is_default_session_title(title: str | None, default_title: str) -> bool:
    normalized = (title or "").strip()
    return not normalized or normalized == default_title


def fallback_session_title(messages: Iterable[AgentMessage], default_title: str) -> str:
    for message in messages:
        if message.role != "user":
            continue
        title = _normalize_title(message.content)
        if title:
            return title
    return default_title


def generate_session_title(
    messages: list[AgentMessage],
    *,
    default_title: str,
    llm_client: LlmClient | None = None,
) -> str:
    fallback_title = fallback_session_title(messages, default_title)
    user_message = next((message.content.strip() for message in messages if message.role == "user" and message.content.strip()), "")
    assistant_message = next(
        (message.content.strip() for message in messages if message.role == "assistant" and message.content.strip()),
        "",
    )
    if not user_message or not assistant_message:
        return fallback_title

    llm_client = llm_client or LlmClient()
    prompt = (
        "请根据下面这段反诈会话，生成一个简短中文标题。\n"
        "要求：\n"
        "1. 只输出标题本身\n"
        "2. 8到18个汉字\n"
        "3. 不要标点\n"
        "4. 不要空泛词，例如 会话 对话 问题 咨询\n"
        "5. 更像摘要式短标题\n\n"
        f"用户：{user_message[:160]}\n"
        f"助手：{assistant_message[:200]}"
    )
    try:
        response = llm_client.complete(
            model=None,
            messages=[
                {"role": "system", "content": "你是一个擅长提炼中文会话标题的助手。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=32,
            temperature=0.2,
        )
        content = _extract_content_text(response)
        normalized = _normalize_title(content)
        return normalized or fallback_title
    except Exception:
        return fallback_title


def _extract_content_text(response) -> str:
    content = response.choices[0].message.content
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(text_parts).strip()
    return (content or "").strip()


def _normalize_title(text: str | None) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return ""
    normalized = normalized.splitlines()[0].strip()
    normalized = normalized.replace("标题：", "").replace("标题:", "").strip()
    normalized = _PUNCTUATION_PATTERN.sub("", normalized)
    normalized = normalized[:TITLE_MAX_LENGTH]
    return normalized
