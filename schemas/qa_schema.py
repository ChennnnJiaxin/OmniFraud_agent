from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common_schema import ServiceError


@dataclass(slots=True)
class ChatRequest:
    question: str
    context: dict[str, Any] | None = None


@dataclass(slots=True)
class ChatReference:
    title: str
    source: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class ChatResponse:
    success: bool
    answer: str = ""
    references: list[ChatReference] = field(default_factory=list)
    error: ServiceError | None = None
