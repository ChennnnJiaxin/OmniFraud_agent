from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common_schema import ServiceError


@dataclass(slots=True)
class SmsRecognizeRequest:
    text: str


@dataclass(slots=True)
class SmsRecognizeResponse:
    success: bool
    risk_level: str = "未知"
    fraud_type: str | None = None
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None
    error: ServiceError | None = None
