from __future__ import annotations

from dataclasses import dataclass

from .common_schema import ServiceError


@dataclass(slots=True)
class OcrResponse:
    success: bool
    text: str = ""
    confidence: float | None = None
    error: ServiceError | None = None
