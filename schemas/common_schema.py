from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ServiceError:
    code: str
    message: str
    detail: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolLikeResult:
    success: bool
    data: dict[str, Any] | None = None
    error: ServiceError | None = None
