from __future__ import annotations

from dataclasses import dataclass, field

from .common_schema import ServiceError


@dataclass(slots=True)
class CaseSearchRequest:
    query: str
    fraud_type: str | None = None
    limit: int = 5


@dataclass(slots=True)
class CaseItem:
    title: str
    summary: str
    source: str | None = None
    similarity: float | None = None
    url: str | None = None
    case_type: str | None = None
    fraud_types: list[str] = field(default_factory=list)
    fraud_subtypes: list[str] = field(default_factory=list)
    suspects: list[str] = field(default_factory=list)
    victims: list[str] = field(default_factory=list)
    money: float | None = None
    locations: list[str] = field(default_factory=list)
    laws: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CaseSearchResponse:
    success: bool
    cases: list[CaseItem] = field(default_factory=list)
    total_count: int = 0
    error: ServiceError | None = None
