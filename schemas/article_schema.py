from __future__ import annotations

from dataclasses import dataclass, field

from .common_schema import ServiceError


@dataclass(slots=True)
class ArticleCreateRequest:
    title: str
    content: str
    author: str = "匿名"
    is_top: bool = False


@dataclass(slots=True)
class ArticleSummary:
    id: str
    title: str
    content: str
    author: str
    publish_time: str
    view_count: int
    is_top: bool = False


@dataclass(slots=True)
class ArticleDetail(ArticleSummary):
    view_timestamps: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ArticleListResponse:
    success: bool
    articles: list[ArticleSummary] = field(default_factory=list)
    error: ServiceError | None = None
