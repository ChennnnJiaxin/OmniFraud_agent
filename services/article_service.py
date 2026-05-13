from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from clients.storage_client import JsonStorageClient
from schemas.article_schema import ArticleCreateRequest, ArticleDetail, ArticleListResponse, ArticleSummary
from schemas.common_schema import ServiceError, ToolLikeResult

ARTICLE_PATH = Path("show/articles.json")


def _storage() -> JsonStorageClient:
    client = JsonStorageClient(ARTICLE_PATH)
    client.ensure_exists(default={"articles": []})
    return client


def _to_summary(article: dict) -> ArticleSummary:
    return ArticleSummary(
        id=article["id"],
        title=article["title"],
        content=article["content"],
        author=article.get("author", "匿名"),
        publish_time=article.get("publish_time", ""),
        view_count=len(article.get("view_timestamps", [])),
        is_top=bool(article.get("is_top", False)),
    )


def publish_article(request: ArticleCreateRequest) -> ToolLikeResult:
    if not request.title.strip() or not request.content.strip():
        return ToolLikeResult(
            success=False,
            error=ServiceError(code="INVALID_ARTICLE", message="标题和内容不能为空。"),
        )

    storage = _storage()
    data = storage.load()
    new_article = {
        "id": str(uuid.uuid4()),
        "title": request.title,
        "content": request.content,
        "author": request.author or "匿名",
        "publish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "view_timestamps": [],
        "is_top": request.is_top,
    }
    data.setdefault("articles", []).append(new_article)
    storage.save(data)
    return ToolLikeResult(success=True, data={"id": new_article["id"]})


def list_articles(sort_by: str = "latest", limit: int = 10) -> ArticleListResponse:
    try:
        articles = _storage().load().get("articles", [])
        if sort_by == "hot":
            sorted_articles = sorted(articles, key=lambda item: len(item.get("view_timestamps", [])), reverse=True)
        else:
            sorted_articles = sorted(
                articles,
                key=lambda item: (-int(item.get("is_top", False)), item.get("publish_time", "")),
                reverse=True,
            )
        return ArticleListResponse(
            success=True,
            articles=[_to_summary(article) for article in sorted_articles[:limit]],
        )
    except Exception as exc:
        return ArticleListResponse(
            success=False,
            error=ServiceError(code="ARTICLE_LIST_FAILED", message="文章列表加载失败。", detail={"error": str(exc)}),
        )


def get_article_detail(article_id: str, increment_view: bool = True) -> ArticleDetail | None:
    storage = _storage()
    data = storage.load()
    article = next((item for item in data.get("articles", []) if item.get("id") == article_id), None)
    if not article:
        return None
    if increment_view:
        article.setdefault("view_timestamps", []).append(datetime.now().isoformat())
        storage.save(data)
    return ArticleDetail(
        id=article["id"],
        title=article["title"],
        content=article["content"],
        author=article.get("author", "匿名"),
        publish_time=article.get("publish_time", ""),
        view_count=len(article.get("view_timestamps", [])),
        is_top=bool(article.get("is_top", False)),
        view_timestamps=list(article.get("view_timestamps", [])),
    )
