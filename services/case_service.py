from __future__ import annotations

from clients.neo4j_client import Neo4jClient
from infra.config import AppConfig
from schemas.case_schema import CaseItem, CaseSearchResponse
from schemas.common_schema import ServiceError


COUNT_QUERY = """
MATCH (case:案件)
WHERE case.content CONTAINS $keyword
    OR case.description CONTAINS $keyword
    OR case.name CONTAINS $keyword
RETURN count(DISTINCT case.name) AS count
"""

SEARCH_QUERY = """
MATCH (case:案件)
WHERE case.content CONTAINS $keyword
    OR case.description CONTAINS $keyword
    OR case.name CONTAINS $keyword
OPTIONAL MATCH (case:案件)-[:涉及嫌疑人]->(suspect)
OPTIONAL MATCH (case:案件)-[:涉及被害人]->(victim)
OPTIONAL MATCH (case:案件)-[:诈骗类型]->(fraud_type)
OPTIONAL MATCH (case:案件)-[:涉案资产]->(asset {type:"钱财"})
OPTIONAL MATCH (case:案件)-[]->(location:地点)
OPTIONAL MATCH (case:案件)-[]->(law:法律法规)
RETURN
    case.name AS name,
    case.description AS description,
    case.type AS type,
    COLLECT(DISTINCT fraud_type.name) AS types,
    COLLECT(DISTINCT fraud_type.subtype) AS subtypes,
    COLLECT(DISTINCT suspect.name) AS suspects,
    COLLECT(DISTINCT victim.name) AS victims,
    SUM(asset.amount) AS money,
    COLLECT(DISTINCT location.province) AS locations,
    COLLECT(DISTINCT law.name) AS laws
SKIP $skip LIMIT $limit
"""

RECOMMEND_QUERY = """
MATCH (c:案件)
RETURN c.name AS name
ORDER BY rand()
LIMIT $limit
"""


def search_cases(
    query: str,
    fraud_type: str | None = None,
    limit: int = 5,
    skip: int = 0,
    config: AppConfig | None = None,
) -> CaseSearchResponse:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return CaseSearchResponse(
            success=False,
            error=ServiceError(code="EMPTY_QUERY", message="请输入要检索的案件关键词。"),
        )

    try:
        client = Neo4jClient(config)
        count_rows = client.query(COUNT_QUERY, keyword=normalized_query)
        total_count = int(count_rows[0]["count"]) if count_rows else 0
        rows = client.query(SEARCH_QUERY, keyword=normalized_query, skip=skip, limit=limit)

        cases: list[CaseItem] = []
        for row in rows:
            types = [item for item in row.get("types", []) if item]
            if fraud_type and fraud_type not in types:
                continue
            cases.append(
                CaseItem(
                    title=row.get("name") or "",
                    summary=row.get("description") or "",
                    source="neo4j",
                    case_type=row.get("type"),
                    fraud_types=types,
                    fraud_subtypes=[item for item in row.get("subtypes", []) if item],
                    suspects=[item for item in row.get("suspects", []) if item],
                    victims=[item for item in row.get("victims", []) if item],
                    money=row.get("money"),
                    locations=[item for item in row.get("locations", []) if item],
                    laws=[item for item in row.get("laws", []) if item],
                )
            )
        return CaseSearchResponse(success=True, cases=cases, total_count=total_count)
    except Exception as exc:
        return CaseSearchResponse(
            success=False,
            error=ServiceError(code="CASE_SEARCH_FAILED", message="案件检索失败。", detail={"error": str(exc)}),
        )


def get_case_names(limit: int = 5, config: AppConfig | None = None) -> list[str]:
    try:
        rows = Neo4jClient(config).query(RECOMMEND_QUERY, limit=limit)
        return [row["name"] for row in rows if row.get("name")]
    except Exception:
        return []
