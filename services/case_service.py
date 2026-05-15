from __future__ import annotations

import re
from typing import Any

from clients.neo4j_client import Neo4jClient
from infra.config import AppConfig
from schemas.case_schema import CaseItem, CaseSearchResponse
from schemas.common_schema import ServiceError


CASE_LABELS = ["案件", "妗堜欢"]
PERSON_LABELS = ["人物", "浜虹墿"]
ORG_LABELS = ["机构", "鏈烘瀯"]
LOCATION_LABELS = ["地点", "鍦扮偣"]
LAW_LABELS = ["法律法规", "娉曞緥娉曡"]

SUSPECT_RELS = ["涉及嫌疑人", "涉案嫌疑人", "娑夊強瀚岀枒浜"]
VICTIM_RELS = ["涉及被害人", "涉案被害人", "娑夊強琚浜"]
FRAUD_TYPE_RELS = ["诈骗类型", "璇堥獥绫诲瀷"]
ASSET_RELS = ["涉案资产", "娑夋璧勪骇"]


COUNT_QUERY = """
MATCH (case)
WHERE any(label IN labels(case) WHERE label IN $case_labels)
  AND (
    coalesce(case.content, "") CONTAINS $keyword
    OR coalesce(case.description, "") CONTAINS $keyword
    OR coalesce(case.name, "") CONTAINS $keyword
    OR coalesce(case.title, "") CONTAINS $keyword
  )
RETURN count(DISTINCT coalesce(case.name, case.title, elementId(case))) AS count
"""

SEARCH_QUERY = """
MATCH (case)
WHERE any(label IN labels(case) WHERE label IN $case_labels)
  AND (
    coalesce(case.content, "") CONTAINS $keyword
    OR coalesce(case.description, "") CONTAINS $keyword
    OR coalesce(case.name, "") CONTAINS $keyword
    OR coalesce(case.title, "") CONTAINS $keyword
  )
OPTIONAL MATCH (case)-[suspect_rel]->(suspect)
WHERE type(suspect_rel) IN $suspect_rels
OPTIONAL MATCH (case)-[victim_rel]->(victim)
WHERE type(victim_rel) IN $victim_rels
OPTIONAL MATCH (case)-[fraud_type_rel]->(fraud_type)
WHERE type(fraud_type_rel) IN $fraud_type_rels
OPTIONAL MATCH (case)-[asset_rel]->(asset)
WHERE type(asset_rel) IN $asset_rels
OPTIONAL MATCH (case)-[]->(location)
WHERE any(label IN labels(location) WHERE label IN $location_labels)
OPTIONAL MATCH (case)-[]->(law)
WHERE any(label IN labels(law) WHERE label IN $law_labels)
RETURN
    coalesce(case.name, case.title, "未命名案件") AS name,
    coalesce(case.description, case.summary, case.content, "") AS description,
    case.type AS type,
    COLLECT(DISTINCT fraud_type.name) AS types,
    COLLECT(DISTINCT fraud_type.subtype) AS subtypes,
    COLLECT(DISTINCT suspect.name) AS suspects,
    COLLECT(DISTINCT victim.name) AS victims,
    SUM(CASE WHEN coalesce(asset.type, "") IN ["钱财", "閽辫储"] THEN asset.amount ELSE null END) AS money,
    COLLECT(DISTINCT coalesce(location.province, location.name)) AS locations,
    COLLECT(DISTINCT law.name) AS laws
SKIP $skip LIMIT $limit
"""

RECOMMEND_QUERY = """
MATCH (c)
WHERE any(label IN labels(c) WHERE label IN $case_labels)
RETURN coalesce(c.name, c.title) AS name
ORDER BY rand()
LIMIT $limit
"""

SCHEMA_SUMMARY_QUERY = """
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY count DESC
LIMIT $limit
"""

CASE_INTENT_PREFIXES = (
    "和我讲一个",
    "给我讲一个",
    "帮我讲一个",
    "讲一个",
    "讲讲",
    "说说",
    "介绍一个",
    "分析一个",
    "总结一个",
    "聊聊",
)

CASE_INTENT_SUFFIXES = (
    "这个问题",
    "这个案件",
    "这个案子",
    "这个案",
    "的问题",
    "的情况",
)


def _query_params(**extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "case_labels": CASE_LABELS,
        "person_labels": PERSON_LABELS,
        "org_labels": ORG_LABELS,
        "location_labels": LOCATION_LABELS,
        "law_labels": LAW_LABELS,
        "suspect_rels": SUSPECT_RELS,
        "victim_rels": VICTIM_RELS,
        "fraud_type_rels": FRAUD_TYPE_RELS,
        "asset_rels": ASSET_RELS,
    }
    params.update(extra)
    return params


def extract_case_search_keyword(query: str) -> str:
    normalized = re.sub(r"\s+", "", (query or "").strip())
    if not normalized:
        return ""

    cleaned = normalized
    for prefix in CASE_INTENT_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    for suffix in CASE_INTENT_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    named_case_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,40}(?:诈骗案|案件|案情|判决书))", cleaned)
    if named_case_match:
        return named_case_match.group(1)

    fraud_case_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,30}(?:集资诈骗|合同诈骗|电信诈骗|诈骗))", cleaned)
    if fraud_case_match:
        return fraud_case_match.group(1)

    return cleaned


def _keyword_candidates(query: str) -> list[str]:
    normalized = (query or "").strip()
    extracted = extract_case_search_keyword(normalized)
    candidates = [candidate for candidate in (extracted, normalized) if candidate]

    if "杀猪盘" in normalized:
        candidates.extend(["婚恋", "交友", "投资", "虚假投资"])
    if extracted.endswith("诈骗案"):
        candidates.append(extracted[:-1])
    if extracted.endswith("案件"):
        candidates.append(extracted[:-2])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


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
        last_total_count = 0
        rows: list[dict] = []
        for keyword in _keyword_candidates(normalized_query):
            count_rows = client.query(COUNT_QUERY, **_query_params(keyword=keyword))
            total_count = int(count_rows[0]["count"]) if count_rows else 0
            candidate_rows = client.query(SEARCH_QUERY, **_query_params(keyword=keyword, skip=skip, limit=limit))
            last_total_count = total_count
            if candidate_rows:
                rows = candidate_rows
                break

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
        return CaseSearchResponse(success=True, cases=cases, total_count=last_total_count)
    except Exception as exc:
        return CaseSearchResponse(
            success=False,
            error=ServiceError(code="CASE_SEARCH_FAILED", message="案件检索失败。", detail={"error": str(exc)}),
        )


def get_case_names(limit: int = 5, config: AppConfig | None = None) -> list[str]:
    rows = Neo4jClient(config).query(RECOMMEND_QUERY, **_query_params(limit=limit))
    return [row["name"] for row in rows if row.get("name")]


def get_schema_summary(limit: int = 20, config: AppConfig | None = None) -> list[dict[str, Any]]:
    return Neo4jClient(config).query(SCHEMA_SUMMARY_QUERY, limit=limit)
