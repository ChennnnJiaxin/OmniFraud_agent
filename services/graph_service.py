from __future__ import annotations

import re
from typing import Any

from clients.neo4j_client import Neo4jClient
from infra.config import AppConfig
from schemas.common_schema import ServiceError
from schemas.graph_schema import GraphEdge, GraphNode, GraphQueryResponse

CASE_LABELS = ["案件", "妗堜欢"]
TOOL_LABELS = ["工具", "宸ュ叿"]
FRAUD_TYPE_LABELS = ["诈骗类型", "璇堥獥绫诲瀷"]
LOCATION_LABELS = ["地点", "鍦扮偣"]
PERSON_LABELS = ["人物", "浜虹墿"]

TOOL_RELS = ["涉案工具", "娑夋宸ュ叿"]
FRAUD_TYPE_RELS = ["诈骗类型", "璇堥獥绫诲瀷"]
LOCATION_RELS = ["案发地点", "所在地", "妗堝彂鍦扮偣", "鎵€鍦ㄥ湴"]
VICTIM_RELS = ["涉及被害人", "涉案被害人", "娑夊強琚浜"]
SUSPECT_RELS = ["涉及嫌疑人", "涉案嫌疑人", "娑夊強瀚岀枒浜"]

NODE_COLOR_MAP = {
    "案件": "#FF6347",
    "妗堜欢": "#FF6347",
    "人物": "#1E90FF",
    "浜虹墿": "#1E90FF",
    "机构": "#20B2AA",
    "鏈烘瀯": "#20B2AA",
    "地点": "#3CB371",
    "鍦扮偣": "#3CB371",
    "工具": "#FFA500",
    "宸ュ叿": "#FFA500",
    "诈骗类型": "#BA55D3",
    "璇堥獥绫诲瀷": "#BA55D3",
    "实体资产": "#FFD700",
    "瀹炰綋璧勪骇": "#FFD700",
    "罪名": "#A9A9A9",
    "缃悕": "#A9A9A9",
    "法律法规": "#CD853F",
    "娉曞緥娉曡": "#CD853F",
}

REL_COLOR_MAP = {
    "涉及被害人": "#FF69B4",
    "涉案被害人": "#FF69B4",
    "涉及嫌疑人": "#00CED1",
    "涉案嫌疑人": "#00CED1",
    "属于组织": "#7B68EE",
    "所在地": "#32CD32",
    "案发地点": "#FF4500",
    "触犯法律法规": "#8B0000",
    "诈骗类型": "#9400D3",
    "涉案工具": "#FF8C00",
    "人物关系": "#4682B4",
    "涉案资产": "#9ACD32",
    "罪名": "#808080",
    "刑事判决": "#DC143C",
    "赔偿金额": "#00FA9A",
    "赔偿给": "#00BFFF",
}

CASE_EXAMPLE_QUERY = """
MATCH (case)
WHERE any(label IN labels(case) WHERE label IN $case_labels)
OPTIONAL MATCH (case)-[tool_rel]->(tool)
WHERE type(tool_rel) IN $tool_rels
OPTIONAL MATCH (case)-[fraud_type_rel]->(fraud_type)
WHERE type(fraud_type_rel) IN $fraud_type_rels
OPTIONAL MATCH (case)-[location_rel]->(location)
WHERE type(location_rel) IN $location_rels
OPTIONAL MATCH (case)-[victim_rel]->(victim)
WHERE type(victim_rel) IN $victim_rels
OPTIONAL MATCH (case)-[suspect_rel]->(suspect)
WHERE type(suspect_rel) IN $suspect_rels
WITH case, tool, fraud_type, location, victim, suspect
WHERE
    coalesce(case.name, "") CONTAINS $keyword
    OR coalesce(case.title, "") CONTAINS $keyword
    OR coalesce(case.description, "") CONTAINS $keyword
    OR coalesce(case.content, "") CONTAINS $keyword
    OR coalesce(tool.name, "") CONTAINS $keyword
    OR coalesce(tool.type, "") CONTAINS $keyword
    OR coalesce(tool.usage, "") CONTAINS $keyword
    OR coalesce(fraud_type.name, "") CONTAINS $keyword
    OR coalesce(fraud_type.subtype, "") CONTAINS $keyword
WITH
    case,
    collect(DISTINCT tool.name) AS tools,
    collect(DISTINCT fraud_type.name) AS fraud_types,
    collect(DISTINCT fraud_type.subtype) AS fraud_subtypes,
    collect(DISTINCT location.province) AS provinces,
    collect(DISTINCT victim.name) AS victims,
    collect(DISTINCT suspect.name) AS suspects
RETURN
    coalesce(case.name, case.title, "未命名案件") AS name,
    coalesce(case.description, case.summary, "") AS description,
    case.type AS type,
    tools,
    fraud_types,
    fraud_subtypes,
    provinces,
    victims,
    suspects
LIMIT $limit
"""


def _query_params(**extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "case_labels": CASE_LABELS,
        "tool_labels": TOOL_LABELS,
        "fraud_type_labels": FRAUD_TYPE_LABELS,
        "location_labels": LOCATION_LABELS,
        "person_labels": PERSON_LABELS,
        "tool_rels": TOOL_RELS,
        "fraud_type_rels": FRAUD_TYPE_RELS,
        "location_rels": LOCATION_RELS,
        "victim_rels": VICTIM_RELS,
        "suspect_rels": SUSPECT_RELS,
    }
    params.update(extra)
    return params


def _extract_case_keywords(question: str) -> list[str]:
    normalized = re.sub(r"\s+", "", (question or "").strip())
    if not normalized:
        return []

    if "杀猪盘" in normalized:
        return ["杀猪盘", "婚恋", "交友", "投资"]

    keywords = []
    for keyword in (
        "刷单",
        "投资",
        "贷款",
        "客服",
        "公检法",
        "冒充",
        "虚假",
        "电信",
        "网络",
        "交友",
        "购物",
        "银行卡",
        "合同诈骗",
        "集资诈骗",
        "诈骗",
    ):
        if keyword in normalized:
            keywords.append(keyword)

    match = re.search(r"([\u4e00-\u9fff]{2,12})(?:诈骗案|案例|案件|案子|特征|套路)", normalized)
    if match:
        keywords.append(match.group(1))

    deduped: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword and keyword not in seen:
            seen.add(keyword)
            deduped.append(keyword)
    return deduped


def _format_case_examples(rows: list[dict], keyword: str, original_question: str) -> str:
    if not rows:
        if "杀猪盘" in original_question:
            return (
                "知识图谱中未直接命中“杀猪盘”字样，但可参考婚恋交友、虚假投资相关节点。"
                "这类诈骗通常先通过社交或婚恋关系建立信任，再引导受害人进入虚假投资、博彩或理财平台，"
                "常见特征包括长期情感铺垫、小额盈利诱导、催促加大投入、提现受阻后索要保证金或解冻费。"
            )
        return f"知识图谱中暂未检索到与“{keyword}”直接匹配的案件。"

    lines = [f"知识图谱中检索到以下与“{keyword}”相关的案件线索："]
    for index, row in enumerate(rows, start=1):
        name = row.get("name") or "未命名案件"
        description = row.get("description") or "暂无案情摘要"
        case_type = row.get("type") or "未标注类型"
        tools = "、".join(item for item in row.get("tools", []) if item) or "未标注"
        fraud_types = "、".join(
            item for item in (row.get("fraud_types", []) + row.get("fraud_subtypes", [])) if item
        ) or "未标注"
        provinces = "、".join(item for item in row.get("provinces", []) if item) or "未标注"
        lines.append(
            f"{index}. {name}：{description} 类型：{case_type}；诈骗类型：{fraud_types}；涉案工具：{tools}；地点：{provinces}。"
        )

    if "杀猪盘" in original_question:
        lines.append("结合这些婚恋交友和投资类线索，杀猪盘的核心特征是先建立亲密信任，再把关系转化为投资、博彩或转账诱导。")
    return "\n".join(lines)


def _query_case_examples(question: str, config: AppConfig | None = None) -> str:
    keywords = _extract_case_keywords(question)
    if not keywords:
        return "知识图谱暂未识别出明确的案件检索关键词，已跳过图谱扩展查询。"

    client = Neo4jClient(config)
    last_keyword = keywords[0]
    for keyword in keywords:
        rows = client.query(CASE_EXAMPLE_QUERY, **_query_params(keyword=keyword, limit=8))
        if rows:
            return _format_case_examples(rows, keyword, question)
        last_keyword = keyword
    return _format_case_examples([], last_keyword, question)


def query_fraud_graph(query: str, entity: str | None = None, config: AppConfig | None = None) -> GraphQueryResponse:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return GraphQueryResponse(
            success=False,
            error=ServiceError(code="EMPTY_QUERY", message="请输入图谱查询问题。"),
        )

    try:
        final_query = f"{normalized_query}，重点关注：{entity}" if entity else normalized_query
        explanation = _query_case_examples(final_query, config=config)
        return GraphQueryResponse(success=True, explanation=explanation)
    except Exception as exc:
        return GraphQueryResponse(
            success=False,
            error=ServiceError(code="GRAPH_QUERY_FAILED", message="图谱查询失败。", detail={"error": str(exc)}),
        )


def get_case_detail(case_name: str, config: AppConfig | None = None) -> dict | None:
    cypher_query = """
    MATCH (case)
    WHERE any(label IN labels(case) WHERE label IN $case_labels)
      AND coalesce(case.name, case.title) = $case_name
    RETURN case
    """
    try:
        record = Neo4jClient(config).query_single(cypher_query, **_query_params(case_name=case_name))
        if not record:
            return None
        case = record["case"]
        return {
            "name": case.get("name") or case.get("title"),
            "description": case.get("description") or case.get("summary"),
            "content": case.get("content"),
        }
    except Exception:
        return None


def get_case_graph_data(case_name: str, config: AppConfig | None = None) -> GraphQueryResponse:
    cypher_query = """
    MATCH (case)
    WHERE any(label IN labels(case) WHERE label IN $case_labels)
      AND coalesce(case.name, case.title) = $case_name
    OPTIONAL MATCH path=(case)-[rels*1..2]-(related)
    WHERE ALL(
        n IN nodes(path)
        WHERE NOT any(label IN labels(n) WHERE label IN $case_labels) OR n = case
    )
    WITH case, collect(path) AS paths
    WITH
        reduce(all_nodes = [case], p IN paths | CASE WHEN p IS NULL THEN all_nodes ELSE all_nodes + nodes(p) END) AS raw_nodes,
        reduce(all_rels = [], p IN paths | CASE WHEN p IS NULL THEN all_rels ELSE all_rels + relationships(p) END) AS raw_rels
    UNWIND raw_nodes AS node
    WITH collect(DISTINCT node)[..100] AS nodes, raw_rels
    RETURN nodes,
           [rel IN raw_rels WHERE rel IS NOT NULL][..150] AS rels
    """

    try:
        record = Neo4jClient(config).query_single(cypher_query, **_query_params(case_name=case_name))
        if not record or not record["nodes"]:
            return GraphQueryResponse(
                success=False,
                error=ServiceError(code="CASE_NOT_FOUND", message="未找到相关案件信息。"),
            )

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_nodes: set[str] = set()
        for node in record["nodes"]:
            node_id = node.element_id
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            labels = list(node.labels)
            node_type = labels[0] if labels else "其他"
            props = {key: value for key, value in node.items() if key != "content" and not key.startswith("_")}
            nodes.append(
                GraphNode(
                    id=node_id,
                    label=node.get("name") or node.get("title") or "未知名称",
                    type=node_type,
                    properties=props,
                )
            )

        seen_edges: set[str] = set()
        for rel in record["rels"]:
            rel_id = rel.element_id
            if rel_id in seen_edges:
                continue
            seen_edges.add(rel_id)
            edges.append(
                GraphEdge(
                    source=rel.start_node.element_id,
                    target=rel.end_node.element_id,
                    type=rel.type,
                )
            )
        return GraphQueryResponse(success=True, entities=nodes, relations=edges)
    except Exception as exc:
        return GraphQueryResponse(
            success=False,
            error=ServiceError(code="GRAPH_DATA_FAILED", message="案件图谱加载失败。", detail={"error": str(exc)}),
        )


def create_case_network(case_name: str, config: AppConfig | None = None):
    from pyvis.network import Network

    graph_data = get_case_graph_data(case_name, config=config)
    if not graph_data.success:
        return None

    net = Network(directed=True, height="800px", width="100%", notebook=False, cdn_resources="in_line")
    net.set_options(
        """
        {
            "physics": {
                "enabled": true,
                "stabilization": {"enabled": true, "iterations": 100},
                "timestep": 0.5,
                "adaptiveTimestep": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 0.5
                }
            }
        }
        """
    )
    node_ids = set()
    for node in graph_data.entities:
        title = "\n".join(f"{key}: {value}" for key, value in node.properties.items())
        net.add_node(
            node.id,
            label=node.label,
            title=title,
            color=NODE_COLOR_MAP.get(node.type, "#888888"),
            font={"size": 12},
        )
        node_ids.add(node.id)
    for rel in graph_data.relations:
        if rel.source not in node_ids or rel.target not in node_ids:
            continue
        net.add_edge(
            rel.source,
            rel.target,
            label=rel.type,
            color=REL_COLOR_MAP.get(rel.type, "#666666"),
            width=1.5,
            arrows="to",
        )
    return net
