from __future__ import annotations

import re
from functools import lru_cache

from clients.neo4j_client import Neo4jClient
from infra.config import AppConfig, load_app_config
from schemas.common_schema import ServiceError
from schemas.graph_schema import GraphEdge, GraphNode, GraphQueryResponse

NODE_COLOR_MAP = {
    "案件": "#FF6347",
    "人物": "#1E90FF",
    "机构": "#20B2AA",
    "地点": "#3CB371",
    "工具": "#FFA500",
    "诈骗类型": "#BA55D3",
    "实体资产": "#FFD700",
    "罪名": "#A9A9A9",
    "法律法规": "#CD853F",
}

REL_COLOR_MAP = {
    "涉及被害人": "#FF69B4",
    "涉及嫌疑人": "#00CED1",
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
    "赔偿给": "#00FA9A",
    "赔偿量": "#00BFFF",
}

CASE_EXAMPLE_QUERY = """
MATCH (case:案件)
OPTIONAL MATCH (case)-[:涉案工具]->(tool:工具)
OPTIONAL MATCH (case)-[:诈骗类型]->(fraud_type:诈骗类型)
OPTIONAL MATCH (case)-[:案发地点]->(location:地点)
OPTIONAL MATCH (case)-[:涉及被害人]->(victim:人物)
OPTIONAL MATCH (case)-[:涉及嫌疑人]->(suspect:人物)
WHERE
    case.name CONTAINS $keyword
    OR case.description CONTAINS $keyword
    OR case.content CONTAINS $keyword
    OR tool.name CONTAINS $keyword
    OR tool.type CONTAINS $keyword
    OR tool.usage CONTAINS $keyword
    OR fraud_type.name CONTAINS $keyword
    OR fraud_type.subtype CONTAINS $keyword
WITH
    case,
    collect(DISTINCT tool.name) AS tools,
    collect(DISTINCT fraud_type.name) AS fraud_types,
    collect(DISTINCT fraud_type.subtype) AS fraud_subtypes,
    collect(DISTINCT location.province) AS provinces,
    collect(DISTINCT victim.name) AS victims,
    collect(DISTINCT suspect.name) AS suspects
RETURN
    case.name AS name,
    case.description AS description,
    case.type AS type,
    tools,
    fraud_types,
    fraud_subtypes,
    provinces,
    victims,
    suspects
LIMIT $limit
"""

_CYPHER_PROMPT_TEMPLATE = """
You are an expert Neo4j developer.
Translate the user question into Cypher based only on the provided schema.
The graph uses Chinese labels and relationship names. Never translate, invent, or alias schema labels or relationships.

Important exact labels:
案件, 人物, 机构, 地点, 工具, 诈骗类型, 实体资产, 罪名, 法律法规

Important exact relationships:
涉及被害人, 涉及嫌疑人, 属于组织, 所在地, 案发地点, 触犯法律法规, 诈骗类型, 涉案工具, 人物关系, 涉案资产, 罪名, 刑事判决, 赔偿给, 赔偿量

Use 案件, not 案例. Use 涉案工具/诈骗类型, not 涉及.
Do not return entire nodes. Limit example lists to at most 8 records.

Schema:
{schema}

Question:
{question}
"""


def _config_key(config: AppConfig | None) -> tuple[str | None, ...]:
    current = config or load_app_config()
    return (
        current.openai_api_key,
        current.openai_model,
        current.openai_base_url,
        current.neo4j_uri,
        current.neo4j_username,
        current.neo4j_password,
        current.neo4j_database,
    )


def _extract_case_keyword(question: str) -> str | None:
    normalized = (question or "").strip()
    if not normalized:
        return None
    if not any(word in normalized for word in ("案例", "案件", "案子")):
        return None

    for keyword in ("手机", "刷单", "投资", "贷款", "客服", "公检法", "冒充", "虚假", "电信", "网络", "交友", "购物"):
        if keyword in normalized:
            return keyword
    match = re.search(r"([\u4e00-\u9fff]{2,8})(?:诈骗|案例|案件|案子)", normalized)
    return match.group(1) if match else "诈骗"


def _format_case_examples(rows: list[dict], keyword: str) -> str:
    if not rows:
        return f"知识图谱中暂未检索到与“{keyword}”直接匹配的案件。"

    lines = [f"知识图谱中检索到以下与“{keyword}”相关的案件："]
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
    return "\n".join(lines)


def _query_case_examples(question: str, config: AppConfig | None = None) -> str | None:
    keyword = _extract_case_keyword(question)
    if keyword is None:
        return None
    rows = Neo4jClient(config).query(CASE_EXAMPLE_QUERY, keyword=keyword, limit=8)
    return _format_case_examples(rows, keyword)


@lru_cache(maxsize=8)
def _build_cypher_chain(config_key: tuple[str | None, ...]):
    from langchain.prompts.prompt import PromptTemplate
    from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
    from langchain_openai import ChatOpenAI

    config = AppConfig(
        openai_api_key=config_key[0],
        openai_model=config_key[1],
        openai_base_url=config_key[2],
        neo4j_uri=config_key[3],
        neo4j_username=config_key[4],
        neo4j_password=config_key[5],
        neo4j_database=config_key[6],
    )
    llm = ChatOpenAI(
        openai_api_key=config.openai_api_key,
        model=config.openai_model,
        base_url=config.openai_base_url,
    )
    graph = Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database,
        refresh_schema=True,
    )
    cypher_prompt = PromptTemplate.from_template(_CYPHER_PROMPT_TEMPLATE)
    return GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True,
    )


def query_fraud_graph(query: str, entity: str | None = None, config: AppConfig | None = None) -> GraphQueryResponse:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return GraphQueryResponse(
            success=False,
            error=ServiceError(code="EMPTY_QUERY", message="请输入图谱查询问题。"),
        )

    try:
        deterministic_answer = _query_case_examples(normalized_query, config=config)
        if deterministic_answer is not None:
            return GraphQueryResponse(success=True, explanation=deterministic_answer)

        final_query = f"{normalized_query}（重点关注：{entity}）" if entity else normalized_query
        result = _build_cypher_chain(_config_key(config)).invoke({"query": final_query})
        explanation = result.get("result") if isinstance(result, dict) else str(result)
        return GraphQueryResponse(success=True, explanation=explanation)
    except Exception as exc:
        return GraphQueryResponse(
            success=False,
            error=ServiceError(code="GRAPH_QUERY_FAILED", message="图谱查询失败。", detail={"error": str(exc)}),
        )


def get_case_detail(case_name: str, config: AppConfig | None = None) -> dict | None:
    cypher_query = """
    MATCH (case:案件 {name: $case_name})
    RETURN case
    """
    try:
        client = Neo4jClient(config)
        with client.create_driver() as driver:
            with driver.session() as session:
                record = session.run(cypher_query, case_name=case_name).single()
                if not record:
                    return None
                case = record["case"]
                return {
                    "name": case.get("name"),
                    "description": case.get("description"),
                    "content": case.get("content"),
                }
    except Exception:
        return None


def get_case_graph_data(case_name: str, config: AppConfig | None = None) -> GraphQueryResponse:
    cypher_query = """
    MATCH (case:案件 {name: $case_name})
    OPTIONAL MATCH path=(case)-[rels*1..2]-(related)
    WHERE ALL(
        n IN nodes(path)
        WHERE NOT n:案件 OR n = case
    )
    WITH case,
         [n IN nodes(path) WHERE NOT n:案件] AS path_nodes,
         rels AS path_rels
    UNWIND (path_nodes + [case]) AS node
    UNWIND path_rels AS rel
    RETURN collect(DISTINCT node)[..100] AS nodes,
           collect(DISTINCT rel)[..150] AS rels
    """

    try:
        client = Neo4jClient(config)
        with client.create_driver() as driver:
            with driver.session() as session:
                record = session.run(cypher_query, case_name=case_name).single()
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
                            label=node.get("name", "未知名称"),
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

    net = Network(directed=True, height="800px", width="100%", notebook=False, cdn_resources="remote")
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
    for node in graph_data.entities:
        title = "\n".join(f"{key}: {value}" for key, value in node.properties.items())
        net.add_node(
            node.id,
            label=node.label,
            title=title,
            color=NODE_COLOR_MAP.get(node.type, "#888888"),
            font={"size": 12},
        )
    for rel in graph_data.relations:
        net.add_edge(
            rel.source,
            rel.target,
            label=rel.type,
            color=REL_COLOR_MAP.get(rel.type, "#666666"),
            width=1.5,
            arrows="to",
        )
    return net
