from __future__ import annotations

from functools import lru_cache

from infra.config import AppConfig, load_app_config
from schemas.common_schema import ServiceError
from schemas.qa_schema import ChatResponse


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


@lru_cache(maxsize=8)
def _build_agent(config_key: tuple[str | None, ...]):
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.schema import StrOutputParser
    from langchain.tools import Tool
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_neo4j import Neo4jChatMessageHistory, Neo4jGraph
    from langchain_openai import ChatOpenAI

    from services.graph_service import query_fraud_graph

    config = AppConfig(
        openai_api_key=config_key[0],
        openai_model=config_key[1],
        openai_base_url=config_key[2],
        neo4j_uri=config_key[3],
        neo4j_username=config_key[4],
        neo4j_password=config_key[5],
        neo4j_database=config_key[6],
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an anti-fraud publicity expert who can provide people with various anti-fraud knowledge."),
            ("human", "{input}"),
        ]
    )
    agent_prompt = PromptTemplate.from_template(
        """
You are an anti-fraud publicity expert who can provide people with various anti-fraud knowledge.
At the same time, you have a knowledge graph of fraud cases.

If the user asks about fraud cases, prefer the knowledge graph tool.
Do not answer unrelated questions.
Your final answer should be in the language of the user.

TOOLS:
------
{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
    )
    llm = ChatOpenAI(
        openai_api_key=config.openai_api_key,
        model=config.openai_model,
        base_url=config.openai_base_url,
    )
    general_chat = chat_prompt | llm | StrOutputParser()
    graph = Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database,
        refresh_schema=False,
    )
    tools = [
        Tool.from_function(
            name="General Chat",
            description="For general anti-fraud chat not covered by other tools",
            func=general_chat.invoke,
        ),
        Tool.from_function(
            name="Fraud Cases Knowledge Graph",
            description="Provide fraud case information using the knowledge graph",
            func=lambda question: (query_fraud_graph(question, config=config).explanation or "未查询到结果"),
        ),
    ]
    agent = create_react_agent(llm, tools, agent_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return RunnableWithMessageHistory(
        executor,
        lambda session_id: Neo4jChatMessageHistory(session_id=session_id, graph=graph),
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def _fallback_chat(question: str, config: AppConfig | None = None) -> str:
    from langchain_openai import ChatOpenAI

    current = config or load_app_config()
    llm = ChatOpenAI(
        openai_api_key=current.openai_api_key,
        model=current.openai_model,
        base_url=current.openai_base_url,
    )
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "你是一个反诈助手。请直接用自然中文回答用户，"
                    "优先给出判断、原因和下一步建议，不要写成工具报告。"
                ),
            },
            {"role": "user", "content": question},
        ]
    )
    return getattr(response, "content", None) or str(response)


def chat_with_anti_fraud_bot(question: str, context: dict | None = None, config: AppConfig | None = None) -> ChatResponse:
    normalized_question = (question or "").strip()
    if not normalized_question:
        return ChatResponse(
            success=False,
            error=ServiceError(code="EMPTY_QUESTION", message="请输入问题。"),
        )

    try:
        session_id = (context or {}).get("session_id", "default-session")
        response = _build_agent(_config_key(config)).invoke(
            {"input": normalized_question},
            {"configurable": {"session_id": session_id}},
        )
        return ChatResponse(success=True, answer=response["output"])
    except Exception as exc:
        try:
            fallback_answer = _fallback_chat(normalized_question, config=config)
            if fallback_answer:
                return ChatResponse(success=True, answer=fallback_answer)
        except Exception:
            pass
        return ChatResponse(
            success=False,
            error=ServiceError(code="QA_FAILED", message="问答服务暂时不可用。", detail={"error": str(exc)}),
        )


def chat_with_anti_fraud_bot_stream(question: str, context: dict | None = None, config: AppConfig | None = None):
    session_id = (context or {}).get("session_id", "default-session")
    return _build_agent(_config_key(config)).stream(
        {"input": question},
        {"configurable": {"session_id": session_id}},
    )
