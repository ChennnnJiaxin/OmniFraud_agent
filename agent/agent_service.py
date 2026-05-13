from __future__ import annotations

from dataclasses import asdict
import re
from time import perf_counter
from typing import Any

from agent.memory import (
    AgentMemoryContext,
    FOLLOW_UP_TYPE_EXPLANATION,
    FOLLOW_UP_TYPE_NEXT_STEP,
    FOLLOW_UP_TYPE_PROCEDURE,
    FOLLOW_UP_TYPE_REMEDIAL,
    build_memory_context,
)
from agent.models import (
    AgentRunRequest,
    AgentRunResponse,
    AgentToolResult,
    TASK_CASE_SUMMARY,
    TASK_IMAGE_RISK,
    TASK_OUT_OF_SCOPE,
    TASK_TEXT_RISK,
    TASK_UNKNOWN,
    TASK_USER_RISK_PROFILE,
)
from agent.policy import AGENT_PROMPT_VERSION, RESPONSE_POLICY_VERSION, SAFETY_POLICY_VERSION
from agent.prompts import (
    build_case_summary_prompt,
    build_follow_up_prompt,
    build_profile_advice_prompt,
    build_remedial_follow_up_prompt,
    build_sms_failure_fallback_prompt,
    build_text_risk_advice_prompt,
)
from agent.response_builder import (
    build_out_of_scope_response,
    build_related_cases,
    build_safety_response,
    build_tool_trace,
    build_unknown_response,
    dedupe_suggestions,
    graph_payload,
)
from agent.router import detect_out_of_scope, route_agent_task
from agent.safety import SAFETY_INTENT_NORMAL, SafetyClassification, classify_safety_intent
from agent.session_models import AgentTaskRecord, AgentToolTraceRecord, new_id
from agent.session_store import get_agent_session_store
from agent.tools import case_search_tool, graph_query_tool, ocr_tool, qa_chat_tool, risk_report_tool, sms_recognize_tool
from schemas.common_schema import ServiceError

DEFAULT_SAFE_SUGGESTIONS = [
    "不要点击陌生链接，也不要向对方提供验证码、银行卡号或支付密码。",
    "通过官方 App、官方网站、官方客服电话或线下网点核实信息真伪。",
    "如果已经转账或泄露敏感信息，请立即联系银行并报警。",
]
REMEDIAL_REQUIRED_ACTIONS = [
    "立即停止继续操作，不要再点击链接或继续与对方通话。",
    "立刻联系银行或支付平台，申请冻结、挂失或限制账户。",
    "尽快修改银行、支付平台、电商和邮箱等重要账户密码。",
    "如果已泄露验证码或发生转账，尽快拨打银行客服、96110 或报警。",
    "保留短信、聊天记录、电话号码、链接、转账记录和 App 下载记录。",
    "警惕冒充客服、公检法、网警或资金追回人员的二次诈骗。",
]

PLATFORM_KEYWORDS = ("微信", "抖音", "QQ", "支付宝", "淘宝", "拼多多", "快手", "微博", "小红书")
FRAUD_TYPE_HINTS = ("刷单", "刷单返利", "投资理财", "冒充客服", "杀猪盘", "中奖", "保健品")
PROCEDURE_ALARM_HINTS = ("报警电话", "报警", "110")
PROCEDURE_96110_HINTS = ("96110", "反诈电话", "反诈中心")
PROCEDURE_BANK_HINTS = ("银行", "冻结", "挂失", "止付", "支付宝", "微信支付", "支付平台", "账户")
PROCEDURE_EVIDENCE_HINTS = ("证据", "聊天记录", "截图", "转账记录", "手机号", "账号", "链接")


def run_agent(request: AgentRunRequest | dict[str, Any]) -> AgentRunResponse:
    started_at = perf_counter()
    agent_request = _ensure_request(request)
    if not agent_request.user_input:
        return AgentRunResponse(
            success=False,
            task_type=TASK_UNKNOWN,
            conclusion="请输入需要分析的内容",
            suggestions=[
                "请补充短信原文、截图文字或用户画像信息。",
                "如果涉及陌生链接、验证码或转账，请先暂停操作。",
            ],
            fallback_message="输入为空，Agent 无法完成分析。",
            error=ServiceError(code="EMPTY_INPUT", message="user_input 不能为空"),
            handler_name="empty_input_handler",
            audit_info={"error_type": "EMPTY_INPUT", "duration_ms": 0},
        )

    store = get_agent_session_store()
    session = store.get_or_create_session(agent_request.session_id)
    agent_request.session_id = session.session_id
    store.append_message(session.session_id, "user", agent_request.user_input)
    memory_context = build_memory_context(
        store,
        session.session_id,
        user_input=agent_request.user_input,
        use_memory=agent_request.options.use_memory,
        history_limit=agent_request.options.history_limit,
    )
    safety = classify_safety_intent(
        agent_request.user_input,
        memory_summary=memory_context.memory_summary,
        recent_conclusion=memory_context.recent_task.conclusion if memory_context.recent_task else None,
    )

    try:
        response = _execute_agent(agent_request, memory_context, safety)
    except Exception:
        response = AgentRunResponse(
            success=False,
            task_type=TASK_UNKNOWN,
            conclusion="暂时无法完成当前分析",
            suggestions=DEFAULT_SAFE_SUGGESTIONS[:3],
            fallback_message="Agent 执行异常，已返回基础安全建议。",
            error=ServiceError(code="AGENT_EXECUTION_FAILED", message="Agent 执行异常"),
            handler_name="agent_exception_handler",
            audit_info={"error_type": "AGENT_EXECUTION_FAILED"},
        )

    response.session_id = session.session_id
    response.context_used = memory_context.context_used
    response.memory_summary = memory_context.memory_summary
    response.safety_intent = response.safety_intent or SAFETY_INTENT_NORMAL
    if not response.task_id:
        response.task_id = new_id()

    response.audit_info = dict(response.audit_info or {})
    response.audit_info.setdefault("context_used", memory_context.context_used)
    response.audit_info.setdefault("safety_intent", response.safety_intent)
    response.audit_info.setdefault("safety_severity", response.safety_severity)
    response.audit_info.setdefault("handler_name", response.handler_name)
    response.audit_info.setdefault("topic_switch_detected", memory_context.topic_switch_detected)
    response.audit_info.setdefault("topic_switch_reason", memory_context.topic_switch_reason)
    response.audit_info.setdefault("suggested_task_type", memory_context.topic_switch_task_type)
    response.audit_info.setdefault("previous_task_type", memory_context.recent_task.task_type if memory_context.recent_task else None)
    response.audit_info.setdefault("prompt_version", AGENT_PROMPT_VERSION)
    response.audit_info.setdefault("response_policy_version", RESPONSE_POLICY_VERSION)
    response.audit_info.setdefault("safety_policy_version", SAFETY_POLICY_VERSION)
    response.audit_info["duration_ms"] = int((perf_counter() - started_at) * 1000)

    assistant_message = _build_assistant_message_content(response)
    store.append_message(
        session.session_id,
        "assistant",
        assistant_message,
        metadata={
            "task_id": response.task_id,
            "task_type": response.task_type,
            "risk_level": response.risk_level,
            "safety_intent": response.safety_intent,
            "handler_name": response.handler_name,
        },
    )
    store.append_task_record(_build_task_record(session.session_id, agent_request, response, memory_context))
    return response


def _ensure_request(request: AgentRunRequest | dict[str, Any]) -> AgentRunRequest:
    if isinstance(request, AgentRunRequest):
        return request
    return AgentRunRequest(**request)


def create_agent_session(title: str | None = None, metadata: dict[str, Any] | None = None):
    return get_agent_session_store().create_session(title=title, metadata=metadata)


def get_agent_session(session_id: str):
    return get_agent_session_store().get_session(session_id)


def list_agent_session_messages(session_id: str, limit: int = 20):
    return get_agent_session_store().list_messages(session_id, limit=limit)


def list_agent_session_tasks(session_id: str, limit: int = 20):
    return get_agent_session_store().list_task_records(session_id, limit=limit)


def _execute_agent(
    request: AgentRunRequest,
    memory_context: AgentMemoryContext,
    safety: SafetyClassification,
) -> AgentRunResponse:
    if safety.safety_intent != SAFETY_INTENT_NORMAL:
        return _run_safety_intent(request, memory_context, safety)

    explicit_out_of_scope = detect_out_of_scope(request.user_input)
    if explicit_out_of_scope.is_out_of_scope:
        memory_context.follow_up_detected = False
        memory_context.follow_up_type = None

    if request.options.use_memory and memory_context.topic_switch_detected and memory_context.topic_switch_task_type:
        if memory_context.topic_switch_task_type == TASK_TEXT_RISK:
            response = _run_text_risk(request)
            return _annotate_response(
                response,
                handler_name="text_risk_handler",
                route_reason=memory_context.topic_switch_reason,
                router_decision=memory_context.topic_switch_task_type,
                matched_rules=memory_context.topic_switch_keywords,
                safety=safety,
                context_used=memory_context.context_used,
            )
        if memory_context.topic_switch_task_type == TASK_CASE_SUMMARY:
            response = _run_case_summary(request)
            return _annotate_response(
                response,
                handler_name="case_summary_handler",
                route_reason=memory_context.topic_switch_reason,
                router_decision=memory_context.topic_switch_task_type,
                matched_rules=memory_context.topic_switch_keywords,
                safety=safety,
                context_used=memory_context.context_used,
            )
        if memory_context.topic_switch_task_type == TASK_USER_RISK_PROFILE:
            response = _run_user_profile(request)
            return _annotate_response(
                response,
                handler_name="user_risk_profile_handler",
                route_reason=memory_context.topic_switch_reason,
                router_decision=memory_context.topic_switch_task_type,
                matched_rules=memory_context.topic_switch_keywords,
                safety=safety,
                context_used=memory_context.context_used,
            )

    if (
        request.options.use_memory
        and memory_context.follow_up_detected
        and memory_context.recent_task is not None
        and memory_context.recent_task.task_type != TASK_OUT_OF_SCOPE
    ):
        response = _run_follow_up(request, memory_context)
        return _annotate_response(
            response,
            handler_name=response.handler_name or "follow_up_handler",
            follow_up_type=memory_context.follow_up_type,
            safety=safety,
            context_used=memory_context.context_used,
        )

    if memory_context.follow_up_candidate_type == FOLLOW_UP_TYPE_PROCEDURE and memory_context.recent_task is None:
        response = _run_procedure_question_without_history(request)
        return _annotate_response(
            response,
            handler_name="follow_up_handler",
            follow_up_type=FOLLOW_UP_TYPE_PROCEDURE,
            safety=safety,
            context_used=False,
        )

    decision = route_agent_task(request)
    if decision.task_type == TASK_TEXT_RISK:
        response = _run_text_risk(request)
        return _annotate_response(
            response,
            handler_name="text_risk_handler",
            route_reason=decision.reason,
            router_decision=decision.task_type,
            matched_rules=decision.matched_rules,
            safety=safety,
            context_used=memory_context.context_used,
        )
    if decision.task_type == TASK_IMAGE_RISK:
        response = _run_image_risk(request)
        return _annotate_response(
            response,
            handler_name="image_risk_handler",
            route_reason=decision.reason,
            router_decision=decision.task_type,
            matched_rules=decision.matched_rules,
            safety=safety,
            context_used=memory_context.context_used,
        )
    if decision.task_type == TASK_CASE_SUMMARY:
        response = _run_case_summary(request)
        return _annotate_response(
            response,
            handler_name="case_summary_handler",
            route_reason=decision.reason,
            router_decision=decision.task_type,
            matched_rules=decision.matched_rules,
            safety=safety,
            context_used=memory_context.context_used,
        )
    if decision.task_type == TASK_USER_RISK_PROFILE:
        response = _run_user_profile(request)
        return _annotate_response(
            response,
            handler_name="user_risk_profile_handler",
            route_reason=decision.reason,
            router_decision=decision.task_type,
            matched_rules=decision.matched_rules,
            safety=safety,
            context_used=memory_context.context_used,
        )
    if decision.task_type == TASK_OUT_OF_SCOPE:
        response = _run_out_of_scope(request, decision)
        return _annotate_response(
            response,
            handler_name="out_of_scope_handler",
            route_reason=decision.reason,
            router_decision=decision.task_type,
            matched_rules=decision.matched_rules,
            safety=safety,
            context_used=False,
            error_type="OUT_OF_SCOPE",
        )

    response = build_unknown_response("输入信息不足，Agent 无法完成完整分析。", request.options.return_trace)
    return _annotate_response(
        response,
        handler_name="unknown_handler",
        route_reason=decision.reason,
        router_decision=decision.task_type,
        matched_rules=decision.matched_rules,
        safety=safety,
        context_used=memory_context.context_used,
        error_type="ROUTER_UNKNOWN",
    )


def _run_safety_intent(
    request: AgentRunRequest,
    memory_context: AgentMemoryContext,
    safety: SafetyClassification,
) -> AgentRunResponse:
    task_type = _resolve_safety_task_type(request, memory_context)
    tool_results: list[AgentToolResult] = []
    if memory_context.follow_up_detected and memory_context.recent_task is not None:
        tool_results.append(
            _local_tool_result(
                "follow_up_handler",
                True,
                "在会话追问上下文中触发安全兜底处理",
                data={"follow_up_type": memory_context.follow_up_type},
                handler_name="follow_up_handler",
            )
        )
    tool_results.extend(
        [
        _local_tool_result(
            "safety_classifier",
            True,
            f"识别为 {safety.safety_intent}，严重程度={safety.severity}",
            data={
                "matched_keywords": list(safety.matched_keywords),
                "reason": safety.reason,
            },
            handler_name="safety_classifier",
            error_type=safety.error_type,
        ),
        _local_tool_result(
            safety.handler_name,
            True,
            "已使用本地安全模板生成高危响应",
            data={"task_type": task_type},
            handler_name=safety.handler_name,
            error_type=safety.error_type,
        ),
        ]
    )
    response = build_safety_response(
        safety=safety,
        task_type=task_type,
        include_trace=request.options.return_trace,
        context_used=memory_context.context_used,
    )
    if memory_context.recent_task and memory_context.recent_task.fraud_type:
        response.fraud_type = memory_context.recent_task.fraud_type
    response.evidence = [safety.reason]
    if safety.matched_keywords:
        response.evidence.append(f"命中关键词：{'、'.join(safety.matched_keywords[:6])}")
    response.tool_trace = build_tool_trace(tool_results, request.options.return_trace)
    return _annotate_response(
        response,
        handler_name=safety.handler_name,
        safety=safety,
        context_used=memory_context.context_used,
        follow_up_type=memory_context.follow_up_type,
        error_type=safety.error_type,
    )


def _resolve_safety_task_type(request: AgentRunRequest, memory_context: AgentMemoryContext) -> str:
    if memory_context.recent_task and memory_context.recent_task.task_type not in {TASK_UNKNOWN, TASK_OUT_OF_SCOPE}:
        return memory_context.recent_task.task_type
    if request.input_type == "image" or request.image is not None:
        return TASK_IMAGE_RISK
    if request.profile or request.input_type == "profile":
        return TASK_USER_RISK_PROFILE
    return TASK_TEXT_RISK


def _run_out_of_scope(request: AgentRunRequest, decision) -> AgentRunResponse:
    tool_results = [
        _local_tool_result(
            "out_of_scope_detector",
            True,
            decision.reason,
            data={"matched_keywords": list(decision.matched_rules)},
            handler_name="out_of_scope_handler",
            error_type="OUT_OF_SCOPE",
            metadata={
                "out_of_scope_reason": decision.reason,
                "matched_keywords": list(decision.matched_rules),
            },
        )
    ]
    response = build_out_of_scope_response(request.options.return_trace)
    response.tool_trace = build_tool_trace(tool_results, request.options.return_trace)
    response.audit_info.update(
        {
            "out_of_scope_reason": decision.reason,
            "matched_rules": list(decision.matched_rules),
        }
    )
    return response


def _annotate_response(
    response: AgentRunResponse,
    *,
    handler_name: str,
    safety: SafetyClassification,
    context_used: bool,
    router_decision: str | None = None,
    route_reason: str | None = None,
    matched_rules: list[str] | None = None,
    follow_up_type: str | None = None,
    error_type: str | None = None,
) -> AgentRunResponse:
    response.handler_name = handler_name
    response.safety_intent = response.safety_intent or safety.safety_intent
    if response.safety_intent != SAFETY_INTENT_NORMAL:
        response.safety_severity = response.safety_severity or safety.severity
    response.audit_info = dict(response.audit_info or {})
    response.audit_info.update(
        {
            "router_decision": router_decision,
            "route_reason": route_reason,
            "matched_rules": list(matched_rules or []),
            "safety_intent": response.safety_intent,
            "safety_severity": response.safety_severity,
            "matched_keywords": list(safety.matched_keywords),
            "follow_up_type": follow_up_type,
            "handler_name": handler_name,
            "context_used": context_used,
            "prompt_version": AGENT_PROMPT_VERSION,
            "response_policy_version": RESPONSE_POLICY_VERSION,
            "safety_policy_version": SAFETY_POLICY_VERSION,
            "error_type": error_type or response.audit_info.get("error_type"),
        }
    )
    return response


def _run_follow_up(request: AgentRunRequest, memory_context: AgentMemoryContext) -> AgentRunResponse:
    recent_task = memory_context.recent_task
    if recent_task is None:
        return build_unknown_response("当前没有可承接的历史分析结果。", request.options.return_trace)

    if memory_context.follow_up_type == FOLLOW_UP_TYPE_REMEDIAL:
        return _run_remedial_follow_up(request, memory_context)
    if memory_context.follow_up_type == FOLLOW_UP_TYPE_PROCEDURE:
        return _run_procedure_follow_up(request, memory_context)

    prompt = build_follow_up_prompt(
        user_query=request.user_input,
        recent_task_type=recent_task.task_type,
        recent_conclusion=recent_task.conclusion,
        recent_risk_level=recent_task.risk_level,
        recent_fraud_type=recent_task.fraud_type,
        recent_evidence=list(recent_task.evidence),
        recent_suggestions=list(recent_task.suggestions),
        memory_summary=memory_context.memory_summary,
        follow_up_type=memory_context.follow_up_type,
    )
    qa_result = qa_chat_tool(prompt)
    tool_results = [
        _local_tool_result(
            "follow_up_handler",
            True,
            f"承接上下文处理追问，类型={memory_context.follow_up_type or 'unknown'}",
            handler_name="follow_up_handler",
            data={"follow_up_type": memory_context.follow_up_type},
        ),
        qa_result,
    ]
    answer = (qa_result.data or {}).get("answer") if qa_result.success else _build_follow_up_fallback_answer(recent_task, memory_context)
    suggestions = dedupe_suggestions(recent_task.suggestions, DEFAULT_SAFE_SUGGESTIONS)
    return AgentRunResponse(
        success=qa_result.success or bool(answer),
        task_type=recent_task.task_type,
        conclusion=_build_follow_up_conclusion(recent_task, memory_context.follow_up_type),
        risk_level=recent_task.risk_level if recent_task.risk_level != "unknown" else "high",
        fraud_type=recent_task.fraud_type,
        evidence=list(recent_task.evidence),
        suggestions=suggestions[:5],
        answer=answer,
        tool_trace=build_tool_trace(tool_results, request.options.return_trace),
        fallback_message=None if qa_result.success else "问答服务暂时不可用，已基于上一轮结果返回承接建议。",
        error=None if qa_result.success else qa_result.error,
        handler_name="follow_up_handler",
        audit_info={"error_type": qa_result.error_type if not qa_result.success else None},
    )


def _run_remedial_follow_up(request: AgentRunRequest, memory_context: AgentMemoryContext) -> AgentRunResponse:
    recent_task = memory_context.recent_task
    if recent_task is None:
        return build_unknown_response("当前没有可承接的历史分析结果。", request.options.return_trace)

    required_actions = list(REMEDIAL_REQUIRED_ACTIONS)
    answer = _build_remedial_answer(request.user_input, recent_task, required_actions)
    tool_results = [
        _local_tool_result(
            "follow_up_handler",
            True,
            "识别为事后补救型追问，已优先返回强安全补救建议",
            data={"follow_up_type": FOLLOW_UP_TYPE_REMEDIAL},
            handler_name="follow_up_handler",
            error_type="SAFETY_REMEDIAL",
        )
    ]
    if recent_task.answer or recent_task.evidence:
        remedial_prompt = build_remedial_follow_up_prompt(
            user_query=request.user_input,
            recent_conclusion=recent_task.conclusion,
            recent_risk_level=recent_task.risk_level,
            recent_fraud_type=recent_task.fraud_type,
            memory_summary=memory_context.memory_summary,
            required_actions=required_actions,
        )
        tool_results[0].metadata = {"prompt_preview": remedial_prompt[:120]}

    suggestions = dedupe_suggestions(
        [
            "立即停止继续操作，不要再点击链接、共享屏幕或向对方提供任何验证码。",
            "立刻联系银行或支付平台，申请冻结、挂失或限制银行卡、支付账户和快捷支付。",
            "尽快修改银行、支付平台、电商、邮箱等重要账户密码，并检查是否新增陌生设备或收款人。",
            "如果已经泄露验证码、支付密码或发生转账，请尽快拨打银行客服、96110 或报警。",
            "保留短信、聊天记录、电话号码、链接、转账记录和 App 下载记录，警惕二次诈骗。",
        ],
        recent_task.suggestions,
        DEFAULT_SAFE_SUGGESTIONS,
    )
    return AgentRunResponse(
        success=True,
        task_type=recent_task.task_type,
        conclusion="如果已经填写银行卡号、验证码或完成相关敏感操作，存在账户被盗刷或被进一步诱导转账的高风险，应立即采取补救措施。",
        risk_level=_follow_up_risk_level(recent_task, force_high=True),
        fraud_type=recent_task.fraud_type,
        evidence=list(recent_task.evidence),
        suggestions=suggestions[:5],
        answer=answer,
        tool_trace=build_tool_trace(tool_results, request.options.return_trace),
        fallback_message=None,
        error=None,
        safety_intent="remedial_action",
        safety_severity="high",
        handler_name="follow_up_handler",
        audit_info={"error_type": "SAFETY_REMEDIAL"},
    )


def _run_procedure_follow_up(request: AgentRunRequest, memory_context: AgentMemoryContext) -> AgentRunResponse:
    recent_task = memory_context.recent_task
    if recent_task is None:
        return _run_procedure_question_without_history(request)

    answer = _build_procedure_answer(request.user_input, recent_task)
    suggestions = dedupe_suggestions(
        _procedure_suggestions(request.user_input),
        recent_task.suggestions,
        DEFAULT_SAFE_SUGGESTIONS,
    )
    tool_results = [
        _local_tool_result(
            "follow_up_handler",
            True,
            "识别为处置细节类追问，已基于上一轮风险结果返回本地处置说明",
            data={"follow_up_type": FOLLOW_UP_TYPE_PROCEDURE},
            handler_name="follow_up_handler",
            error_type="FOLLOW_UP_PROCEDURE",
        )
    ]
    return AgentRunResponse(
        success=True,
        task_type=recent_task.task_type,
        conclusion=_build_follow_up_conclusion(recent_task, FOLLOW_UP_TYPE_PROCEDURE),
        risk_level=_follow_up_risk_level(
            recent_task,
            force_high=(recent_task.metadata or {}).get("safety_intent") in {"remedial_action", "emergency_loss", "secondary_fraud"},
        ),
        fraud_type=recent_task.fraud_type,
        evidence=list(recent_task.evidence),
        suggestions=suggestions[:5],
        answer=answer,
        tool_trace=build_tool_trace(tool_results, request.options.return_trace),
        fallback_message=None,
        error=None,
        handler_name="follow_up_handler",
        audit_info={"error_type": "FOLLOW_UP_PROCEDURE"},
    )


def _run_procedure_question_without_history(request: AgentRunRequest) -> AgentRunResponse:
    answer = _build_generic_procedure_answer(request.user_input)
    suggestions = dedupe_suggestions(_procedure_suggestions(request.user_input), DEFAULT_SAFE_SUGGESTIONS)
    tool_results = [
        _local_tool_result(
            "follow_up_handler",
            True,
            "识别为反诈处置细节问题，已返回本地通用说明",
            data={"follow_up_type": FOLLOW_UP_TYPE_PROCEDURE},
            handler_name="follow_up_handler",
            error_type="FOLLOW_UP_PROCEDURE",
        )
    ]
    return AgentRunResponse(
        success=True,
        task_type=TASK_TEXT_RISK,
        conclusion="已返回反诈处置细节说明，可继续补充是否涉及验证码泄露、转账或账户异常，以便给出更具体的补救建议。",
        risk_level="unknown",
        suggestions=suggestions[:5],
        answer=answer,
        tool_trace=build_tool_trace(tool_results, request.options.return_trace),
        fallback_message=None,
        error=None,
        handler_name="follow_up_handler",
        audit_info={"error_type": "FOLLOW_UP_PROCEDURE"},
    )


def _follow_up_risk_level(recent_task: AgentTaskRecord, *, force_high: bool = False) -> str:
    if force_high:
        if recent_task.risk_level in {"high", "critical"}:
            return recent_task.risk_level
        return "high"
    if recent_task.risk_level and recent_task.risk_level != "unknown":
        return recent_task.risk_level
    return "high"


def _build_follow_up_conclusion(recent_task: AgentTaskRecord, follow_up_type: str | None) -> str:
    if follow_up_type == FOLLOW_UP_TYPE_EXPLANATION:
        return "已结合上一轮高风险分析结果，继续说明这类内容为什么可疑。"
    if follow_up_type == FOLLOW_UP_TYPE_PROCEDURE:
        return "已结合上一轮高风险分析结果，继续说明报警、反诈电话、冻结止付和证据保留等处置细节。"
    if follow_up_type == FOLLOW_UP_TYPE_NEXT_STEP:
        return "已结合上一轮高风险分析结果，继续给出当前应采取的步骤。"
    return f"已结合上一轮 {recent_task.task_type} 结果继续回答当前追问。"


def _build_follow_up_fallback_answer(recent_task: AgentTaskRecord, memory_context: AgentMemoryContext) -> str:
    if memory_context.follow_up_type == FOLLOW_UP_TYPE_EXPLANATION:
        evidence_text = "；".join(recent_task.evidence[:3]) if recent_task.evidence else recent_task.conclusion
        fraud_type_text = recent_task.fraud_type or "高风险诈骗"
        return f"上一轮之所以判断为 {fraud_type_text}，主要因为存在这些风险点：{evidence_text}。因此不应继续按对方要求操作。"
    if memory_context.follow_up_type == FOLLOW_UP_TYPE_PROCEDURE:
        return _build_procedure_answer(memory_context.recent_task.user_input if memory_context.recent_task else "", recent_task)
    if memory_context.follow_up_type == FOLLOW_UP_TYPE_NEXT_STEP:
        suggestion_text = "；".join(recent_task.suggestions[:3]) if recent_task.suggestions else "立即停止操作，并通过官方渠道核实。"
        return f"建议优先执行这些步骤：{suggestion_text}"
    return recent_task.answer or recent_task.conclusion


def _build_procedure_answer(user_input: str, recent_task: AgentTaskRecord) -> str:
    base = (
        "如果你已经泄露验证码、银行卡信息或发生转账风险，紧急情况下可以拨打 110 报警；"
        "96110 是全国反诈预警劝阻咨询电话，可用于反诈咨询、风险核实和劝阻提醒。"
    )
    bank = (
        "涉及银行卡或支付账户时，也应立即联系银行或支付平台官方客服申请冻结、挂失或止付，"
        "不要使用对方提供的电话或链接。"
    )
    evidence = "同时要保留短信、聊天记录、转账记录、手机号、收款账号、链接、App 和操作截图等证据。"

    if any(keyword in user_input for keyword in PROCEDURE_EVIDENCE_HINTS):
        return f"{base}{bank}{evidence}"
    if any(keyword in user_input for keyword in PROCEDURE_BANK_HINTS):
        return (
            "如果怀疑银行卡或支付账户存在风险，应尽快通过银行或支付平台官方 App、官网或官方客服办理冻结、挂失或止付，"
            "不要再使用对方提供的联系方式。"
            "如已泄露验证码、密码或已转账，同时可以拨打 110 和 96110，并保留聊天记录、转账记录、手机号和链接等证据。"
        )
    if any(keyword in user_input for keyword in PROCEDURE_96110_HINTS):
        return (
            "96110 是全国反诈预警劝阻咨询电话，主要用于反诈预警、风险核实和咨询；"
            "如果已经发生紧急资金风险或需要报警，仍应优先拨打 110。"
            "如涉及银行卡或支付账户异常，也要立即联系银行或支付平台官方客服申请冻结、挂失或止付，并保留证据。"
        )
    return f"{base}{bank}{evidence}"


def _build_generic_procedure_answer(user_input: str) -> str:
    if any(keyword in user_input for keyword in PROCEDURE_96110_HINTS):
        return (
            "96110 是全国反诈预警劝阻咨询电话，可用于反诈咨询、风险核实和预警提醒；"
            "如果已经发生紧急资金风险或需要正式报警，应该拨打 110。"
            "如涉及银行卡或支付账户异常，也请尽快联系银行或支付平台官方客服申请冻结、挂失或止付。"
        )
    if any(keyword in user_input for keyword in PROCEDURE_BANK_HINTS):
        return (
            "如果怀疑银行卡或支付账户存在风险，应尽快通过银行或支付平台官方 App、官网或官方客服办理冻结、挂失或止付；"
            "如已泄露验证码、密码或已发生转账风险，也可以拨打 110 和 96110，并注意保留聊天记录、转账记录、手机号和链接等证据。"
        )
    if any(keyword in user_input for keyword in PROCEDURE_EVIDENCE_HINTS):
        return (
            "建议保留短信、聊天记录、转账记录、收款账号、手机号、链接、App、通话记录和操作截图等证据；"
            "如果已经发生资金风险，可以拨打 110 报警，并拨打 96110 做反诈咨询，同时联系银行或支付平台官方客服止付或冻结。"
        )
    return (
        "紧急报警电话是 110；96110 是全国反诈预警劝阻咨询电话，可用于反诈咨询和风险核实。"
        "如果涉及银行卡或支付账户异常，也请尽快联系银行或支付平台官方客服申请冻结、挂失或止付，并保留短信、聊天记录、转账记录等证据。"
    )


def _procedure_suggestions(user_input: str) -> list[str]:
    suggestions = [
        "紧急资金风险可拨打 110 报警，反诈咨询和风险核实可拨打 96110。",
        "涉及银行卡或支付账户时，立即联系银行或支付平台官方客服申请冻结、挂失或止付。",
        "不要使用对方提供的电话、链接或二维码，要通过官方 App、官网或官方热线处理。",
        "保留短信、聊天记录、转账记录、手机号、账号、链接、App 和操作截图等证据。",
    ]
    if any(keyword in user_input for keyword in PROCEDURE_EVIDENCE_HINTS):
        suggestions.insert(0, "优先整理聊天记录、转账记录、收款账号、手机号、链接和截图等关键证据。")
    return suggestions


def _build_remedial_answer(user_input: str, recent_task: AgentTaskRecord, required_actions: list[str]) -> str:
    fraud_type_text = recent_task.fraud_type or "当前高风险诈骗场景"
    evidence_text = "；".join(recent_task.evidence[:3]) if recent_task.evidence else recent_task.conclusion
    action_text = "；".join(required_actions[:5])
    return (
        f"如果你已经填写银行卡号、验证码、点开链接或完成类似敏感操作，这仍然应视为 {fraud_type_text} 中的高风险补救场景。"
        f"上一轮已识别出的风险点包括：{evidence_text}。"
        f"现在最重要的是：{action_text}。"
        "请务必使用银行或平台官方 App、官网和客服电话核实，不要再使用对方提供的联系方式，也要警惕后续冒充客服、网警或资金追回人员的二次诈骗。"
    )


def _local_tool_result(
    tool_name: str,
    success: bool,
    summary: str,
    data: dict[str, Any] | None = None,
    *,
    duration_ms: int = 0,
    error: ServiceError | None = None,
    error_type: str | None = None,
    handler_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentToolResult:
    return AgentToolResult(
        tool_name=tool_name,
        success=success,
        data=data,
        error=error,
        summary=summary,
        duration_ms=duration_ms,
        error_type=error_type or (error.code if error else None),
        handler_name=handler_name,
        metadata=dict(metadata or {}),
    )


def _extract_subject_text(user_input: str) -> str:
    text = (user_input or "").strip()
    for separator in ("：", ":"):
        if separator in text:
            prefix, suffix = text.split(separator, 1)
            if len(prefix) <= 20 and suffix.strip():
                return suffix.strip()
    quoted = re.findall(r"[“\"]([^”\"]{6,})[”\"]", text)
    if quoted:
        return quoted[0].strip()
    return text


def _extract_profile(user_input: str, profile: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(profile or {})
    text = (user_input or "").strip()

    age_match = re.search(r"(\d{1,3})\s*岁", text)
    if age_match:
        merged.setdefault("age", int(age_match.group(1)))
    elif "60多岁" in text:
        merged.setdefault("age", 60)
    elif "70多岁" in text:
        merged.setdefault("age", 70)

    occupation_map = {
        "大学生": "学生",
        "学生": "学生",
        "退休": "退休人员",
        "退休人员": "退休人员",
        "宝妈": "自由职业",
        "上班族": "在职员工",
    }
    for keyword, occupation in occupation_map.items():
        if keyword in text and not merged.get("occupation"):
            merged["occupation"] = occupation
            break

    platforms = list(merged.get("platforms") or [])
    for platform in PLATFORM_KEYWORDS:
        if platform in text and platform not in platforms:
            platforms.append(platform)
    if platforms:
        merged["platforms"] = platforms

    fraud_types = list(merged.get("fraud_types") or [])
    for fraud_type in FRAUD_TYPE_HINTS:
        if fraud_type in text and fraud_type not in fraud_types:
            fraud_types.append(fraud_type)
    if fraud_types:
        merged["fraud_types"] = fraud_types

    if not merged.get("recent_experience") and text:
        merged["recent_experience"] = text

    return merged


def _has_minimum_profile(profile: dict[str, Any]) -> bool:
    return any(
        [
            profile.get("age") is not None,
            bool(profile.get("occupation")),
            bool(profile.get("platforms")),
            bool(profile.get("recent_experience")),
        ]
    )


def _normalize_related_cases(case_result: AgentToolResult | None) -> list[dict[str, Any]]:
    if case_result is None or not case_result.success:
        return []
    return (case_result.data or {}).get("cases", [])


def _run_text_risk(request: AgentRunRequest) -> AgentRunResponse:
    tool_results: list[AgentToolResult] = []
    subject_text = _extract_subject_text(request.user_input)

    sms_result = sms_recognize_tool(subject_text)
    tool_results.append(sms_result)

    sms_data = sms_result.data or {}
    fraud_type = sms_data.get("fraud_type") if sms_result.success else None
    case_query = fraud_type or subject_text

    case_result: AgentToolResult | None = None
    if case_query:
        case_result = case_search_tool(case_query, fraud_type=fraud_type, limit=request.options.case_limit)
        tool_results.append(case_result)

    if sms_result.success:
        qa_prompt = build_text_risk_advice_prompt(subject_text, sms_data, _normalize_related_cases(case_result))
        qa_result = qa_chat_tool(qa_prompt)
    else:
        qa_result = qa_chat_tool(build_sms_failure_fallback_prompt(subject_text))
    tool_results.append(qa_result)

    related_cases = build_related_cases(case_result)
    answer = (qa_result.data or {}).get("answer") if qa_result.success else None

    if sms_result.success:
        suggestions = dedupe_suggestions(sms_data.get("suggestions", []), [answer] if answer else [], DEFAULT_SAFE_SUGGESTIONS)
        risk_level = sms_data.get("risk_level", "unknown")
        conclusion = "该内容存在较高诈骗风险" if risk_level not in {"无风险", "low", "unknown"} else "该内容暂未发现明显高危特征"
        return AgentRunResponse(
            success=True,
            task_type=TASK_TEXT_RISK,
            conclusion=conclusion,
            risk_level=risk_level,
            fraud_type=fraud_type,
            confidence=sms_data.get("confidence"),
            evidence=sms_data.get("evidence", []),
            suggestions=suggestions[:5],
            related_cases=related_cases,
            answer=answer,
            tool_trace=build_tool_trace(tool_results, request.options.return_trace),
            fallback_message=None if case_result is None or case_result.success else "案例检索失败，已返回短信识别结果。",
            handler_name="text_risk_handler",
            audit_info={"error_type": qa_result.error_type if not qa_result.success else None},
        )

    return AgentRunResponse(
        success=qa_result.success,
        task_type=TASK_TEXT_RISK,
        conclusion="暂时无法完成精确识别，但该内容仍建议按高风险处理",
        risk_level="unknown",
        evidence=["短信识别服务未返回有效结果。"],
        suggestions=dedupe_suggestions(DEFAULT_SAFE_SUGGESTIONS, [answer] if answer else [])[:5],
        related_cases=related_cases,
        answer=answer,
        tool_trace=build_tool_trace(tool_results, request.options.return_trace),
        fallback_message="短信识别服务暂时不可用，已返回通用安全建议。",
        error=sms_result.error,
        handler_name="text_risk_handler",
        audit_info={"error_type": sms_result.error_type or "TOOL_FAILED"},
    )


def _run_image_risk(request: AgentRunRequest) -> AgentRunResponse:
    tool_results: list[AgentToolResult] = []
    if request.image is None:
        return AgentRunResponse(
            success=False,
            task_type=TASK_IMAGE_RISK,
            conclusion="未检测到可分析的图片输入",
            suggestions=[
                "请重新上传截图，或直接粘贴截图中的文字。",
                "如果截图涉及验证码、链接或转账，请先暂停操作。",
            ],
            fallback_message="图片输入缺失，无法执行 OCR。",
            error=ServiceError(code="MISSING_IMAGE", message="image 字段为空"),
            handler_name="image_risk_handler",
            audit_info={"error_type": "MISSING_IMAGE"},
        )

    ocr_result = ocr_tool(request.image)
    tool_results.append(ocr_result)
    ocr_text = ((ocr_result.data or {}).get("text") or "").strip() if ocr_result.success else ""

    if not ocr_result.success or not ocr_text:
        return AgentRunResponse(
            success=False,
            task_type=TASK_IMAGE_RISK,
            conclusion="当前无法直接完成截图风险分析",
            risk_level="unknown",
            evidence=["OCR 不可用或未提取到有效文本。"],
            suggestions=[
                "请手动输入截图中的聊天内容、短信原文或链接信息。",
                "不要点击截图中的陌生链接，不要向对方提供验证码或银行卡信息。",
                "如对方催促转账或退款，请通过官方渠道核实。",
            ],
            tool_trace=build_tool_trace(tool_results, request.options.return_trace),
            fallback_message="OCR 不可用时，建议改为手动输入截图文字后重新分析。",
            error=ocr_result.error or ServiceError(code="OCR_FAILED", message="OCR 未提取到有效文本"),
            handler_name="image_risk_handler",
            audit_info={"error_type": ocr_result.error_type or "OCR_FAILED"},
        )

    text_request = AgentRunRequest(user_input=ocr_text, input_type="text", options=request.options)
    text_response = _run_text_risk(text_request)
    text_response.task_type = TASK_IMAGE_RISK
    text_response.conclusion = f"基于截图文本识别，{text_response.conclusion}"
    if request.options.return_trace:
        text_response.tool_trace = build_tool_trace(tool_results, True) + text_response.tool_trace
    text_response.handler_name = "image_risk_handler"
    text_response.audit_info = dict(text_response.audit_info or {})
    return text_response


def _run_case_summary(request: AgentRunRequest) -> AgentRunResponse:
    tool_results: list[AgentToolResult] = []
    limit = request.options.case_limit

    case_result = case_search_tool(request.user_input, limit=limit)
    tool_results.append(case_result)

    graph_result = graph_query_tool(request.user_input)
    tool_results.append(graph_result)

    qa_prompt = build_case_summary_prompt(request.user_input, _normalize_related_cases(case_result), graph_payload(graph_result))
    qa_result = qa_chat_tool(qa_prompt)
    tool_results.append(qa_result)

    related_cases = build_related_cases(case_result)
    case_data = case_result.data or {}
    total_count = case_data.get("total_count", len(related_cases))
    fallback_message = None
    if "全部" in request.user_input and total_count > limit:
        fallback_message = f"当前仅展示前 {limit} 条案例，如需更多结果建议分页查看。"

    answer = None
    if qa_result.success:
        answer = (qa_result.data or {}).get("answer")
    elif graph_result.success:
        answer = (graph_result.data or {}).get("explanation")

    suggestions = [
        "重点关注作案话术、诱导转账路径和受害前置信号，避免只看个案表面。",
        "遇到类似情形时，通过官方渠道核验，不要因退款、投资或兼职承诺而先行转账。",
    ]
    if not related_cases:
        suggestions.insert(0, "当前未检索到直接匹配案例，可尝试换用更具体的诈骗类型关键词。")

    return AgentRunResponse(
        success=case_result.success or graph_result.success or qa_result.success,
        task_type=TASK_CASE_SUMMARY,
        conclusion="已完成诈骗案例查询总结" if related_cases or answer else "暂未检索到明确案例结果",
        risk_level="unknown",
        evidence=[f"检索到 {len(related_cases)} 条相关案例。"] if related_cases else [],
        suggestions=suggestions,
        related_cases=related_cases,
        graph_result=graph_payload(graph_result),
        answer=answer,
        tool_trace=build_tool_trace(tool_results, request.options.return_trace),
        fallback_message=fallback_message,
        error=None if (case_result.success or graph_result.success or qa_result.success) else case_result.error,
        handler_name="case_summary_handler",
        audit_info={"error_type": case_result.error_type if not case_result.success else None},
    )


def _run_user_profile(request: AgentRunRequest) -> AgentRunResponse:
    tool_results: list[AgentToolResult] = []
    profile = _extract_profile(request.user_input, request.profile)

    if not _has_minimum_profile(profile):
        return AgentRunResponse(
            success=False,
            task_type=TASK_USER_RISK_PROFILE,
            conclusion="用户画像信息不足，暂时无法完成精准评估",
            risk_level="unknown",
            suggestions=[
                "请补充年龄、职业、常用平台或最近遭遇中的至少一项。",
                "如果已经遇到陌生链接、垫付、客服退款或投资荐股，请先暂停操作。",
            ],
            fallback_message="画像信息不足，建议补充年龄、职业、平台或最近经历。",
            error=ServiceError(code="INSUFFICIENT_PROFILE", message="用户画像信息不足"),
            handler_name="user_risk_profile_handler",
            audit_info={"error_type": "INSUFFICIENT_PROFILE"},
        )

    risk_result = risk_report_tool(profile)
    tool_results.append(risk_result)
    risk_data = risk_result.data or {}
    vulnerable_types = list(risk_data.get("vulnerable_fraud_types", []))

    case_query = vulnerable_types[0] if vulnerable_types else profile.get("recent_experience") or request.user_input
    case_result = case_search_tool(case_query, limit=request.options.case_limit) if case_query else None
    if case_result is not None:
        tool_results.append(case_result)

    qa_prompt = build_profile_advice_prompt(request.user_input, profile, risk_data, _normalize_related_cases(case_result))
    qa_result = qa_chat_tool(qa_prompt)
    tool_results.append(qa_result)

    answer = (qa_result.data or {}).get("answer") if qa_result.success else risk_data.get("report") or None
    related_cases = build_related_cases(case_result)
    reasons = list(risk_data.get("reasons", []))
    suggestions = dedupe_suggestions(list(risk_data.get("suggestions", [])), DEFAULT_SAFE_SUGGESTIONS)

    return AgentRunResponse(
        success=risk_result.success,
        task_type=TASK_USER_RISK_PROFILE,
        conclusion="已完成人群风险分析",
        risk_level=risk_data.get("risk_level", "unknown"),
        evidence=reasons,
        suggestions=suggestions[:5],
        related_cases=related_cases,
        answer=answer,
        vulnerable_fraud_types=vulnerable_types,
        tool_trace=build_tool_trace(tool_results, request.options.return_trace),
        fallback_message=None if qa_result.success else "问答整理失败，已返回风险评估主结果。",
        error=risk_result.error if not risk_result.success else None,
        handler_name="user_risk_profile_handler",
        audit_info={"error_type": risk_result.error_type if not risk_result.success else None},
    )


def _build_assistant_message_content(response: AgentRunResponse) -> str:
    answer = (response.answer or "").strip()
    conclusion = (response.conclusion or "").strip()
    fallback = (response.fallback_message or "").strip()
    suggestions = [item.strip() for item in response.suggestions[:3] if item and item.strip()]
    evidence = [item.strip() for item in (response.evidence or [])[:4] if item and item.strip()]
    fraud_type = (response.fraud_type or "").strip()
    risk_level = (response.risk_level or "").strip()

    if answer:
        parts = [answer]

        if conclusion and conclusion not in answer:
            parts.append(conclusion)

        if evidence:
            evidence_text = "；".join(evidence)
            if fraud_type or risk_level:
                prefix_bits = []
                if fraud_type:
                    prefix_bits.append(f"诈骗类型更像{fraud_type}")
                if risk_level:
                    prefix_bits.append(f"整体风险可以判断为{risk_level}")
                parts.append("我主要是根据这些信号做这个判断的：" + "，".join(prefix_bits) + "。" + evidence_text)
            else:
                parts.append("我主要是根据这些信号做这个判断的：" + evidence_text)

        if suggestions:
            if len(suggestions) == 1:
                parts.append(f"你现在最值得先做的是：{suggestions[0]}")
            else:
                parts.append("如果你现在就要处理，建议先这样做：" + "；".join(suggestions))
        if fallback:
            parts.append(fallback)
        return "\n\n".join(parts)

    parts: list[str] = []
    if conclusion:
        parts.append(conclusion)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"你现在最值得先做的是：{suggestions[0]}")
        else:
            parts.append("你现在可以先这样处理：" + "；".join(suggestions))
    if fallback:
        parts.append(fallback)
    return "\n\n".join(part for part in parts if part)


def _build_task_record(
    session_id: str,
    request: AgentRunRequest,
    response: AgentRunResponse,
    memory_context: AgentMemoryContext,
) -> AgentTaskRecord:
    task_id = response.task_id or new_id()
    response.task_id = task_id
    audit_info = dict(response.audit_info or {})
    error_type = audit_info.get("error_type") or (response.error.code if response.error else None)
    metadata = {
        "context_used": memory_context.context_used,
        "memory_summary": memory_context.memory_summary,
        "input_type": request.input_type,
        "follow_up_detected": memory_context.follow_up_detected,
        "follow_up_type": audit_info.get("follow_up_type") or memory_context.follow_up_type,
        "topic_switch_detected": bool(audit_info.get("topic_switch_detected", memory_context.topic_switch_detected)),
        "topic_switch_reason": audit_info.get("topic_switch_reason") or memory_context.topic_switch_reason,
        "suggested_task_type": audit_info.get("suggested_task_type") or memory_context.topic_switch_task_type,
        "previous_task_type": audit_info.get("previous_task_type")
        or (memory_context.recent_task.task_type if memory_context.recent_task else None),
        "router_decision": audit_info.get("router_decision"),
        "route_reason": audit_info.get("route_reason"),
        "matched_rules": list(audit_info.get("matched_rules") or []),
        "safety_intent": response.safety_intent,
        "safety_severity": response.safety_severity,
        "matched_keywords": list(audit_info.get("matched_keywords") or []),
        "out_of_scope_reason": audit_info.get("out_of_scope_reason"),
        "handler_name": response.handler_name,
        "prompt_version": audit_info.get("prompt_version", AGENT_PROMPT_VERSION),
        "response_policy_version": audit_info.get("response_policy_version", RESPONSE_POLICY_VERSION),
        "safety_policy_version": audit_info.get("safety_policy_version", SAFETY_POLICY_VERSION),
        "error_type": error_type,
        "duration_ms": audit_info.get("duration_ms"),
    }
    return AgentTaskRecord(
        task_id=task_id,
        session_id=session_id,
        user_input=request.user_input,
        task_type=response.task_type,
        success=response.success,
        risk_level=response.risk_level,
        fraud_type=response.fraud_type,
        conclusion=response.conclusion,
        suggestions=list(response.suggestions),
        tool_trace=[
            AgentToolTraceRecord(
                trace_id=new_id(),
                task_id=task_id,
                session_id=session_id,
                tool_name=trace.tool_name,
                success=trace.success,
                summary=trace.summary,
                error=_service_error_to_dict(trace.error),
                duration_ms=trace.duration_ms,
                error_type=trace.error_type or (trace.error.code if trace.error else None),
                handler_name=trace.handler_name,
                metadata=dict(trace.metadata or {}),
            )
            for trace in response.tool_trace
        ],
        fallback_message=response.fallback_message,
        error=_service_error_to_dict(response.error),
        output_text=_build_assistant_message_content(response),
        answer=response.answer,
        evidence=list(response.evidence),
        metadata=metadata,
    )


def _service_error_to_dict(error: ServiceError | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return asdict(error)
