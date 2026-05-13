from __future__ import annotations

from typing import Any

from agent.models import AgentRelatedCase, AgentRunResponse, AgentToolResult, AgentToolTrace, TASK_OUT_OF_SCOPE, TASK_UNKNOWN
from agent.safety import (
    SAFETY_INTENT_EMERGENCY,
    SAFETY_INTENT_REMEDIAL,
    SAFETY_INTENT_SECONDARY,
    SafetyClassification,
)
from schemas.common_schema import ServiceError


def build_tool_trace(results: list[AgentToolResult], include_trace: bool) -> list[AgentToolTrace]:
    if not include_trace:
        return []
    return [
        AgentToolTrace(
            tool_name=result.tool_name,
            success=result.success,
            summary=result.summary,
            error=result.error,
            duration_ms=result.duration_ms,
            error_type=result.error_type,
            handler_name=result.handler_name,
            metadata=dict(result.metadata),
        )
        for result in results
    ]


def build_related_cases(case_result: AgentToolResult | None) -> list[AgentRelatedCase]:
    if case_result is None or not case_result.success:
        return []
    cases = (case_result.data or {}).get("cases", [])
    return [
        AgentRelatedCase(
            title=case.get("title", ""),
            summary=case.get("summary", ""),
            source=case.get("source"),
        )
        for case in cases
    ]


def dedupe_suggestions(*suggestion_groups: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in suggestion_groups:
        for item in group:
            normalized = (item or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
    return ordered


def build_unknown_response(message: str, include_trace: bool = True) -> AgentRunResponse:
    return AgentRunResponse(
        success=False,
        task_type=TASK_UNKNOWN,
        conclusion="暂时无法判断该任务类型",
        evidence=[],
        suggestions=[
            "请补充短信原文、截图文字、对方要求你做什么，或是否涉及转账、验证码、链接、下载 App 等。",
            "如果涉及转账、验证码或陌生链接，请先暂停操作。",
        ],
        tool_trace=[] if not include_trace else [],
        fallback_message=message,
        error=ServiceError(code="UNKNOWN_TASK", message="无法识别用户任务类型"),
        handler_name="unknown_handler",
        audit_info={"error_type": "ROUTER_UNKNOWN"},
    )


def build_out_of_scope_response(include_trace: bool = True) -> AgentRunResponse:
    return AgentRunResponse(
        success=True,
        task_type=TASK_OUT_OF_SCOPE,
        conclusion="你刚才的问题和反诈任务关系不大。",
        risk_level="unknown",
        suggestions=[
            "帮我看看这条短信是不是诈骗：……",
            "有人让我转账、交保证金或给验证码，这靠谱吗？",
            "我爸妈经常用微信和抖音，容易遇到什么骗局？",
            "我已经转账了怎么办？",
            "有人说能帮我追回被骗的钱，但要先交手续费。",
        ],
        answer=(
            "我是反诈助手，主要帮助你识别诈骗风险、查询诈骗案例、分析用户风险画像，"
            "并在被骗或泄露信息后给出补救建议。\n\n"
            "你可以这样问我：\n"
            "1. 帮我看看这条短信是不是诈骗：……\n"
            "2. 有人让我转账、交保证金或给验证码，这靠谱吗？\n"
            "3. 我爸妈经常用微信和抖音，容易遇到什么骗局？\n"
            "4. 我已经转账了怎么办？\n"
            "5. 有人说能帮我追回被骗的钱，但要先交手续费。"
        ),
        tool_trace=[] if not include_trace else [],
        handler_name="out_of_scope_handler",
        audit_info={"error_type": "OUT_OF_SCOPE"},
    )


def graph_payload(graph_result: AgentToolResult | None) -> dict[str, Any] | None:
    if graph_result is None or not graph_result.success:
        return None
    return graph_result.data


def build_safety_response(
    *,
    safety: SafetyClassification,
    task_type: str,
    include_trace: bool,
    context_used: bool,
) -> AgentRunResponse:
    suggestions, conclusion, answer = _build_safety_content(safety)
    return AgentRunResponse(
        success=True,
        task_type=task_type,
        conclusion=conclusion,
        risk_level=safety.severity,
        suggestions=suggestions,
        answer=answer,
        tool_trace=[] if not include_trace else [],
        safety_intent=safety.safety_intent,
        safety_severity=safety.severity,
        handler_name=safety.handler_name,
        audit_info={
            "safety_intent": safety.safety_intent,
            "safety_severity": safety.severity,
            "matched_keywords": list(safety.matched_keywords),
            "handler_name": safety.handler_name,
            "error_type": safety.error_type,
            "context_used": context_used,
        },
    )


def _build_safety_content(safety: SafetyClassification) -> tuple[list[str], str, str]:
    if safety.safety_intent == SAFETY_INTENT_REMEDIAL:
        suggestions = [
            "立即停止继续操作，不要再点击链接、共享屏幕或继续向对方提供任何验证码和密码。",
            "立即联系银行或支付平台，申请冻结、挂失、限制账户或关闭快捷支付。",
            "尽快修改银行、支付、电商、邮箱等重要账户密码，并检查是否新增陌生收款人或授权设备。",
            "如果已经泄露验证码或发生转账，立即拨打 96110 咨询反诈中心，并同步报警。",
            "保留短信、聊天记录、链接、App、电话号码、下载记录和操作截图等证据。",
            "警惕后续冒充客服、网警、退款专员或资金追回团队的二次诈骗。",
        ]
        conclusion = "你已经进行了高风险操作，存在账户被盗刷、隐私泄露或被进一步诱导转账的风险，应立即停止操作并采取补救措施。"
        answer = "这类情况需要立刻止损处理。现在最重要的是先中断与对方的继续接触，再联系银行或支付平台做冻结、挂失、限制账户，同时修改重要账户密码并保留证据；如果验证码已泄露或已经发生转账，要尽快拨打 96110 并报警。"
        return suggestions, conclusion, answer

    if safety.safety_intent == SAFETY_INTENT_EMERGENCY:
        suggestions = [
            "立即停止继续转账或继续与对方沟通，不要再相信任何补缴费用、刷流水或解冻资金说法。",
            "立即拨打 110，或尽快到就近派出所报警。",
            "尽快拨打 96110，联系反诈中心咨询止损和取证处理。",
            "立即联系银行或支付平台，申请止付、冻结、挂失或交易争议处理。",
            "保存聊天记录、转账记录、收款账户、手机号、链接、App、通话记录和相关截图等证据。",
            "不要相信任何“追回资金”“内部渠道”“网警退款”“律师代追回”且要求先交钱的说法。",
            "通知家人或可信任的人一起处理，避免继续被诱导操作。",
        ]
        conclusion = "你描述的情况属于已发生资金损失的高危诈骗场景，应立即采取止损、报警和证据保全措施。"
        answer = "这属于需要立刻应急处置的高危场景。请马上停止继续转账和沟通，第一时间报警并联系银行或支付平台尝试止付、冻结和挂失，同时完整保存证据；后续凡是声称能追回资金但要求先交钱的，基本都要按二次诈骗处理。"
        return suggestions, conclusion, answer

    if safety.safety_intent == SAFETY_INTENT_SECONDARY:
        suggestions = [
            "不要再转账，也不要再支付任何手续费、保证金、认证费、解冻费或刷流水费用。",
            "不要相信所谓网警、律师、内部人员或资金追回团队的私下联系。",
            "通过 110、96110、银行官方客服电话或平台官方客服渠道核实信息。",
            "保留对方联系方式、聊天记录、付款要求、收款账户和相关截图。",
            "如果已经付款，立即报警，并联系支付平台或银行尝试止付。",
        ]
        conclusion = "这很可能是针对受害者的二次诈骗。真正的警方、银行或正规平台不会要求你先交手续费、保证金或认证费来追回资金。"
        answer = "你现在遇到的更像是“追回资金”名义下的二次诈骗。正规警方、银行和平台不会通过私人聊天让你先交钱才能退款、解冻或追回损失；所谓网警、律师、内部渠道或平台内部人员要求先付款的说法都要高度警惕。请立刻停止付款，通过官方渠道核实，并保留全部证据，如已付款要马上报警和联系支付机构止损。"
        return suggestions, conclusion, answer

    return [], "未命中安全响应", ""
