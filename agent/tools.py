from __future__ import annotations

from dataclasses import asdict, is_dataclass
from time import perf_counter
from typing import Any

from agent.models import AgentToolResult
from schemas.common_schema import ServiceError


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    return value


def _service_error(error: ServiceError | None, fallback_code: str, fallback_message: str) -> ServiceError:
    if error is not None:
        return error
    return ServiceError(code=fallback_code, message=fallback_message)


def _duration_ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)


def _result(
    *,
    tool_name: str,
    success: bool,
    summary: str,
    start: float,
    data: dict[str, Any] | None = None,
    error: ServiceError | None = None,
) -> AgentToolResult:
    return AgentToolResult(
        tool_name=tool_name,
        success=success,
        data=data,
        error=error,
        summary=summary,
        duration_ms=_duration_ms(start),
        error_type=error.code if error else None,
    )


def sms_recognize_tool(text: str) -> AgentToolResult:
    start = perf_counter()
    try:
        from services.sms_service import recognize_sms

        response = recognize_sms(text)
        if response.success:
            return _result(
                tool_name="sms_recognize",
                success=True,
                summary=f"完成短信风险识别，风险等级：{response.risk_level}",
                start=start,
                data=_to_plain_data(response),
            )
        error = _service_error(response.error, "SMS_RECOGNIZE_FAILED", "短信识别服务调用失败")
        return _result(tool_name="sms_recognize", success=False, summary="短信识别失败", start=start, error=error)
    except Exception as exc:
        error = ServiceError(code="SERVICE_ERROR", message="短信识别服务调用失败", detail={"error": str(exc)})
        return _result(tool_name="sms_recognize", success=False, summary="短信识别失败", start=start, error=error)


def case_search_tool(query: str, fraud_type: str | None = None, limit: int = 5) -> AgentToolResult:
    start = perf_counter()
    try:
        from services.case_service import search_cases

        response = search_cases(query=query, fraud_type=fraud_type, limit=limit)
        if response.success:
            total = len(response.cases)
            return _result(
                tool_name="case_search",
                success=True,
                summary=f"检索到 {total} 条相关案例",
                start=start,
                data=_to_plain_data(response),
            )
        error = _service_error(response.error, "CASE_SEARCH_FAILED", "案例搜索服务调用失败")
        return _result(tool_name="case_search", success=False, summary="案例搜索失败", start=start, error=error)
    except Exception as exc:
        error = ServiceError(code="SERVICE_ERROR", message="案例搜索服务调用失败", detail={"error": str(exc)})
        return _result(tool_name="case_search", success=False, summary="案例搜索失败", start=start, error=error)


def qa_chat_tool(question: str, context: dict[str, Any] | None = None) -> AgentToolResult:
    start = perf_counter()
    try:
        from services.qa_service import chat_with_anti_fraud_bot

        response = chat_with_anti_fraud_bot(question=question, context=context)
        if response.success:
            return _result(
                tool_name="qa_chat",
                success=True,
                summary="问答服务生成了补充建议",
                start=start,
                data=_to_plain_data(response),
            )
        error = _service_error(response.error, "QA_FAILED", "问答服务调用失败")
        return _result(tool_name="qa_chat", success=False, summary="问答服务调用失败", start=start, error=error)
    except Exception as exc:
        error = ServiceError(code="SERVICE_ERROR", message="问答服务调用失败", detail={"error": str(exc)})
        return _result(tool_name="qa_chat", success=False, summary="问答服务调用失败", start=start, error=error)


def risk_report_tool(profile: dict[str, Any]) -> AgentToolResult:
    start = perf_counter()
    try:
        from schemas.risk_schema import RiskProfileInput
        from services.risk_service import generate_risk_report

        response = generate_risk_report(RiskProfileInput(**profile))
        if response.success:
            return _result(
                tool_name="risk_report",
                success=True,
                summary=f"完成人群风险评估，风险等级：{response.risk_level}",
                start=start,
                data=_to_plain_data(response),
            )
        error = _service_error(response.error, "RISK_REPORT_FAILED", "风险评估服务调用失败")
        return _result(tool_name="risk_report", success=False, summary="风险评估失败", start=start, error=error)
    except Exception as exc:
        error = ServiceError(code="SERVICE_ERROR", message="风险评估服务调用失败", detail={"error": str(exc)})
        return _result(tool_name="risk_report", success=False, summary="风险评估失败", start=start, error=error)


def graph_query_tool(query: str) -> AgentToolResult:
    start = perf_counter()
    try:
        from services.graph_service import query_fraud_graph
    except Exception:
        error = ServiceError(code="SERVICE_UNAVAILABLE", message="当前项目未启用图谱查询服务")
        return _result(tool_name="graph_query", success=False, summary="图谱查询不可用", start=start, error=error)

    try:
        response = query_fraud_graph(query)
        if response.success:
            return _result(
                tool_name="graph_query",
                success=True,
                summary="完成图谱查询",
                start=start,
                data=_to_plain_data(response),
            )
        error = _service_error(response.error, "GRAPH_QUERY_FAILED", "图谱查询失败")
        return _result(tool_name="graph_query", success=False, summary="图谱查询失败", start=start, error=error)
    except Exception as exc:
        error = ServiceError(code="SERVICE_ERROR", message="图谱查询失败", detail={"error": str(exc)})
        return _result(tool_name="graph_query", success=False, summary="图谱查询失败", start=start, error=error)


def ocr_tool(image_input: Any) -> AgentToolResult:
    start = perf_counter()
    try:
        from services.ocr_service import extract_text_from_image
    except Exception:
        error = ServiceError(code="SERVICE_UNAVAILABLE", message="当前项目未启用 OCR 服务")
        return _result(tool_name="ocr", success=False, summary="OCR 服务不可用", start=start, error=error)

    try:
        response = extract_text_from_image(image_input)
        if response.success:
            text = (response.text or "").strip()
            summary = "OCR 成功提取图片文本" if text else "OCR 执行完成，但未提取到有效文本"
            return _result(
                tool_name="ocr",
                success=True,
                summary=summary,
                start=start,
                data=_to_plain_data(response),
            )
        error = _service_error(response.error, "OCR_FAILED", "OCR 服务调用失败")
        return _result(tool_name="ocr", success=False, summary="OCR 识别失败", start=start, error=error)
    except Exception as exc:
        error = ServiceError(code="SERVICE_ERROR", message="OCR 服务调用失败", detail={"error": str(exc)})
        return _result(tool_name="ocr", success=False, summary="OCR 识别失败", start=start, error=error)
