from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from fastapi.responses import JSONResponse

from schemas.common_schema import ServiceError

DEFAULT_PAGE_SIZE = 5
MAX_PAGE_SIZE = 50

_SERVICE_ERROR_MAP: dict[str, tuple[int, str]] = {
    "EMPTY_TEXT": (400, "INVALID_INPUT"),
    "TEXT_TOO_SHORT": (400, "INVALID_INPUT"),
    "EMPTY_QUESTION": (400, "INVALID_INPUT"),
    "EMPTY_QUERY": (400, "INVALID_INPUT"),
    "CASE_NOT_FOUND": (404, "NOT_FOUND"),
    "SMS_RECOGNIZE_FAILED": (500, "SERVICE_ERROR"),
    "QA_FAILED": (502, "LLM_ERROR"),
    "CASE_SEARCH_FAILED": (500, "DATABASE_ERROR"),
    "GRAPH_QUERY_FAILED": (500, "DATABASE_ERROR"),
    "GRAPH_DATA_FAILED": (500, "DATABASE_ERROR"),
    "RISK_REPORT_FAILED": (500, "SERVICE_ERROR"),
}


def get_sms_service() -> Callable[..., Any]:
    from services.sms_service import recognize_sms

    return recognize_sms


def get_qa_service() -> Callable[..., Any]:
    from services.qa_service import chat_with_anti_fraud_bot

    return chat_with_anti_fraud_bot


def get_case_service() -> Callable[..., Any]:
    from services.case_service import search_cases

    return search_cases


def get_risk_service() -> Callable[..., Any]:
    from services.risk_service import generate_risk_report

    return generate_risk_report


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    return value


def build_error_payload(code: str, message: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "detail": detail or {},
        },
    }


def make_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=build_error_payload(code=code, message=message, detail=detail),
    )


def service_error_to_response(
    error: ServiceError | None,
    *,
    fallback_code: str = "SERVICE_ERROR",
    fallback_message: str = "服务调用失败",
) -> JSONResponse:
    if error is None:
        return make_error_response(status_code=500, code=fallback_code, message=fallback_message)

    status_code, api_code = _SERVICE_ERROR_MAP.get(error.code, (500, fallback_code))
    return make_error_response(
        status_code=status_code,
        code=api_code,
        message=error.message or fallback_message,
        detail=to_plain_data(error.detail) if error.detail else {},
    )
