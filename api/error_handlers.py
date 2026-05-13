from __future__ import annotations

from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.deps import build_error_payload
from infra.logging import get_logger

logger = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=build_error_payload(
                code="INVALID_INPUT",
                message="请求参数不合法",
                detail={"errors": jsonable_encoder(exc.errors())},
            ),
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        message = detail.get("message") if isinstance(detail, dict) else None
        code = detail.get("code") if isinstance(detail, dict) else None
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(
                code=code or "INTERNAL_ERROR",
                message=message or "请求处理失败",
                detail=detail if isinstance(detail, dict) else {},
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content=build_error_payload(
                code="INTERNAL_ERROR",
                message="服务内部错误",
                detail={},
            ),
        )
