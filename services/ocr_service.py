from __future__ import annotations

import base64
from pathlib import Path

from clients.llm_client import LlmClient
from infra.config import AppConfig
from schemas.common_schema import ServiceError
from schemas.ocr_schema import OcrResponse


def _resolve_image_input(image_input) -> tuple[bytes, str]:
    if hasattr(image_input, "getvalue"):
        mime_type = getattr(image_input, "type", None) or "image/png"
        return image_input.getvalue(), mime_type

    if isinstance(image_input, bytes):
        return image_input, "image/png"

    if isinstance(image_input, str):
        if image_input.startswith("data:image"):
            header, encoded = image_input.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            return base64.b64decode(encoded), mime_type
        path = Path(image_input)
        return path.read_bytes(), "image/png"

    raise TypeError("不支持的图片输入类型。")


def extract_text_from_image(image_input, config: AppConfig | None = None) -> OcrResponse:
    try:
        image_bytes, mime_type = _resolve_image_input(image_input)
        text = LlmClient(config).extract_text_from_image_bytes(image_bytes, mime_type=mime_type)
        return OcrResponse(success=True, text=text, confidence=None)
    except Exception as exc:
        return OcrResponse(
            success=False,
            error=ServiceError(code="OCR_FAILED", message="图片文字提取失败。", detail={"error": str(exc)}),
        )
