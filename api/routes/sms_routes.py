from __future__ import annotations

from fastapi import APIRouter

from api.deps import get_sms_service, service_error_to_response, to_plain_data
from api.models import SmsRecognizeApiRequest

router = APIRouter(prefix="/sms", tags=["sms"])


@router.post("/recognize")
def recognize_sms(request: SmsRecognizeApiRequest):
    response = get_sms_service()(request.text)
    if not response.success:
        return service_error_to_response(response.error, fallback_message="短信识别失败")
    return to_plain_data(response)
