from __future__ import annotations

from fastapi import APIRouter

from api.deps import get_qa_service, service_error_to_response, to_plain_data
from api.models import QaChatApiRequest

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/chat")
def chat(request: QaChatApiRequest):
    response = get_qa_service()(question=request.question, context=request.context)
    if not response.success:
        return service_error_to_response(response.error, fallback_message="问答服务暂时不可用")
    return to_plain_data(response)
