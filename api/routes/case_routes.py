from __future__ import annotations

from fastapi import APIRouter

from api.deps import get_case_service, service_error_to_response, to_plain_data
from api.models import CaseSearchApiRequest

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("/search")
def search_cases(request: CaseSearchApiRequest):
    page_size = request.resolved_page_size
    skip = (request.page - 1) * page_size
    response = get_case_service()(
        query=request.query,
        fraud_type=request.fraud_type,
        limit=page_size,
        skip=skip,
    )
    if not response.success:
        return service_error_to_response(response.error, fallback_message="案例检索失败")
    return {
        "success": True,
        "cases": to_plain_data(response.cases),
        "total": response.total_count,
        "page": request.page,
        "page_size": page_size,
        "error": None,
    }
