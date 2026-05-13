from __future__ import annotations

from fastapi import APIRouter

from api.deps import get_risk_service, service_error_to_response, to_plain_data
from api.models import RiskAssessmentApiRequest
from schemas.risk_schema import RiskProfileInput

router = APIRouter(prefix="/risk-assessment", tags=["risk-assessment"])


@router.post("/report")
def generate_risk_report(request: RiskAssessmentApiRequest):
    profile = RiskProfileInput(**request.model_dump())
    response = get_risk_service()(profile)
    if not response.success:
        return service_error_to_response(response.error, fallback_message="风险评估失败")
    return to_plain_data(response)
