from __future__ import annotations

from fastapi import APIRouter

from agent.agent_service import run_agent
from agent.models import AgentRunRequest
from api.deps import to_plain_data
from api.models import AgentRunApiRequest

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run")
def run_agent_route(request: AgentRunApiRequest):
    response = run_agent(
        AgentRunRequest(
            user_input=request.user_input,
            session_id=request.session_id,
            input_type=request.input_type,
            image=request.image,
            profile=request.profile,
            options=request.options.model_dump(),
        )
    )
    return to_plain_data(response)
