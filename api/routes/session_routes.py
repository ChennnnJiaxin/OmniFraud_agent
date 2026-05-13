from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.agent_service import create_agent_session, get_agent_session, list_agent_session_messages, list_agent_session_tasks
from agent.session_store import get_agent_session_store
from api.deps import to_plain_data
from api.models import AgentSessionCreateApiRequest

router = APIRouter(prefix="/agent/sessions", tags=["agent-sessions"])


@router.post("")
def create_session_route(request: AgentSessionCreateApiRequest):
    session = create_agent_session(title=request.title, metadata=request.metadata)
    return {
        "success": True,
        "session_id": session.session_id,
        "session": to_plain_data(session),
    }


@router.get("")
def list_sessions_route(limit: int = 20):
    sessions = get_agent_session_store().list_sessions(limit=limit)
    return {
        "success": True,
        "sessions": to_plain_data(sessions),
    }


@router.get("/{session_id}")
def get_session_route(session_id: str):
    session = get_agent_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={"code": "SESSION_NOT_FOUND", "message": "会话不存在"})
    return {
        "success": True,
        "session": to_plain_data(session),
    }


@router.get("/{session_id}/messages")
def list_messages_route(session_id: str, limit: int = 20):
    session = get_agent_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={"code": "SESSION_NOT_FOUND", "message": "会话不存在"})
    messages = list_agent_session_messages(session_id, limit=limit)
    return {
        "success": True,
        "messages": to_plain_data(messages),
    }


@router.get("/{session_id}/tasks")
def list_tasks_route(session_id: str, limit: int = 20):
    session = get_agent_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={"code": "SESSION_NOT_FOUND", "message": "会话不存在"})
    tasks = list_agent_session_tasks(session_id, limit=limit)
    return {
        "success": True,
        "tasks": to_plain_data(tasks),
    }
