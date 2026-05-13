from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def new_id() -> str:
    return str(uuid4())


@dataclass(slots=True)
class AgentSession:
    session_id: str
    created_at: str
    updated_at: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentMessage:
    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentToolTraceRecord:
    trace_id: str
    task_id: str
    session_id: str
    tool_name: str
    success: bool
    summary: str
    error: dict[str, Any] | None = None
    duration_ms: int | None = None
    error_type: str | None = None
    handler_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class AgentTaskRecord:
    task_id: str
    session_id: str
    user_input: str
    task_type: str
    success: bool
    risk_level: str = "unknown"
    fraud_type: str | None = None
    conclusion: str = ""
    suggestions: list[str] = field(default_factory=list)
    tool_trace: list[AgentToolTraceRecord] = field(default_factory=list)
    fallback_message: str | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(default_factory=utc_now_iso)
    output_text: str | None = None
    answer: str | None = None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
