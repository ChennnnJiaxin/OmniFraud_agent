from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas.common_schema import ServiceError

TASK_TEXT_RISK = "text_risk_analysis"
TASK_IMAGE_RISK = "image_risk_analysis"
TASK_CASE_SUMMARY = "case_summary"
TASK_USER_RISK_PROFILE = "user_risk_profile"
TASK_OUT_OF_SCOPE = "out_of_scope"
TASK_UNKNOWN = "unknown"


@dataclass(slots=True)
class AgentRunOptions:
    return_trace: bool = True
    case_limit: int = 5
    use_memory: bool = True
    history_limit: int = 6

    def __post_init__(self) -> None:
        self.case_limit = max(1, min(int(self.case_limit or 5), 10))
        self.history_limit = max(1, min(int(self.history_limit or 6), 20))


@dataclass(slots=True)
class AgentRunRequest:
    user_input: str
    session_id: str | None = None
    input_type: str = "text"
    image: Any | None = None
    profile: dict[str, Any] | None = None
    options: AgentRunOptions = field(default_factory=AgentRunOptions)

    def __post_init__(self) -> None:
        self.user_input = (self.user_input or "").strip()
        self.input_type = (self.input_type or "text").strip().lower()
        if isinstance(self.options, dict):
            self.options = AgentRunOptions(**self.options)


@dataclass(slots=True)
class AgentRouteDecision:
    task_type: str
    reason: str
    matched_rules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OutOfScopeResult:
    is_out_of_scope: bool
    reason: str | None = None
    matched_keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentToolResult:
    tool_name: str
    success: bool
    data: dict[str, Any] | None = None
    error: ServiceError | None = None
    summary: str = ""
    duration_ms: int | None = None
    error_type: str | None = None
    handler_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentToolTrace:
    tool_name: str
    success: bool
    summary: str
    error: ServiceError | None = None
    duration_ms: int | None = None
    error_type: str | None = None
    handler_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRelatedCase:
    title: str
    summary: str
    source: str | None = None


@dataclass(slots=True)
class AgentRunResponse:
    success: bool
    task_type: str
    conclusion: str
    session_id: str | None = None
    task_id: str | None = None
    context_used: bool = False
    memory_summary: str | None = None
    risk_level: str = "unknown"
    fraud_type: str | None = None
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    related_cases: list[AgentRelatedCase] = field(default_factory=list)
    graph_result: dict[str, Any] | None = None
    answer: str | None = None
    vulnerable_fraud_types: list[str] = field(default_factory=list)
    tool_trace: list[AgentToolTrace] = field(default_factory=list)
    fallback_message: str | None = None
    error: ServiceError | None = None
    safety_intent: str = "normal"
    safety_severity: str | None = None
    handler_name: str | None = None
    audit_info: dict[str, Any] = field(default_factory=dict)
