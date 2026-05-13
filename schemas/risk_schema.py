from __future__ import annotations

from dataclasses import dataclass, field

from .common_schema import ServiceError


@dataclass(slots=True)
class RiskProfileInput:
    age: int | None = None
    occupation: str | None = None
    platforms: list[str] = field(default_factory=list)
    recent_experience: str | None = None
    risk_preference: str | None = None
    residence: str | None = None
    income: str | None = None
    payment_methods: list[str] = field(default_factory=list)
    investment_experience: str | None = None
    fraud_types: list[str] = field(default_factory=list)
    loss_amount: int | None = None
    report_police: bool | None = None
    urgency_react: str | None = None
    stranger_request: str | None = None
    reward_react: str | None = None
    social_media_hours: int | None = None


@dataclass(slots=True)
class RiskReportResponse:
    success: bool
    risk_level: str = "未知"
    vulnerable_fraud_types: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    report: str = ""
    error: ServiceError | None = None
