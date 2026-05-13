from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from api.deps import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE


class SmsRecognizeApiRequest(BaseModel):
    text: str = Field(..., description="待识别短信文本")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("输入不能为空")
        return normalized


class QaChatApiRequest(BaseModel):
    question: str = Field(..., description="反诈问答问题")
    context: dict[str, Any] | None = Field(default=None, description="可选上下文")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized


class CaseSearchApiRequest(BaseModel):
    query: str = Field(..., description="案例搜索关键词")
    fraud_type: str | None = Field(default=None, description="诈骗类型过滤")
    limit: int | None = Field(default=None, ge=1, le=MAX_PAGE_SIZE, description="兼容旧参数，等价于 page_size")
    page: int = Field(default=1, ge=1, description="页码，从 1 开始")
    page_size: int | None = Field(default=None, ge=1, le=MAX_PAGE_SIZE, description="每页数量")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("查询关键词不能为空")
        return normalized

    @property
    def resolved_page_size(self) -> int:
        return self.page_size or self.limit or DEFAULT_PAGE_SIZE


class RiskAssessmentApiRequest(BaseModel):
    age: int | None = Field(default=None, ge=0, le=150)
    occupation: str | None = None
    platforms: list[str] = Field(default_factory=list)
    recent_experience: str | None = None
    risk_preference: str | None = None
    residence: str | None = None
    income: str | None = None
    payment_methods: list[str] = Field(default_factory=list)
    investment_experience: str | None = None
    fraud_types: list[str] = Field(default_factory=list)
    loss_amount: int | None = Field(default=None, ge=0)
    report_police: bool | None = None
    urgency_react: str | None = None
    stranger_request: str | None = None
    reward_react: str | None = None
    social_media_hours: int | None = Field(default=None, ge=0, le=24)

    @field_validator(
        "occupation",
        "recent_experience",
        "risk_preference",
        "residence",
        "income",
        "investment_experience",
        "urgency_react",
        "stranger_request",
        "reward_react",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("platforms", "payment_methods", "fraud_types")
    @classmethod
    def normalize_string_list(cls, values: list[str]) -> list[str]:
        return [item.strip() for item in values if isinstance(item, str) and item.strip()]

    @model_validator(mode="after")
    def validate_minimum_profile(self) -> "RiskAssessmentApiRequest":
        has_core_profile = any(
            [
                self.age is not None,
                bool(self.occupation),
                bool(self.platforms),
                bool(self.recent_experience),
            ]
        )
        if not has_core_profile:
            raise ValueError("至少提供 age、occupation、platforms、recent_experience 中的一项")
        return self


class AgentRunOptionsApiRequest(BaseModel):
    return_trace: bool = Field(default=True)
    case_limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=10)
    use_memory: bool = Field(default=True)
    history_limit: int = Field(default=6, ge=1, le=20)


class AgentRunApiRequest(BaseModel):
    user_input: str = Field(..., description="用户自然语言输入")
    session_id: str | None = Field(default=None, description="可选会话 ID")
    input_type: str = Field(default="text", description="text/image/profile/mixed")
    image: str | None = Field(default=None, description="可选图片路径或 base64")
    profile: dict[str, Any] | None = Field(default=None, description="可选用户画像信息")
    options: AgentRunOptionsApiRequest = Field(default_factory=AgentRunOptionsApiRequest)

    @field_validator("user_input")
    @classmethod
    def validate_user_input(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("user_input 不能为空")
        return normalized

    @field_validator("input_type")
    @classmethod
    def normalize_input_type(cls, value: str) -> str:
        normalized = (value or "text").strip().lower()
        if normalized not in {"text", "image", "profile", "mixed"}:
            raise ValueError("input_type 仅支持 text、image、profile、mixed")
        return normalized


class AgentSessionCreateApiRequest(BaseModel):
    title: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
