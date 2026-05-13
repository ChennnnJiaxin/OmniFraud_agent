from __future__ import annotations

from dataclasses import asdict, is_dataclass

from infra.config import AppConfig
from schemas.common_schema import ServiceError
from schemas.risk_schema import RiskProfileInput, RiskReportResponse


def _profile_items(profile: RiskProfileInput | dict):
    if isinstance(profile, dict):
        return profile.items()
    if is_dataclass(profile):
        return asdict(profile).items()
    return vars(profile).items()


def _build_profile_text(profile: RiskProfileInput | dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in _profile_items(profile) if value not in (None, "", [], {}))


def _infer_risk(profile: RiskProfileInput) -> tuple[int, list[str], list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    suggestions: list[str] = []
    vulnerable_types: list[str] = []

    if profile.age is not None and profile.age >= 60:
        score += 25
        reasons.append("年龄较高，可能更容易受到保健品、冒充熟人等诈骗影响。")
        vulnerable_types.append("保健品诈骗")
    if profile.occupation in {"学生", "退休人员", "自由职业"}:
        score += 15
        reasons.append(f"{profile.occupation}群体在刷单、兼职或情感类诈骗中更容易被针对。")
    if profile.loss_amount and profile.loss_amount > 0:
        score += 20
        reasons.append("近一年已有受骗损失，说明存在重复受骗风险。")
    if profile.report_police is False:
        score += 10
        reasons.append("受骗后不及时报警，可能错过止损窗口。")
    if profile.social_media_hours is not None and profile.social_media_hours >= 6:
        score += 10
        reasons.append("社交媒体暴露时间较长，接触诈骗信息的概率更高。")
    if profile.urgency_react == "立即查看":
        score += 10
        reasons.append("面对紧急通知时容易立即响应，容易被催促类话术利用。")
    if profile.stranger_request == "爽快提供":
        score += 15
        reasons.append("对陌生人索取个人信息防范不足。")
    if profile.reward_react == "积极参与":
        score += 10
        reasons.append("对高回报或优惠信息警惕性较低。")
    if profile.fraud_types:
        score += min(len(profile.fraud_types) * 5, 15)
        vulnerable_types.extend(profile.fraud_types)

    if not suggestions:
        suggestions.extend(
            [
                "涉及转账、验证码、下载 App 的请求务必二次核验。",
                "遇到紧急通知优先通过官方电话或官网确认。",
                "定期与家人沟通常见诈骗手法，建立求证习惯。",
            ]
        )
    return score, vulnerable_types[:5], reasons[:5], suggestions[:5]


def generate_risk_report(
    profile: RiskProfileInput,
    config: AppConfig | None = None,
    include_llm_report: bool = False,
) -> RiskReportResponse:
    try:
        score, vulnerable_types, reasons, suggestions = _infer_risk(profile)
        risk_level = "高风险" if score >= 60 else "中风险" if score >= 35 else "低风险"
        report = ""
        if include_llm_report:
            from clients.llm_client import LlmClient

            prompt = f"""
以下是用户填写的反诈风险画像信息：
{_build_profile_text(profile)}

请输出一份简洁、实用、约200字的风险分析报告，并给出条理清晰的防范建议。
"""
            response = LlmClient(config).complete(
                model=None,
                messages=[
                    {"role": "system", "content": "你是一名反诈宣传助手。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.8,
            )
            report = response.choices[0].message.content or ""
        return RiskReportResponse(
            success=True,
            risk_level=risk_level,
            vulnerable_fraud_types=vulnerable_types,
            reasons=reasons,
            suggestions=suggestions,
            report=report,
        )
    except Exception as exc:
        return RiskReportResponse(
            success=False,
            error=ServiceError(code="RISK_REPORT_FAILED", message="风险评估失败。", detail={"error": str(exc)}),
        )


def generate_risk_report_stream(profile: RiskProfileInput | dict, config: AppConfig | None = None):
    from clients.llm_client import LlmClient

    prompt = f"""
我这里有一个用户对风险评估问卷填写的信息，以下是信息内容：
{_build_profile_text(profile)}

请根据用户画像生成一份风险分析报告和防范建议，约200字，适合普通用户阅读。
"""
    return LlmClient(config).stream_chat(
        model=None,
        messages=[
            {"role": "system", "content": "你是一名反诈宣传助手。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=1.0,
    )
