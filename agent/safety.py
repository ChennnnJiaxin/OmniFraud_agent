from __future__ import annotations

from dataclasses import dataclass, field
import re

SAFETY_INTENT_NORMAL = "normal"
SAFETY_INTENT_REMEDIAL = "remedial_action"
SAFETY_INTENT_EMERGENCY = "emergency_loss"
SAFETY_INTENT_SECONDARY = "secondary_fraud"

SAFETY_ERROR_TYPE_MAP = {
    SAFETY_INTENT_REMEDIAL: "SAFETY_REMEDIAL",
    SAFETY_INTENT_EMERGENCY: "SAFETY_EMERGENCY_LOSS",
    SAFETY_INTENT_SECONDARY: "SAFETY_SECONDARY_FRAUD",
}

LOSS_KEYWORDS = (
    "被骗",
    "被诈",
    "转账了",
    "转过去了",
    "转过去",
    "汇款了",
    "付款了",
    "打款了",
    "打过去了",
    "钱转出去了",
    "钱转过去了",
    "损失了",
    "被骗走了",
    "被骗走钱",
    "交了保证金",
    "交了解冻费",
    "交了手续费",
    "交了认证费",
    "交了押金",
    "充了值",
    "给他转了",
)
LOSS_ACTION_KEYWORDS = (
    "转账",
    "汇款",
    "付款",
    "打款",
    "损失",
    "被骗",
    "被骗走",
    "保证金",
    "解冻费",
    "手续费",
    "认证费",
    "押金",
)
SECONDARY_RECOVERY_KEYWORDS = (
    "追回",
    "追回资金",
    "追回被骗的钱",
    "退钱",
    "退款",
    "退回来",
    "解冻资金",
    "解封",
    "退款到账",
    "网警",
    "律师",
    "内部渠道",
    "内部人员",
    "追回团队",
    "资金追回",
    "资金被冻结",
    "被冻结",
)
SECONDARY_PAYMENT_KEYWORDS = (
    "手续费",
    "认证费",
    "保证金",
    "解冻费",
    "先付款",
    "先交钱",
    "先交费",
    "交费",
    "刷流水",
    "解封费",
    "激活费",
)
REMEDIAL_PAST_ACTION_KEYWORDS = (
    "已经填写了",
    "已经填了",
    "已经输入了",
    "已经提交了",
    "已经点了",
    "已经下载了",
    "已经安装了",
    "已经开了",
    "已经共享了",
    "已经告诉他了",
    "已经发给他了",
    "已经扫码了",
    "已经加了好友",
    "已经泄露了",
    "已经按他说的做了",
    "已经操作了",
)
REMEDIAL_SENSITIVE_KEYWORDS = (
    "银行卡",
    "银行卡号",
    "验证码",
    "身份证号",
    "身份证",
    "密码",
    "支付密码",
    "短信验证码",
    "链接",
    "app",
    "App",
    "软件下载",
    "屏幕共享",
    "共享屏幕",
    "远程控制",
    "人脸识别",
    "资料",
    "扫码",
    "好友",
)
GENERIC_REMEDIAL_HINTS = (
    "已经这样做了",
    "已经照做了",
    "已经按他说的做了",
    "已经操作过了",
)
SENSITIVE_CONTEXT_HINTS = (
    "验证码",
    "银行卡",
    "链接",
    "app",
    "屏幕共享",
    "转账",
    "安全账户",
)
AMOUNT_PATTERNS = (
    r"(?:¥|￥|RMB|rmb)\s*\d+(?:\.\d+)?(?:万|千|百)?",
    r"\d+(?:\.\d+)?\s*(?:元|块|万|w)",
    r"(?:一|二|两|三|四|五|六|七|八|九|十|百|千|万)+(?:元|块|万)",
    r"(?:十几万|几十万|上百万|百万|几万|几千|几百)",
)


@dataclass(slots=True)
class SafetyClassification:
    safety_intent: str = SAFETY_INTENT_NORMAL
    severity: str = "low"
    matched_keywords: list[str] = field(default_factory=list)
    reason: str = ""
    handler_name: str = "default_handler"
    error_type: str | None = None


def classify_safety_intent(
    user_input: str,
    *,
    memory_summary: str | None = None,
    recent_conclusion: str | None = None,
) -> SafetyClassification:
    text = (user_input or "").strip()
    if not text:
        return SafetyClassification(reason="输入为空")

    recovery_hits = _find_keywords(text, SECONDARY_RECOVERY_KEYWORDS)
    secondary_fee_hits = _find_keywords(text, SECONDARY_PAYMENT_KEYWORDS)
    if recovery_hits and secondary_fee_hits:
        return SafetyClassification(
            safety_intent=SAFETY_INTENT_SECONDARY,
            severity="high",
            matched_keywords=_dedupe(recovery_hits + secondary_fee_hits),
            reason="用户描述的是追回资金、解冻或退款前要求先交钱的二次诈骗场景",
            handler_name="secondary_fraud_handler",
            error_type=SAFETY_ERROR_TYPE_MAP[SAFETY_INTENT_SECONDARY],
        )

    matched_amounts = _find_amounts(text)
    emergency_hits = _find_keywords(text, LOSS_KEYWORDS)
    if emergency_hits or (matched_amounts and _contains_any(text, LOSS_ACTION_KEYWORDS)):
        matched = _dedupe(emergency_hits + matched_amounts)
        severity = "critical" if matched_amounts or any(keyword in text for keyword in ("被骗", "损失")) else "high"
        return SafetyClassification(
            safety_intent=SAFETY_INTENT_EMERGENCY,
            severity=severity,
            matched_keywords=matched,
            reason="用户表示已经发生资金损失或已完成转账付款等高危操作",
            handler_name="emergency_loss_handler",
            error_type=SAFETY_ERROR_TYPE_MAP[SAFETY_INTENT_EMERGENCY],
        )

    remedial_action_hits = _find_keywords(text, REMEDIAL_PAST_ACTION_KEYWORDS)
    remedial_sensitive_hits = _find_keywords(text, REMEDIAL_SENSITIVE_KEYWORDS)
    if remedial_action_hits and remedial_sensitive_hits:
        return SafetyClassification(
            safety_intent=SAFETY_INTENT_REMEDIAL,
            severity="high",
            matched_keywords=_dedupe(remedial_action_hits + remedial_sensitive_hits),
            reason="用户已经完成敏感信息泄露、点链接、下载 App 或屏幕共享等高危操作",
            handler_name="remedial_action_handler",
            error_type=SAFETY_ERROR_TYPE_MAP[SAFETY_INTENT_REMEDIAL],
        )

    if remedial_sensitive_hits and any(keyword in text for keyword in ("已经", "告诉他", "发给他", "给他了", "开了", "点了")):
        return SafetyClassification(
            safety_intent=SAFETY_INTENT_REMEDIAL,
            severity="high",
            matched_keywords=_dedupe(remedial_sensitive_hits),
            reason="用户表达了已经执行的敏感操作，需要立即补救",
            handler_name="remedial_action_handler",
            error_type=SAFETY_ERROR_TYPE_MAP[SAFETY_INTENT_REMEDIAL],
        )

    context_text = " ".join(item for item in (memory_summary, recent_conclusion) if item)
    matched_generic_hints = [phrase for phrase in GENERIC_REMEDIAL_HINTS if phrase in text]
    if context_text and matched_generic_hints:
        context_hits = _find_keywords(context_text, SENSITIVE_CONTEXT_HINTS)
        if context_hits:
            return SafetyClassification(
                safety_intent=SAFETY_INTENT_REMEDIAL,
                severity="high",
                matched_keywords=_dedupe(matched_generic_hints + context_hits),
                reason="结合会话上下文，用户是在追问已经完成的高危操作补救措施",
                handler_name="remedial_action_handler",
                error_type=SAFETY_ERROR_TYPE_MAP[SAFETY_INTENT_REMEDIAL],
            )

    return SafetyClassification(reason="未命中高危安全兜底规则")


def _find_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _find_amounts(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in AMOUNT_PATTERNS:
        matches.extend(re.findall(pattern, text))
    return _dedupe(matches)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = (item or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered
