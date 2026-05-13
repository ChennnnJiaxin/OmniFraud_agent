from __future__ import annotations

from dataclasses import dataclass, field
import re

from agent.session_models import AgentMessage, AgentTaskRecord
from agent.session_store import AgentSessionStore
from agent.models import TASK_CASE_SUMMARY, TASK_TEXT_RISK, TASK_USER_RISK_PROFILE

FOLLOW_UP_TYPE_REMEDIAL = "remedial_action"
FOLLOW_UP_TYPE_EXPLANATION = "explanation"
FOLLOW_UP_TYPE_NEXT_STEP = "next_step"
FOLLOW_UP_TYPE_PROCEDURE = "procedure_question"

NEXT_STEP_KEYWORDS = (
    "怎么办",
    "怎么处理",
    "怎么做",
    "还需要做什么",
    "然后呢",
    "接下来",
    "那我现在",
    "那这个呢",
    "继续",
    "会有什么后果",
)
EXPLANATION_KEYWORDS = (
    "为什么",
    "哪里可疑",
    "哪儿可疑",
    "为什么是诈骗",
    "为什么这是诈骗",
    "有什么问题",
    "依据是什么",
)
REMEDIAL_ACTION_KEYWORDS = (
    "已经填了",
    "已经填写了",
    "已经输入了",
    "已经提交了",
    "已经转账了",
    "已经付款了",
    "已经汇款了",
    "已经扫码了",
    "已经点了链接",
    "已经下载了",
    "已经加了好友",
    "已经给了",
    "已经发了",
    "已经泄露了",
    "填了银行卡",
    "填写银行卡",
    "输入银行卡",
    "银行卡号",
    "验证码",
    "身份证号",
    "密码",
    "支付密码",
    "人脸识别",
    "屏幕共享",
    "安全账户",
)
SENSITIVE_REMEDIAL_KEYWORDS = (
    "银行卡",
    "银行卡号",
    "验证码",
    "身份证号",
    "密码",
    "支付密码",
    "转账",
    "付款",
    "汇款",
    "共享屏幕",
    "屏幕共享",
    "人脸识别",
    "安全账户",
    "链接",
    "好友",
    "app",
    "下载",
)
PAST_ACTION_HINTS = ("已经", "填了", "填写了", "输入了", "提交了", "转账了", "付款了", "汇款了", "泄露了", "点了", "下载了", "发了")
SHORT_FOLLOW_UP_PREFIXES = ("那", "还", "要不要", "能不能", "是否", "需不需要", "需要", "怎么", "为什么")
PROCEDURE_ALARM_KEYWORDS = (
    "报警电话是什么",
    "报警电话是多少",
    "报警打什么电话",
    "打哪个电话报警",
    "怎么报警",
    "如何报警",
    "要不要报警",
    "需要报警吗",
    "报警有用吗",
)
PROCEDURE_96110_KEYWORDS = (
    "96110是什么",
    "96110 是什么",
    "96110是报警电话吗",
    "96110 是报警电话吗",
    "96110能干什么",
    "96110 能干什么",
    "96110要不要打",
    "96110 要不要打",
    "反诈电话是什么",
    "反诈中心电话是什么",
    "反诈中心怎么联系",
)
PROCEDURE_BANK_KEYWORDS = (
    "银行电话是什么",
    "怎么冻结银行卡",
    "怎么挂失银行卡",
    "怎么止付",
    "怎么联系银行",
    "支付宝怎么冻结",
    "微信支付怎么冻结",
    "怎么冻结账户",
)
PROCEDURE_EVIDENCE_KEYWORDS = (
    "要保留什么证据",
    "证据怎么保存",
    "聊天记录要不要截图",
    "转账记录要不要保存",
)
PROFILE_TOPIC_KEYWORDS = (
    "我爸妈",
    "父母",
    "老人",
    "老年人",
    "60多岁",
    "70多岁",
    "退休",
    "退休人员",
    "大学生",
    "学生",
    "中学生",
    "用户画像",
    "风险画像",
    "容易遇到什么骗局",
    "容易被骗",
    "帮我分析风险",
    "适合给什么建议",
)
PROFILE_PLATFORM_KEYWORDS = ("微信", "抖音", "网购", "直播", "短视频", "快手", "小红书", "QQ")
CASE_TOPIC_KEYWORDS = (
    "有什么特征",
    "有哪些套路",
    "类似案例",
    "典型案例",
    "杀猪盘",
    "刷单返利",
    "冒充客服退款",
    "虚假投资理财",
    "总结一下",
    "总结",
    "共性",
)
TEXT_TOPIC_KEYWORDS = (
    "帮我看看这条短信",
    "帮我看看这条新短信",
    "帮我看看这段短信",
    "这条短信是不是诈骗",
    "新短信",
    "对方发来",
    "下面这段",
    "这段话是不是诈骗",
    "这段内容是不是诈骗",
)
TEXT_RISK_HINT_KEYWORDS = (
    "短信",
    "链接",
    "验证码",
    "转账",
    "客服",
    "退款",
    "账户异常",
    "点击链接",
    "银行卡",
    "安全账户",
    "下载app",
    "下载 App",
)
TOPIC_SWITCH_OPENERS = ("另外", "换个问题", "还有一个问题", "再问一个", "我想问", "顺便问一下", "另一个场景", "那我爸妈", "那如果是我父母")


@dataclass(slots=True)
class FollowUpClassification:
    is_follow_up: bool
    follow_up_type: str | None = None
    matched_keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentMemoryContext:
    session_id: str
    context_used: bool = False
    follow_up_detected: bool = False
    follow_up_type: str | None = None
    follow_up_candidate_type: str | None = None
    topic_switch_detected: bool = False
    topic_switch_reason: str | None = None
    topic_switch_task_type: str | None = None
    topic_switch_keywords: list[str] = field(default_factory=list)
    memory_summary: str | None = None
    recent_messages: list[AgentMessage] = field(default_factory=list)
    recent_task: AgentTaskRecord | None = None


@dataclass(slots=True)
class TopicSwitchResult:
    is_topic_switch: bool
    reason: str | None = None
    suggested_task_type: str | None = None
    matched_keywords: list[str] = field(default_factory=list)


def _contains_any(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def _extract_prefixed_payload(text: str) -> tuple[str, str] | None:
    for separator in ("：", ":"):
        if separator not in text:
            continue
        prefix, suffix = text.split(separator, 1)
        prefix = prefix.strip()
        suffix = suffix.strip()
        if prefix and suffix and len(prefix) <= 20 and len(suffix) >= 8:
            return prefix, suffix
    return None


def _looks_like_new_text_analysis(text: str) -> tuple[bool, list[str]]:
    matched_keywords = _contains_any(text, TEXT_TOPIC_KEYWORDS)
    payload = _extract_prefixed_payload(text)
    if matched_keywords:
        return True, matched_keywords

    if "是不是诈骗" in text and any(keyword in text for keyword in ("短信", "消息", "内容", "文本")):
        return True, ["是不是诈骗"]

    if payload is None:
        return False, []

    _, suffix = payload
    suffix_hits = _contains_any(suffix, TEXT_RISK_HINT_KEYWORDS)
    has_link = bool(re.search(r"(https?://|www\.)", suffix, flags=re.IGNORECASE))
    if suffix_hits or has_link:
        return True, suffix_hits or ["new_text_payload"]
    return False, []


def detect_topic_switch(
    user_input: str,
    memory_context: AgentMemoryContext | None = None,
) -> TopicSwitchResult:
    text = (user_input or "").strip()
    if not text or memory_context is None or memory_context.recent_task is None:
        return TopicSwitchResult(is_topic_switch=False)

    profile_hits = _contains_any(text, PROFILE_TOPIC_KEYWORDS)
    platform_hits = _contains_any(text, PROFILE_PLATFORM_KEYWORDS)
    has_profile_demographic = any(
        keyword in text for keyword in ("我爸妈", "父母", "老人", "老年人", "60多岁", "70多岁", "退休", "大学生", "学生", "中学生")
    )
    if profile_hits or (has_profile_demographic and platform_hits):
        matched_keywords = profile_hits or platform_hits
        return TopicSwitchResult(
            is_topic_switch=True,
            reason="用户输入包含新的用户画像风险分析意图",
            suggested_task_type=TASK_USER_RISK_PROFILE,
            matched_keywords=matched_keywords,
        )

    text_switch, text_hits = _looks_like_new_text_analysis(text)
    if text_switch:
        return TopicSwitchResult(
            is_topic_switch=True,
            reason="用户输入包含新的短信或文本风险分析任务",
            suggested_task_type=TASK_TEXT_RISK,
            matched_keywords=text_hits,
        )

    case_hits = _contains_any(text, CASE_TOPIC_KEYWORDS)
    if case_hits:
        return TopicSwitchResult(
            is_topic_switch=True,
            reason="用户输入包含新的案例总结或诈骗套路归纳意图",
            suggested_task_type=TASK_CASE_SUMMARY,
            matched_keywords=case_hits,
        )

    if any(text.startswith(opener) for opener in TOPIC_SWITCH_OPENERS):
        if has_profile_demographic or platform_hits:
            return TopicSwitchResult(
                is_topic_switch=True,
                reason="用户以换话题方式引入了新的用户画像风险分析任务",
                suggested_task_type=TASK_USER_RISK_PROFILE,
                matched_keywords=platform_hits,
            )
        if case_hits:
            return TopicSwitchResult(
                is_topic_switch=True,
                reason="用户以换话题方式引入了新的案例总结任务",
                suggested_task_type=TASK_CASE_SUMMARY,
                matched_keywords=case_hits,
            )
        if text_switch:
            return TopicSwitchResult(
                is_topic_switch=True,
                reason="用户以换话题方式引入了新的文本风险分析任务",
                suggested_task_type=TASK_TEXT_RISK,
                matched_keywords=text_hits,
            )

    return TopicSwitchResult(is_topic_switch=False)


def classify_follow_up(user_input: str) -> FollowUpClassification:
    text = (user_input or "").strip()
    if not text:
        return FollowUpClassification(is_follow_up=False)

    remedial_hits = [keyword for keyword in REMEDIAL_ACTION_KEYWORDS if keyword in text]
    if remedial_hits:
        return FollowUpClassification(
            is_follow_up=True,
            follow_up_type=FOLLOW_UP_TYPE_REMEDIAL,
            matched_keywords=remedial_hits,
        )

    sensitive_hits = [keyword for keyword in SENSITIVE_REMEDIAL_KEYWORDS if keyword in text]
    has_past_action = any(keyword in text for keyword in PAST_ACTION_HINTS)
    if has_past_action and sensitive_hits:
        return FollowUpClassification(
            is_follow_up=True,
            follow_up_type=FOLLOW_UP_TYPE_REMEDIAL,
            matched_keywords=sensitive_hits,
        )

    explanation_hits = [keyword for keyword in EXPLANATION_KEYWORDS if keyword in text]
    if explanation_hits:
        return FollowUpClassification(
            is_follow_up=True,
            follow_up_type=FOLLOW_UP_TYPE_EXPLANATION,
            matched_keywords=explanation_hits,
        )

    procedure_hits = []
    for keywords in (
        PROCEDURE_ALARM_KEYWORDS,
        PROCEDURE_96110_KEYWORDS,
        PROCEDURE_BANK_KEYWORDS,
        PROCEDURE_EVIDENCE_KEYWORDS,
    ):
        procedure_hits.extend([keyword for keyword in keywords if keyword in text])
    if procedure_hits:
        return FollowUpClassification(
            is_follow_up=True,
            follow_up_type=FOLLOW_UP_TYPE_PROCEDURE,
            matched_keywords=procedure_hits,
        )

    next_step_hits = [keyword for keyword in NEXT_STEP_KEYWORDS if keyword in text]
    if next_step_hits:
        return FollowUpClassification(
            is_follow_up=True,
            follow_up_type=FOLLOW_UP_TYPE_NEXT_STEP,
            matched_keywords=next_step_hits,
        )

    if (
        len(text) <= 18
        and ("?" in text or "？" in text)
        and (any(text.startswith(prefix) for prefix in SHORT_FOLLOW_UP_PREFIXES) or len(text) <= 10)
    ):
        return FollowUpClassification(
            is_follow_up=True,
            follow_up_type=FOLLOW_UP_TYPE_NEXT_STEP,
        )

    return FollowUpClassification(is_follow_up=False)


def looks_like_follow_up(user_input: str) -> bool:
    return classify_follow_up(user_input).is_follow_up


def build_memory_context(
    store: AgentSessionStore,
    session_id: str,
    *,
    user_input: str,
    use_memory: bool,
    history_limit: int,
) -> AgentMemoryContext:
    context = AgentMemoryContext(session_id=session_id)
    if not use_memory:
        return context

    recent_messages = store.list_messages(session_id, limit=max(2, history_limit + 1))
    recent_tasks = store.list_task_records(session_id, limit=1)
    recent_task = recent_tasks[-1] if recent_tasks else None
    follow_up_result = classify_follow_up(user_input)
    follow_up = follow_up_result.is_follow_up and recent_task is not None

    if recent_messages and recent_messages[-1].role == "user" and recent_messages[-1].content.strip() == user_input.strip():
        recent_messages = recent_messages[:-1]
    recent_messages = recent_messages[-max(1, history_limit) :]

    context.recent_messages = recent_messages
    context.recent_task = recent_task
    context.context_used = bool(recent_messages or recent_task)
    context.memory_summary = summarize_recent_context(recent_task, recent_messages)
    topic_switch = detect_topic_switch(user_input, context)
    context.topic_switch_detected = topic_switch.is_topic_switch
    context.topic_switch_reason = topic_switch.reason
    context.topic_switch_task_type = topic_switch.suggested_task_type
    context.topic_switch_keywords = list(topic_switch.matched_keywords)
    context.follow_up_detected = follow_up and not topic_switch.is_topic_switch
    context.follow_up_candidate_type = follow_up_result.follow_up_type
    context.follow_up_type = follow_up_result.follow_up_type if context.follow_up_detected else None
    return context


def summarize_recent_context(
    recent_task: AgentTaskRecord | None,
    recent_messages: list[AgentMessage],
) -> str | None:
    if recent_task is not None:
        parts = [
            f"上一轮任务类型是 {recent_task.task_type}。",
            f"结论：{recent_task.conclusion or '无'}。",
        ]
        if recent_task.risk_level and recent_task.risk_level != "unknown":
            parts.append(f"风险等级：{recent_task.risk_level}。")
        if recent_task.fraud_type:
            parts.append(f"诈骗类型：{recent_task.fraud_type}。")
        if recent_task.suggestions:
            parts.append(f"建议：{'；'.join(recent_task.suggestions[:3])}。")
        if recent_task.fallback_message:
            parts.append(f"降级说明：{recent_task.fallback_message}。")
        return "".join(parts)

    if recent_messages:
        last_user = next((item for item in reversed(recent_messages) if item.role == "user"), None)
        if last_user is not None:
            return f"最近一轮用户输入：{last_user.content[:120]}"
    return None
