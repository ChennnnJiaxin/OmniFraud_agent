from __future__ import annotations

import re

from agent.models import (
    AgentRouteDecision,
    AgentRunRequest,
    TASK_CASE_SUMMARY,
    TASK_IMAGE_RISK,
    TASK_OUT_OF_SCOPE,
    TASK_TEXT_RISK,
    TASK_UNKNOWN,
    TASK_USER_RISK_PROFILE,
    OutOfScopeResult,
)

IMAGE_KEYWORDS = ("图片", "截图", "聊天截图", "这张图", "照片")
PROFILE_KEYWORDS = (
    "我爸妈",
    "老人",
    "老年人",
    "父母",
    "大学生",
    "学生",
    "用户画像",
    "风险画像",
    "容易遇到什么骗局",
    "适合给什么建议",
)
TEXT_RISK_KEYWORDS = (
    "短信",
    "验证码",
    "链接",
    "账户异常",
    "快递",
    "退款",
    "客服",
    "中奖",
    "转账",
    "垫付",
    "刷单",
    "银行卡",
    "冻结账户",
    "是不是诈骗",
    "靠谱不",
    "安全账户",
    "洗钱",
    "投资",
    "充值",
    "下载app",
    "下载 App",
    "平台",
)
CASE_KEYWORDS = (
    "案例",
    "类似案例",
    "有哪些套路",
    "有什么特征",
    "杀猪盘",
    "刷单返利",
    "冒充客服",
    "投资理财诈骗",
    "知识图谱",
    "关系",
    "总结",
    "共性",
)
GREETING_KEYWORDS = (
    "你好",
    "您好",
    "嗨",
    "hello",
    "hi",
    "你是谁",
    "你叫什么",
    "你会什么",
    "介绍一下你自己",
)
MATH_KEYWORDS = (
    "等于几",
    "多少",
    "求导",
    "解方程",
    "计算面积",
    "次方",
)
WEATHER_TIME_NEWS_KEYWORDS = (
    "天气",
    "几点",
    "时间",
    "新闻",
    "头条",
)
WRITING_ENTERTAINMENT_KEYWORDS = (
    "作文",
    "笑话",
    "电影",
    "写一首诗",
    "歌词",
)
PROGRAMMING_KEYWORDS = (
    "python",
    "java",
    "c++",
    "html",
    "css",
    "javascript",
    "代码",
    "编程",
    "冒泡排序",
    "算法",
)
ENCYCLOPEDIA_KEYWORDS = (
    "牛顿是谁",
    "地球有多大",
    "世界杯谁赢了",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def _looks_like_analysis_text(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) >= 12:
        return True
    return any(marker in normalized for marker in ("：", ":", "“", "\"", "QQ", "http", "www."))


def _looks_like_math_question(text: str) -> bool:
    normalized = text.replace(" ", "")
    has_operator = any(operator in normalized for operator in ("+", "-", "*", "/", "="))
    has_digits = any(char.isdigit() for char in normalized)
    return (has_operator and has_digits) or bool(_contains_any(normalized, MATH_KEYWORDS))


def _looks_like_case_query(text: str) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []

    matches: list[str] = []
    explicit_markers = (
        "诈骗案",
        "案件",
        "案情",
        "判决书",
        "非法集资案",
        "集资诈骗案",
    )
    matches.extend([marker for marker in explicit_markers if marker in normalized])

    if re.search(r"[\u4e00-\u9fffA-Za-z0-9]{2,30}(诈骗案|案件|案情|判决书)", normalized):
        matches.append("named_case_pattern")

    ask_markers = ("讲一下", "讲讲", "说说", "介绍一下", "分析一下", "聊聊", "科普一下")
    if any(marker in normalized for marker in ask_markers) and any(marker in normalized for marker in explicit_markers):
        matches.append("case_explainer_pattern")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in matches:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def detect_out_of_scope(user_input: str) -> OutOfScopeResult:
    text = (user_input or "").strip()
    lowered = text.lower()
    if not text:
        return OutOfScopeResult(is_out_of_scope=False)

    greeting_hits = _contains_any(lowered, tuple(keyword.lower() for keyword in GREETING_KEYWORDS))
    if greeting_hits:
        return OutOfScopeResult(
            is_out_of_scope=True,
            reason="用户输入是普通问候或自我介绍类闲聊，与反诈任务无关",
            matched_keywords=greeting_hits,
        )

    if _looks_like_math_question(text):
        matched_keywords = _contains_any(text, MATH_KEYWORDS)
        if any(char in text for char in ("+", "-", "*", "/", "=")) and any(char.isdigit() for char in text):
            matched_keywords = matched_keywords or ["math_expression"]
        return OutOfScopeResult(
            is_out_of_scope=True,
            reason="用户输入是数学或计算问题，与反诈任务无关",
            matched_keywords=matched_keywords,
        )

    weather_hits = _contains_any(text, WEATHER_TIME_NEWS_KEYWORDS)
    if weather_hits:
        return OutOfScopeResult(
            is_out_of_scope=True,
            reason="用户输入是天气、时间或新闻查询，与反诈任务无关",
            matched_keywords=weather_hits,
        )

    writing_hits = _contains_any(text, WRITING_ENTERTAINMENT_KEYWORDS)
    if writing_hits and any(keyword in text for keyword in ("写", "讲", "推荐")):
        return OutOfScopeResult(
            is_out_of_scope=True,
            reason="用户输入是写作或娱乐请求，与反诈任务无关",
            matched_keywords=writing_hits,
        )

    programming_hits = _contains_any(lowered, tuple(keyword.lower() for keyword in PROGRAMMING_KEYWORDS))
    if programming_hits and any(keyword in lowered for keyword in ("怎么写", "代码", "编程", "算法")):
        return OutOfScopeResult(
            is_out_of_scope=True,
            reason="用户输入是编程开发问题，与反诈任务无关",
            matched_keywords=programming_hits,
        )

    encyclopedia_hits = _contains_any(text, ENCYCLOPEDIA_KEYWORDS)
    if encyclopedia_hits:
        return OutOfScopeResult(
            is_out_of_scope=True,
            reason="用户输入是百科常识问题，与反诈任务无关",
            matched_keywords=encyclopedia_hits,
        )

    return OutOfScopeResult(is_out_of_scope=False)


def route_agent_task(request: AgentRunRequest) -> AgentRouteDecision:
    text = request.user_input.strip()

    if request.input_type == "image" or request.image is not None:
        return AgentRouteDecision(
            task_type=TASK_IMAGE_RISK,
            reason="请求包含图片输入",
            matched_rules=["input_type=image" if request.input_type == "image" else "image field present"],
        )

    image_hits = _contains_any(text, IMAGE_KEYWORDS)
    if image_hits:
        return AgentRouteDecision(task_type=TASK_IMAGE_RISK, reason="命中图片分析关键词", matched_rules=image_hits)

    profile_hits = _contains_any(text, PROFILE_KEYWORDS)
    if request.profile or request.input_type == "profile" or profile_hits:
        matched_rules = profile_hits or []
        if request.profile:
            matched_rules.append("profile field present")
        if request.input_type == "profile":
            matched_rules.append("input_type=profile")
        return AgentRouteDecision(task_type=TASK_USER_RISK_PROFILE, reason="命中用户画像规则", matched_rules=matched_rules)

    case_hits = _contains_any(text, CASE_KEYWORDS)
    strong_case_hits = [keyword for keyword in case_hits if keyword in {"案例", "类似案例", "有哪些套路", "有什么特征", "总结", "共性"}]
    if strong_case_hits:
        return AgentRouteDecision(task_type=TASK_CASE_SUMMARY, reason="命中显式案例总结关键词", matched_rules=strong_case_hits)

    natural_case_hits = _looks_like_case_query(text)
    if natural_case_hits:
        return AgentRouteDecision(task_type=TASK_CASE_SUMMARY, reason="命中自然语言案件查询模式", matched_rules=natural_case_hits)

    text_risk_hits = _contains_any(text.lower(), tuple(keyword.lower() for keyword in TEXT_RISK_KEYWORDS))
    if text_risk_hits and _looks_like_analysis_text(text):
        return AgentRouteDecision(task_type=TASK_TEXT_RISK, reason="命中文本风险分析关键词", matched_rules=text_risk_hits)

    if case_hits:
        return AgentRouteDecision(task_type=TASK_CASE_SUMMARY, reason="命中案例总结关键词", matched_rules=case_hits)

    out_of_scope = detect_out_of_scope(text)
    if out_of_scope.is_out_of_scope:
        return AgentRouteDecision(
            task_type=TASK_OUT_OF_SCOPE,
            reason=out_of_scope.reason or "用户输入与反诈任务无关",
            matched_rules=out_of_scope.matched_keywords,
        )

    return AgentRouteDecision(task_type=TASK_UNKNOWN, reason="未命中任何任务规则", matched_rules=[])
