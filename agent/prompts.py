from __future__ import annotations

from typing import Any


def _format_cases(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "无直接匹配案例。"
    lines = []
    for index, case in enumerate(cases[:5], start=1):
        title = case.get("title", "")
        summary = case.get("summary", "")
        source = case.get("source", "")
        lines.append(f"{index}. {title}\n摘要：{summary}\n来源：{source}")
    return "\n\n".join(lines)


def build_text_risk_advice_prompt(
    user_text: str,
    sms_result: dict[str, Any] | None,
    related_cases: list[dict[str, Any]],
) -> str:
    return (
        "你是一个像 ChatGPT 一样自然交流的反诈助手。请直接和用户说话，不要写成报告，不要使用"
        "“结论：”“建议1、2、3”这样的模板。\n\n"
        f"用户发来的内容：{user_text}\n"
        f"识别结果：{sms_result or '暂无'}\n"
        f"相关案例：\n{_format_cases(related_cases)}\n\n"
        "请用自然、正常的中文对话回答，先说你的判断，再解释为什么需要警惕，最后给出下一步最重要的提醒。"
        "除非非常必要，不要分点，不要刻意凑字数。"
    )


def build_case_summary_prompt(
    user_query: str,
    related_cases: list[dict[str, Any]],
    graph_result: dict[str, Any] | None,
) -> str:
    return (
        "你是一个自然交流的反诈助手。用户在问案例时，请像聊天一样回答，不要写成生硬摘要。"
        "可以引用检索到的案例，但不要编造细节。\n\n"
        f"用户问题：{user_query}\n"
        f"案例结果：\n{_format_cases(related_cases)}\n"
        f"图谱结果：{graph_result or '暂无'}\n\n"
        "请直接告诉用户查到了什么、这些案例大致反映了什么套路、对现实里有什么提醒。"
    )


def build_profile_advice_prompt(
    user_query: str,
    profile: dict[str, Any],
    risk_result: dict[str, Any] | None,
    related_cases: list[dict[str, Any]],
) -> str:
    return (
        "你是一个自然交流的反诈助手。请根据用户画像，用像正常对话一样的方式给建议，不要报告腔。"
        "\n\n"
        f"用户请求：{user_query}\n"
        f"用户画像：{profile}\n"
        f"风险评估结果：{risk_result or '暂无'}\n"
        f"相关案例：\n{_format_cases(related_cases)}\n\n"
        "请告诉用户这类人容易遇到什么骗局、为什么容易被盯上、平时最值得注意什么。"
    )


def build_sms_failure_fallback_prompt(user_text: str) -> str:
    return (
        "你是一个自然交流的反诈助手。识别服务暂时不可用，请直接根据用户发来的内容给出稳妥判断。"
        "\n\n"
        f"用户内容：{user_text}\n\n"
        "请用自然中文提醒用户先不要点击链接、不要泄露验证码或银行卡信息，并建议通过官方渠道核实。"
    )


def build_follow_up_prompt(
    *,
    user_query: str,
    recent_task_type: str,
    recent_conclusion: str,
    recent_risk_level: str,
    recent_fraud_type: str | None,
    recent_evidence: list[str],
    recent_suggestions: list[str],
    memory_summary: str | None,
    follow_up_type: str | None,
) -> str:
    return (
        "你是一个自然交流的反诈助手，正在承接上一轮分析继续和用户聊天。"
        "不要把回答写成报告，也不要重新从零开始。\n\n"
        f"追问类型：{follow_up_type or 'unknown'}\n"
        f"上一轮任务类型：{recent_task_type}\n"
        f"上一轮结论：{recent_conclusion}\n"
        f"上一轮风险等级：{recent_risk_level}\n"
        f"上一轮诈骗类型：{recent_fraud_type or 'unknown'}\n"
        f"上一轮证据：{'；'.join(recent_evidence[:5]) or '无'}\n"
        f"上一轮建议：{'；'.join(recent_suggestions[:5]) or '无'}\n"
        f"历史摘要：{memory_summary or '无'}\n"
        f"用户当前追问：{user_query}\n\n"
        "请像延续同一段对话那样回答。如果用户问下一步怎么做，就明确告诉他先做什么；"
        "如果用户问为什么危险，就结合上一轮证据解释。"
    )


def build_remedial_follow_up_prompt(
    *,
    user_query: str,
    recent_conclusion: str,
    recent_risk_level: str,
    recent_fraud_type: str | None,
    memory_summary: str | None,
    required_actions: list[str],
) -> str:
    return (
        "你是一个自然交流的反诈助手，正在处理已经发生高风险操作后的补救追问。"
        "不要弱化风险，也不要写成条款式说明。\n\n"
        f"上一轮结论：{recent_conclusion}\n"
        f"上一轮风险等级：{recent_risk_level}\n"
        f"上一轮诈骗类型：{recent_fraud_type or 'unknown'}\n"
        f"历史摘要：{memory_summary or '无'}\n"
        f"用户当前追问：{user_query}\n"
        f"必须覆盖的补救动作：{'；'.join(required_actions)}\n\n"
        "请先明确告诉用户这件事要尽快处理，再自然地说明最优先的补救动作和为什么。"
    )
