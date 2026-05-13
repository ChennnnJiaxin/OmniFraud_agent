from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
import html
from pathlib import Path

import streamlit as st

from agent.agent_service import (
    create_agent_session,
    list_agent_session_messages,
    list_agent_session_tasks,
    run_agent,
)
from agent.models import AgentRunRequest
from agent.session_models import AgentMessage, AgentSession
from agent.session_store import get_agent_session_store
from agent.session_title import generate_session_title, is_default_session_title

try:
    from streamlit_extras.stylable_container import stylable_container
except Exception:

    @contextmanager
    def stylable_container(key: str, css_styles):
        del key, css_styles
        yield


DEFAULT_SESSION_TITLE = "新的反诈会话"

SESSION_KEY = "agent_page_session_id"
SESSION_LIST_KEY = "agent_page_session_list"
LAST_RESPONSE_KEY = "agent_page_last_response"
RUNNING_KEY = "agent_page_is_running"
FORM_VERSION_KEY = "agent_page_form_version"
PENDING_FORM_RESET_KEY = "agent_page_pending_form_reset"
INPUT_KEY = "agent_image_prompt"
CHAT_INPUT_KEY = "agent_chat_input"
RENAME_INPUT_KEY = "agent_rename_input"
SESSION_MENU_KEY = "agent_session_menu_open"
UPLOAD_INPUT_KEY = "agent_inline_upload"

UI_FIGURES_DIR = Path(__file__).resolve().parent.parent / "ui_figures"


def _session_store():
    return get_agent_session_store()


def _image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".") or "png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_timestamp(value: str | None) -> str:
    parsed = _parse_timestamp(value)
    if parsed is not None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    if not value:
        return "-"
    return value.replace("T", " ").replace("Z", "")


def _format_sidebar_time(value: str | None) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return "-"

    now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.now()
    if parsed.date() == now.date():
        return parsed.strftime("%H:%M")
    if parsed.date() == now.date() - timedelta(days=1):
        return f"昨天 {parsed.strftime('%H:%M')}"
    return parsed.strftime("%m-%d %H:%M")


def _format_message_time(value: str | None) -> str:
    parsed = _parse_timestamp(value)
    if parsed is not None:
        return parsed.strftime("%H:%M:%S")
    return _format_timestamp(value)


def _short_session_id(session_id: str | None) -> str:
    return (session_id or "")[:8] or "-"


def _default_title() -> str:
    return DEFAULT_SESSION_TITLE


def _truncate_text(value: str | None, max_length: int = 26) -> str:
    normalized = " ".join((value or "").strip().split())
    if not normalized:
        return ""
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max(1, max_length - 1)].rstrip() + "…"


def _sidebar_button_title(value: str | None) -> str:
    normalized = (value or "").strip() or _default_title()
    return f"\u2003{normalized}"


def _normalize_text_lines(value: str | None) -> list[str]:
    if not value:
        return []
    lines: list[str] = []
    for raw_line in value.replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        lines.append(line.lstrip("-• ").strip())
    return lines


def _escape_multiline(value: str | None) -> str:
    return html.escape(value or "").replace("\n", "<br>")


def _list_to_html(items: list[str], *, ordered: bool = False, empty_text: str = "暂无") -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return f'<div class="analysis-empty">{html.escape(empty_text)}</div>'

    tag = "ol" if ordered else "ul"
    body = "".join(f"<li>{html.escape(item)}</li>" for item in cleaned)
    return f'<{tag} class="analysis-list">{body}</{tag}>'


def _response_detail_items(response) -> list[str]:
    items = list(getattr(response, "evidence", None) or [])
    fraud_type = getattr(response, "fraud_type", None)
    risk_level = getattr(response, "risk_level", None)
    vulnerable_types = getattr(response, "vulnerable_fraud_types", None)
    memory_summary = getattr(response, "memory_summary", None)

    if fraud_type:
        items.append(f"诈骗类型：{fraud_type}")
    if risk_level:
        items.append(f"风险等级：{risk_level}")
    if vulnerable_types:
        items.append(f"易受骗类型：{'、'.join(vulnerable_types)}")
    if memory_summary:
        items.append(f"上下文摘要：{memory_summary}")
    return items


def _response_suggestion_items(response) -> list[str]:
    suggestions = list(getattr(response, "suggestions", None) or [])
    if suggestions:
        return suggestions

    answer_lines = _normalize_text_lines(getattr(response, "answer", None))
    if answer_lines:
        return answer_lines[:6]

    conclusion = getattr(response, "conclusion", None)
    return [conclusion] if conclusion else []


def _response_lead_text(response) -> str:
    return getattr(response, "answer", None) or getattr(response, "conclusion", None) or "已完成本轮分析。"


def _ensure_page_state() -> None:
    st.session_state.setdefault(RUNNING_KEY, False)
    st.session_state.setdefault(FORM_VERSION_KEY, 0)
    st.session_state.setdefault(PENDING_FORM_RESET_KEY, False)
    st.session_state.setdefault(INPUT_KEY, "")
    st.session_state.setdefault(CHAT_INPUT_KEY, "")
    st.session_state.setdefault(RENAME_INPUT_KEY, "")
    st.session_state.setdefault(SESSION_MENU_KEY, "")
    _apply_pending_form_reset()


def _apply_form_reset() -> None:
    st.session_state[INPUT_KEY] = ""
    st.session_state[CHAT_INPUT_KEY] = ""
    st.session_state[FORM_VERSION_KEY] = st.session_state.get(FORM_VERSION_KEY, 0) + 1
    st.session_state[PENDING_FORM_RESET_KEY] = False


def _apply_pending_form_reset() -> None:
    if st.session_state.get(PENDING_FORM_RESET_KEY):
        _apply_form_reset()


def _schedule_form_reset() -> None:
    st.session_state[PENDING_FORM_RESET_KEY] = True


def _set_active_session(session_id: str) -> None:
    st.session_state[SESSION_KEY] = session_id
    st.session_state[SESSION_LIST_KEY] = session_id
    st.session_state.pop(LAST_RESPONSE_KEY, None)
    st.session_state[SESSION_MENU_KEY] = ""
    _schedule_form_reset()


def _get_session(session_id: str | None) -> AgentSession | None:
    if not session_id:
        return None
    return _session_store().get_session(session_id)


def _ensure_session() -> str:
    session_id = st.session_state.get(SESSION_KEY)
    session = _get_session(session_id)
    if session is not None:
        st.session_state[SESSION_LIST_KEY] = session.session_id
        return session.session_id

    created = create_agent_session(title=_default_title())
    st.session_state[SESSION_KEY] = created.session_id
    st.session_state[SESSION_LIST_KEY] = created.session_id
    return created.session_id


def _create_new_session() -> str:
    session = create_agent_session(title=_default_title())
    _set_active_session(session.session_id)
    return session.session_id


def _delete_session_and_recover(session_id: str, current_session_id: str) -> str:
    sessions = [item for item in _session_store().list_sessions(limit=100) if item.session_id != session_id]
    _session_store().clear_session(session_id)
    st.session_state.pop(LAST_RESPONSE_KEY, None)

    if session_id != current_session_id:
        return current_session_id

    if sessions:
        next_session_id = sessions[0].session_id
        _set_active_session(next_session_id)
        return next_session_id

    return _create_new_session()


def _title_can_be_generated(messages: list[AgentMessage]) -> bool:
    has_user = any(message.role == "user" and message.content.strip() for message in messages)
    has_assistant = any(message.role == "assistant" and message.content.strip() for message in messages)
    return has_user and has_assistant


def _maybe_update_session_title(session_id: str) -> None:
    session = _get_session(session_id)
    if session is None or not is_default_session_title(session.title, _default_title()):
        return

    messages = list_agent_session_messages(session_id, limit=6)
    if not _title_can_be_generated(messages):
        return

    title = generate_session_title(messages, default_title=_default_title())
    if title:
        _session_store().update_session_title(session_id, title)


def _get_recent_sessions(current_session_id: str, limit: int = 20) -> list[AgentSession]:
    sessions = _session_store().list_sessions(limit=limit)
    if any(item.session_id == current_session_id for item in sessions):
        return sessions

    current_session = _get_session(current_session_id)
    if current_session is None:
        return sessions
    return [current_session, *sessions[: max(0, limit - 1)]]


def _render_sidebar_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background: var(--agent-surface);
            border-right: 1px solid var(--agent-border);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            padding-left: 0.95rem;
            padding-right: 0.85rem;
            padding-bottom: 1rem;
        }
        [data-testid="stSidebar"] .stButton > button {
            box-shadow: none;
            border: none;
            background: transparent;
            text-align: left;
            justify-content: flex-start;
            transition: background-color 0.16s ease, color 0.16s ease;
        }
        [data-testid="stSidebar"] [data-testid="column"]:first-child .stButton > button {
            margin-left: 0.28rem;
            width: calc(100% - 0.28rem) !important;
        }
        [data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button {
            margin-right: 0.28rem;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            box-shadow: none;
            border: none;
            background: color-mix(in srgb, var(--agent-text) 8%, transparent);
        }
        [data-testid="stSidebar"] .stButton > button p {
            width: 100%;
            text-align: left;
        }
        [data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button p {
            text-align: center;
        }
        [data-testid="stSidebar"] [data-testid="stPopover"] button {
            box-shadow: none;
            border: none;
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light dark;
            --agent-app-bg:
                radial-gradient(circle at top right, rgba(190, 227, 255, 0.42), transparent 28%),
                linear-gradient(180deg, #f5fafe 0%, #f8fbff 28%, #fcfdff 100%);
            --agent-orb-bg: radial-gradient(circle, rgba(138, 203, 255, 0.28) 0%, rgba(138, 203, 255, 0.08) 42%, transparent 72%);
            --agent-surface: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(252, 254, 255, 0.98));
            --agent-surface-strong: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(245, 250, 255, 0.84));
            --agent-surface-soft: linear-gradient(180deg, #f7fbff 0%, #eef6ff 100%);
            --agent-user-surface: linear-gradient(180deg, #eef6ff 0%, #e8f2ff 100%);
            --agent-badge-surface: linear-gradient(180deg, #f6fbff 0%, #dceefe 100%);
            --agent-composer-surface: rgba(255, 255, 255, 0.78);
            --agent-border: rgba(160, 199, 234, 0.34);
            --agent-border-strong: rgba(160, 199, 234, 0.38);
            --agent-border-soft: rgba(181, 210, 238, 0.5);
            --agent-border-faint: rgba(218, 230, 242, 0.82);
            --agent-shadow: 0 22px 46px rgba(148, 184, 216, 0.14);
            --agent-shadow-soft: 0 14px 30px rgba(148, 184, 216, 0.12);
            --agent-text: #1e293b;
            --agent-text-soft: #64748b;
            --agent-text-muted: #5f6f85;
            --agent-accent: #2563eb;
            --agent-accent-strong: #215b9f;
            --agent-accent-soft: #31506f;
            --agent-input-text: #0f172a;
            --agent-placeholder: #9aa9bb;
            --agent-session-active-bg: linear-gradient(180deg, #eaf3ff 0%, #e1efff 100%);
            --agent-session-active-border: rgba(143, 191, 236, 0.46);
            --agent-session-active-text: #0f3f7d;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --agent-app-bg:
                    radial-gradient(circle at top right, rgba(59, 130, 246, 0.16), transparent 28%),
                    linear-gradient(180deg, #0b1220 0%, #0f172a 38%, #111827 100%);
                --agent-orb-bg: radial-gradient(circle, rgba(96, 165, 250, 0.18) 0%, rgba(96, 165, 250, 0.05) 42%, transparent 72%);
                --agent-surface: linear-gradient(180deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 42, 0.98));
                --agent-surface-strong: linear-gradient(180deg, rgba(17, 24, 39, 0.9), rgba(15, 23, 42, 0.94));
                --agent-surface-soft: linear-gradient(180deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.96));
                --agent-user-surface: linear-gradient(180deg, rgba(30, 41, 59, 0.96), rgba(22, 31, 49, 0.98));
                --agent-badge-surface: linear-gradient(180deg, rgba(30, 41, 59, 0.98), rgba(37, 99, 235, 0.32));
                --agent-composer-surface: rgba(15, 23, 42, 0.88);
                --agent-border: rgba(96, 165, 250, 0.18);
                --agent-border-strong: rgba(96, 165, 250, 0.24);
                --agent-border-soft: rgba(96, 165, 250, 0.2);
                --agent-border-faint: rgba(71, 85, 105, 0.85);
                --agent-shadow: 0 22px 46px rgba(2, 6, 23, 0.38);
                --agent-shadow-soft: 0 14px 30px rgba(2, 6, 23, 0.28);
                --agent-text: #e5eefb;
                --agent-text-soft: #bfd0ea;
                --agent-text-muted: #c8d5e6;
                --agent-accent: #60a5fa;
                --agent-accent-strong: #93c5fd;
                --agent-accent-soft: #d7e8ff;
                --agent-input-text: #e5eefb;
                --agent-placeholder: #8ea5c4;
                --agent-session-active-bg: linear-gradient(180deg, rgba(37, 99, 235, 0.24) 0%, rgba(30, 64, 175, 0.22) 100%);
                --agent-session-active-border: rgba(96, 165, 250, 0.34);
                --agent-session-active-text: #dbeafe;
            }
        }
        [data-testid="stHeader"] {
            height: 0;
            background: transparent;
        }
        [data-testid="stToolbar"] {
            top: 0.35rem;
            right: 0.5rem;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--agent-app-bg);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 0.35rem;
            padding-bottom: 6.2rem;
        }
        .agent-shell {
            position: relative;
        }
        .agent-shell::before {
            content: "";
            position: absolute;
            top: -18px;
            right: 52px;
            width: 180px;
            height: 180px;
            border-radius: 999px;
            background: var(--agent-orb-bg);
            filter: blur(8px);
            pointer-events: none;
        }
        .agent-topbar {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.95rem 1.1rem 1rem;
            border: 1px solid var(--agent-border-strong);
            border-radius: 22px;
            background: var(--agent-surface-strong);
            box-shadow: var(--agent-shadow-soft);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
            overflow: hidden;
        }
        .agent-topbar::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(141, 197, 255, 0.08), transparent 40%);
            pointer-events: none;
        }
        .agent-identity {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            gap: 0.9rem;
            min-width: 0;
        }
        .agent-badge {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--agent-badge-surface);
            border: 1px solid var(--agent-border-strong);
            color: var(--agent-accent);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.88);
            font-size: 1.25rem;
            flex-shrink: 0;
        }
        .agent-title-wrap {
            min-width: 0;
        }
        .agent-title {
            margin: 0;
            color: var(--agent-text);
            font-size: 1.42rem;
            line-height: 1.1;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .agent-subtitle {
            margin: 0.24rem 0 0;
            color: var(--agent-text-soft);
            font-size: 0.94rem;
            line-height: 1.45;
        }
        .agent-status-chip {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.52rem 0.78rem;
            border-radius: 999px;
            border: 1px solid var(--agent-border);
            background: color-mix(in srgb, var(--agent-composer-surface) 65%, transparent);
            color: var(--agent-accent-strong);
            font-size: 0.82rem;
            font-weight: 600;
            white-space: nowrap;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.92);
        }
        .agent-chat-card {
            position: relative;
            border-radius: 28px;
            padding: 1.1rem 1.15rem 1.3rem;
            border: 1px solid var(--agent-border);
            background: var(--agent-surface);
            box-shadow: var(--agent-shadow);
            min-height: clamp(320px, 46vh, 560px);
        }
        .agent-chat-card::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            pointer-events: none;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.2), transparent 20%),
                radial-gradient(circle at top right, rgba(171, 219, 255, 0.14), transparent 24%);
        }
        .agent-empty-state {
            padding: 0.4rem 0.2rem 0.1rem;
        }
        .agent-empty-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.32rem 0.68rem;
            border-radius: 999px;
            background: color-mix(in srgb, var(--agent-badge-surface) 80%, transparent);
            color: var(--agent-accent);
            font-size: 0.78rem;
            font-weight: 600;
        }
        .agent-empty-title {
            margin: 0.9rem 0 0.42rem;
            color: var(--agent-text);
            font-size: 1.3rem;
            font-weight: 700;
            line-height: 1.25;
        }
        .agent-empty-copy {
            margin: 0;
            max-width: 680px;
            color: var(--agent-text-muted);
            font-size: 0.98rem;
            line-height: 1.7;
        }
        .agent-empty-prompts {
            display: flex;
            flex-wrap: wrap;
            gap: 0.68rem;
            margin-top: 0.85rem;
        }
        .agent-empty-prompt {
            padding: 0.72rem 0.9rem;
            border-radius: 16px;
            background: var(--agent-surface-soft);
            border: 1px solid var(--agent-border-soft);
            color: var(--agent-accent-soft);
            font-size: 0.89rem;
            line-height: 1.45;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
        }
        [data-testid="stChatMessage"] {
            margin-bottom: 0.8rem;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.72;
            color: var(--agent-text);
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            background: var(--agent-user-surface);
            border: 1px solid var(--agent-border-soft);
            border-radius: 22px;
            box-shadow: var(--agent-shadow-soft);
            padding: 0.3rem 0.5rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            background: var(--agent-surface);
            border: 1px solid var(--agent-border-faint);
            border-radius: 22px;
            box-shadow: var(--agent-shadow-soft);
            padding: 0.3rem 0.5rem;
        }
        .agent-composer-shell {
            margin-top: 0.95rem;
        }
        .agent-composer-note {
            margin: 0 0 0.42rem 0.55rem;
            color: var(--agent-text-soft);
            font-size: 0.82rem;
            line-height: 1.45;
        }
        .agent-upload-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: color-mix(in srgb, var(--agent-badge-surface) 80%, transparent);
            color: var(--agent-accent-strong);
            font-size: 0.8rem;
            font-weight: 600;
        }
        .agent-upload-help {
            margin: 0.42rem 0 0;
            color: var(--agent-text-soft);
            font-size: 0.8rem;
            line-height: 1.5;
        }
        .agent-upload-popover p {
            margin-bottom: 0;
        }
        .agent-upload-popover [data-testid="stFileUploaderDropzone"] {
            border-radius: 16px;
            border: 1px dashed var(--agent-border-soft);
            background: color-mix(in srgb, var(--agent-surface-soft) 85%, transparent);
        }
        .agent-upload-popover [data-testid="stFileUploaderDropzoneInstructions"] svg {
            color: var(--agent-accent);
        }
        .agent-inline-composer {
            width: 100%;
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 0.65rem;
            padding: 0.72rem 0.78rem;
            border-radius: 999px;
            background: var(--agent-composer-surface);
            box-shadow: var(--agent-shadow);
            backdrop-filter: blur(12px);
        }
        .agent-inline-composer [data-testid="stPopover"] {
            display: flex;
            align-items: center;
        }
        .agent-inline-composer [data-testid="stPopover"] > div > button,
        .agent-inline-composer .stButton > button {
            min-height: 44px;
            width: 44px;
            border-radius: 999px;
            border: none;
            background: var(--agent-user-surface);
            color: var(--agent-accent);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.88);
            font-size: 1.4rem;
            font-weight: 500;
            padding: 0;
        }
        .agent-inline-composer .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #4b9cff 0%, #2f7eea 100%);
            color: #ffffff;
            box-shadow: 0 10px 20px rgba(52, 120, 215, 0.26);
        }
        .agent-inline-composer div[data-baseweb="input"] {
            border: none;
            background: transparent;
            box-shadow: none;
        }
        .agent-inline-composer div[data-baseweb="input"] > div {
            border: none;
            background: transparent;
            box-shadow: none;
        }
        .agent-inline-composer input {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            color: var(--agent-input-text);
            font-size: 1rem;
            padding-left: 0.15rem;
        }
        .agent-inline-composer input::placeholder {
            color: var(--agent-placeholder);
        }
        .agent-inline-composer input:focus {
            outline: none;
            box-shadow: none;
        }
        .agent-inline-composer hr {
            display: none;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.3rem;
            }
            .agent-topbar {
                flex-direction: column;
                align-items: flex-start;
            }
            .agent-status-chip {
                white-space: normal;
            }
            .agent-chat-card {
                min-height: 300px;
                padding: 0.95rem 0.8rem 1.1rem;
            }
            .agent-inline-composer {
                grid-template-columns: auto 1fr auto;
                gap: 0.45rem;
                padding: 0.6rem 0.62rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_session_item(session: AgentSession, current_session_id: str) -> None:
    is_active = session.session_id == current_session_id
    title = session.title or _default_title()
    button_title = _sidebar_button_title(title)
    time_text = _format_sidebar_time(session.updated_at)
    rename_key = f"{RENAME_INPUT_KEY}_{session.session_id}"
    menu_button_key = f"session_menu_toggle_{session.session_id}"
    menu_open = st.session_state.get(SESSION_MENU_KEY) == session.session_id
    st.session_state.setdefault(rename_key, title)
    active_background = "var(--agent-session-active-bg)" if is_active else "transparent"
    active_border = "var(--agent-session-active-border)" if is_active else "transparent"
    active_text = "var(--agent-session-active-text)" if is_active else "var(--agent-text)"
    active_weight = "700" if is_active else "500"

    with stylable_container(
        key=f"session_row_{session.session_id}",
        css_styles=f"""
        {{
            margin-bottom: 0.16rem;
        }}
        [data-testid="column"]:first-child .stButton > button {{
            width: 100% !important;
            min-width: 0 !important;
            max-width: none !important;
            height: auto !important;
            min-height: 42px !important;
            border-radius: 14px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0.38rem 0.78rem 0.38rem 1.05rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            align-items: center !important;
            transform: none !important;
            aspect-ratio: auto !important;
            overflow: hidden !important;
            transition: background-color 0.16s ease, color 0.16s ease, border-color 0.16s ease;
            background: {active_background} !important;
            border-color: {active_border} !important;
        }}
        [data-testid="column"]:first-child .stButton > button:hover,
        [data-testid="column"]:first-child .stButton > button:focus,
        [data-testid="column"]:first-child .stButton > button:focus-visible,
        [data-testid="column"]:first-child .stButton > button:active {{
            background: color-mix(in srgb, var(--agent-text) 8%, transparent);
            color: var(--agent-text);
            border-radius: 14px !important;
            outline: none !important;
            box-shadow: none !important;
        }}
        [data-testid="column"]:first-child .stButton > button p {{
            margin: 0;
            padding-left: 0.12rem !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.25;
            font-size: 0.92rem;
            color: {active_text};
            font-weight: {active_weight};
        }}
        [data-testid="column"]:last-child .stButton > button {{
            min-height: 38px !important;
            height: 38px !important;
            width: 38px !important;
            min-width: 38px !important;
            max-width: 38px !important;
            border-radius: 12px !important;
            padding: 0 !important;
            justify-content: center !important;
            aspect-ratio: 1 / 1 !important;
        }}
        [data-testid="column"]:last-child .stButton > button p {{
            text-align: center;
        }}
        div[data-testid="stTextInput"] {{
            margin-top: 0.15rem;
        }}
        """,
    ):
        gutter_left, left_col, right_col, gutter_right = st.columns(
            [0.36, 5.84, 1.05, 0.36],
            gap="small",
            vertical_alignment="center",
        )
        with gutter_left:
            st.markdown("&nbsp;", unsafe_allow_html=True)
        with left_col:
            if st.button(
                button_title,
                key=f"open_session_{session.session_id}",
                use_container_width=True,
                type="secondary",
            ):
                if not is_active:
                    _set_active_session(session.session_id)
                    st.rerun()
        with right_col:
            if st.button("⋯", key=menu_button_key, use_container_width=True, type="secondary"):
                st.session_state[SESSION_MENU_KEY] = "" if menu_open else session.session_id
                st.rerun()
        with gutter_right:
            st.markdown("&nbsp;", unsafe_allow_html=True)
        if menu_open:
            st.caption(f"最近更新 {time_text}")
            st.text_input(
                "重命名会话",
                key=rename_key,
                label_visibility="collapsed",
                placeholder="为当前会话设置新标题",
            )
            action_left, action_right = st.columns([1, 1], gap="small")
            with action_left:
                if st.button("保存标题", key=f"rename_session_{session.session_id}", use_container_width=True):
                    new_title = (st.session_state.get(rename_key) or "").strip() or _default_title()
                    _session_store().update_session_title(session.session_id, new_title)
                    st.session_state.pop(LAST_RESPONSE_KEY, None)
                    st.session_state[SESSION_MENU_KEY] = ""
                    if hasattr(st, "toast"):
                        st.toast("会话标题已更新")
                    st.rerun()
            with action_right:
                if st.button("删除会话", key=f"delete_session_{session.session_id}", use_container_width=True):
                    _delete_session_and_recover(session.session_id, current_session_id)
                    st.session_state[SESSION_MENU_KEY] = ""
                    if hasattr(st, "toast"):
                        st.toast("会话已删除")
                    st.rerun()


def _render_page_header() -> None:
    st.markdown(
        """
        <div class="agent-shell">
          <section class="agent-topbar">
            <div class="agent-identity">
              <div class="agent-badge">🛡️</div>
              <div class="agent-title-wrap">
                <h1 class="agent-title">统一反诈智能体</h1>
                <p class="agent-subtitle">把短信、聊天记录、截图或你的疑虑发过来，我们像正常聊天一样一起判断风险。</p>
              </div>
            </div>
            <div class="agent-status-chip">对话模式</div>
          </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_session_summary(session: AgentSession | None) -> None:
    pass  # removed for cleaner UI


def _render_message_bubble(role: str, content: str, created_at: str) -> None:
    avatar = "🙂" if role == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        st.write(content)


def _render_response(response, use_tool_trace_expander: bool = True) -> None:
    with st.chat_message("assistant", avatar="🛡️"):
        lead_text = getattr(response, "answer", None) or getattr(response, "conclusion", None) or "已完成本轮分析。"
        st.write(lead_text)

        meta_parts: list[str] = []
        risk_level = getattr(response, "risk_level", None)
        fraud_type = getattr(response, "fraud_type", None)
        if risk_level and risk_level not in {"unknown", "-"}:
            meta_parts.append(f"风险等级：{risk_level}")
        if fraud_type:
            meta_parts.append(f"识别方向：{fraud_type}")
        if meta_parts:
            st.caption(" | ".join(meta_parts))

        details = _response_detail_items(response)
        if details:
            with st.expander("查看判断依据", expanded=False):
                for detail in details:
                    st.write(f"- {detail}")

        if getattr(response, "related_cases", None):
            with st.expander("相关案例", expanded=False):
                for case in response.related_cases:
                    case_dict = asdict(case)
                    st.markdown(f"**{case_dict['title']}**")
                    st.write(case_dict['summary'])
                    if case_dict.get("source"):
                        st.caption(f"来源：{case_dict['source']}")

        if not getattr(response, "success", True):
            st.caption("这次分析没有完全跑通，你可以补充更多上下文继续追问。")

        if use_tool_trace_expander and getattr(response, "tool_trace", None):
            with st.expander("工具轨迹", expanded=False):
                for trace in response.tool_trace:
                    trace_dict = asdict(trace)
                    status = "成功" if trace_dict["success"] else "失败"
                    st.write(f"{trace_dict['tool_name']}：{status}，{trace_dict['summary']}")
                    if trace_dict.get("error"):
                        st.caption(f"错误：{trace_dict['error'].get('message')}")
                        
def _render_response_analysis(response) -> None:
    if not getattr(response, "tool_trace", None):
        return

    with st.expander("本轮调用流程", expanded=False):
        for trace in response.tool_trace:
            trace_dict = asdict(trace)
            status = "成功" if trace_dict["success"] else "失败"
            st.write(f"{trace_dict['tool_name']}：{status}，{trace_dict['summary']}")
            if trace_dict.get("error"):
                st.caption(f"错误：{trace_dict['error'].get('message')}")


def _render_conversation(session_id: str, response=None) -> None:
    messages = list_agent_session_messages(session_id, limit=20)
    visible_messages = list(messages)

    with stylable_container(
        key=f"agent_chat_card_{session_id}",
        css_styles="""
        {
            margin-top: 0.2rem;
            position: relative;
            border-radius: 28px;
            padding: 1.1rem 1.15rem 1.3rem;
            border: 1px solid var(--agent-border);
            background: var(--agent-surface);
            box-shadow: var(--agent-shadow);
            min-height: clamp(320px, 46vh, 560px);
        }
        """,
    ):
        if not messages and response is None:
            st.markdown(
                """
                <div class="agent-empty-state">
                  <div class="agent-empty-kicker">轻量对话模式</div>
                  <h2 class="agent-empty-title">把你遇到的情况发给我，我会陪你一起判断风险。</h2>
                  <p class="agent-empty-copy">
                    你可以直接粘贴短信内容、聊天记录、链接描述，或者说明转账前后的情况。
                    页面会按对话方式持续记录分析结果和后续建议。
                  </p>
                  <div class="agent-empty-prompts">
                    <div class="agent-empty-prompt">帮我看看这条短信是不是诈骗，应该先核对哪些信息？</div>
                    <div class="agent-empty-prompt">我刚收到陌生链接和验证码要求，这种情况该怎么处理？</div>
                    <div class="agent-empty-prompt">对方要求立刻转账，我想先判断风险再决定要不要继续。</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        for message in visible_messages:
            _render_message_bubble(message.role, message.content, message.created_at)

        if response is not None:
            _render_response_analysis(response)


def _render_task_history(session_id: str) -> None:
    tasks = list_agent_session_tasks(session_id, limit=10)
    if not tasks:
        return

    with st.expander("任务记录", expanded=False):
        for task in reversed(tasks):
            st.markdown(
                f"**{task.task_type}** | 风险等级：{task.risk_level or '-'} | 时间：{_format_timestamp(task.created_at)}"
            )
            st.write(task.answer or task.conclusion or "暂无结论")
            st.divider()


def _build_agent_request(
    session_id: str,
    user_input: str,
    *,
    input_type: str = "text",
    image=None,
) -> AgentRunRequest:
    return AgentRunRequest(
        user_input=user_input,
        session_id=session_id,
        input_type=input_type,
        image=image,
        profile=None,
        options={"return_trace": True, "case_limit": 5, "use_memory": True, "history_limit": 6},
    )


def _execute_agent_request(request: AgentRunRequest, final_user_input: str) -> None:
    del final_user_input

    if st.session_state.get(RUNNING_KEY):
        st.info("当前已有分析任务正在执行，请稍候。")
        return

    st.session_state[RUNNING_KEY] = True
    if hasattr(st, "toast"):
        st.toast("已提交分析任务，反诈助手正在处理...", icon="⏳")

    response = None
    try:
        if hasattr(st, "status"):
            with st.status("反诈助手正在分析...", expanded=True) as status:
                st.write("正在理解你的问题...")
                st.write("正在判断任务类型与风险场景...")
                st.write("正在调用反诈工具并生成建议...")
                response = run_agent(request)
                if response.success:
                    status.update(label="分析完成", state="complete")
                else:
                    status.update(label="分析已完成，但还需要补充信息", state="error")
        else:
            with st.spinner("反诈助手正在分析，请稍候..."):
                response = run_agent(request)
    except Exception:
        st.error("分析失败，请稍后重试或补充更明确的信息。")
        st.warning("建议补充短信原文、截图文字或具体场景；若涉及转账、验证码、陌生链接，请先暂停操作。")
        return
    finally:
        st.session_state[RUNNING_KEY] = False

    if response is None:
        return

    active_session_id = response.session_id or request.session_id
    st.session_state[LAST_RESPONSE_KEY] = response
    st.session_state[SESSION_KEY] = active_session_id
    st.session_state[SESSION_LIST_KEY] = active_session_id
    _maybe_update_session_title(active_session_id)
    _schedule_form_reset()
    if hasattr(st, "toast"):
        toast_message = "分析完成，可继续在当前会话追问。" if response.success else "分析已返回结果，你可以补充信息后继续追问。"
        st.toast(toast_message)
    st.rerun()


def _render_image_upload_area(session_id: str) -> None:
    is_running = bool(st.session_state.get(RUNNING_KEY))
    form_version = st.session_state.get(FORM_VERSION_KEY, 0)

    with st.expander("上传截图分析", expanded=False):
        st.caption("主输入框用于自然语言追问；如需分析聊天截图、转账页面或可疑图片，可在这里上传。")
        uploaded_image = st.file_uploader(
            "上传截图",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"agent_image_upload_{form_version}",
            disabled=is_running,
        )
        image_prompt = st.text_area(
            "图片说明（可选）",
            key=INPUT_KEY,
            height=100,
            disabled=is_running,
            placeholder="例如：请分析这张聊天截图是否存在诈骗风险。",
        )
        submitted = st.button("分析截图", type="primary", use_container_width=True, disabled=is_running)

        if not submitted:
            return

        if uploaded_image is None:
            st.warning("请先上传需要分析的截图。")
            return

        final_user_input = image_prompt.strip() or "请分析这张截图是否存在诈骗风险。"
        request = _build_agent_request(session_id, final_user_input, input_type="image", image=uploaded_image)
        _execute_agent_request(request, final_user_input)


def _render_input_area(session_id: str) -> None:
    is_running = bool(st.session_state.get(RUNNING_KEY))
    form_version = st.session_state.get(FORM_VERSION_KEY, 0)

    prompt = st.session_state.get(CHAT_INPUT_KEY, "")
    submitted = False

    with stylable_container(
        key=f"agent_inline_composer_{form_version}",
        css_styles="""
        {
            width: 100%;
            margin-top: 0.92rem;
            padding: 0.72rem 0.85rem;
            background: var(--agent-surface);
            box-shadow: var(--agent-shadow-soft);
            border: 1px solid var(--agent-border-faint);
            border-radius: 24px;
        }
        > div:first-child {
            gap: 0.72rem;
        }
        [data-testid="column"] {
            display: flex;
            align-items: center;
        }
        [data-testid="column"]:nth-child(1),
        [data-testid="column"]:nth-child(3) {
            justify-content: center;
        }
        [data-testid="stPopover"] button,
        .stButton > button {
            width: 46px;
            min-width: 46px;
            height: 46px;
            min-height: 46px;
            border-radius: 999px;
            border: none !important;
            padding: 0 !important;
            line-height: 1;
            box-shadow: none;
        }
        [data-testid="stPopover"] button svg {
            display: none !important;
        }
        [data-testid="stPopover"] button {
            background: var(--agent-user-surface) !important;
            color: var(--agent-text) !important;
            font-size: 1.25rem !important;
            font-weight: 500 !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #4b9cff 0%, #2f7eea 100%);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
        }
        [data-testid="stTextInput"] {
            width: 100%;
            margin-bottom: 0;
        }
        [data-testid="stTextInput"] > div {
            width: 100%;
        }
        div[data-baseweb="input"] {
            border: none;
            background: transparent;
            box-shadow: none;
            border-radius: 0;
            min-height: 46px;
        }
        div[data-baseweb="input"] > div {
            border: none;
            background: transparent;
            box-shadow: none;
            border-radius: 0;
            padding: 0;
        }
        input {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            color: var(--agent-input-text);
            font-size: 1rem;
            height: 46px;
            padding: 0 0.2rem !important;
        }
        input::placeholder {
            color: var(--agent-placeholder);
        }
        input:focus {
            outline: none;
            box-shadow: none;
        }
        hr {
            display: none;
        }
        """,
    ):
        plus_col, input_col, send_col = st.columns([0.7, 10, 0.7], gap="small", vertical_alignment="center")
        with plus_col:
            with st.popover("＋", disabled=is_running):
                st.markdown('<div class="agent-upload-popover">', unsafe_allow_html=True)
                st.caption("上传聊天截图、转账页面或可疑短信图片")
                st.file_uploader(
                    "上传截图",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"{UPLOAD_INPUT_KEY}_{form_version}",
                    label_visibility="collapsed",
                    disabled=is_running,
                )
                st.markdown(
                    '<p class="agent-upload-help">上传后直接点击右侧发送，我会结合文字和截图一起分析。</p>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with input_col:
            prompt = st.text_input(
                "输入内容",
                key=CHAT_INPUT_KEY,
                label_visibility="collapsed",
                disabled=is_running,
                placeholder="继续追问，或点 + 上传截图让我帮你分析",
            )

        with send_col:
            submitted = st.button("\u27A4", key=f"agent_send_button_{form_version}", type="primary", disabled=is_running)

    uploaded_image = st.session_state.get(f"{UPLOAD_INPUT_KEY}_{form_version}")
    active_upload = uploaded_image is not None
    st.session_state[UPLOAD_INPUT_KEY] = active_upload

    if active_upload:
        st.markdown(
            '<div class="agent-composer-note"><span class="agent-upload-pill">已选择截图，发送时会一起分析</span></div>',
            unsafe_allow_html=True,
        )

    if not submitted:
        return

    normalized_prompt = (prompt or "").strip()
    if uploaded_image is not None:
        final_user_input = normalized_prompt or "请结合这张截图帮我判断是否存在诈骗风险。"
        request = _build_agent_request(session_id, final_user_input, input_type="image", image=uploaded_image)
        _execute_agent_request(request, final_user_input)
        return

    if not normalized_prompt:
        st.warning("请输入要追问的内容，或先通过左侧 + 上传截图。")
        return

    request = _build_agent_request(session_id, normalized_prompt, input_type="text")
    _execute_agent_request(request, normalized_prompt)


def _render_sidebar(current_session_id: str) -> None:
    _render_sidebar_styles()
    with st.sidebar:
        gutter_left, header_left, header_right, gutter_right = st.columns(
            [0.36, 2.68, 2.60, 0.36],
            gap="small",
            vertical_alignment="center",
        )
        with gutter_left:
            st.markdown("&nbsp;", unsafe_allow_html=True)
        with header_left:
            st.markdown("### 最近")
        with header_right:
            with stylable_container(
                key="new_session_header_button",
                css_styles="""
                {
                    display: flex;
                    justify-content: flex-end;
                    padding-right: 0.28rem;
                    padding-top: 2px;
                    transform: translateX(5px);
                }
                .stButton > button {
                    width: 100% !important;
                    min-width: 0 !important;
                    min-height: 38px;
                    border-radius: 12px;
                    padding: 0.28rem 1.22rem 0.28rem 0.7rem;
                    border: 1px solid transparent;
                    box-shadow: none;
                    overflow: hidden !important;
                    color: var(--agent-accent-strong);
                    font-weight: 600;
                }
                .stButton > button:hover,
                .stButton > button:focus,
                .stButton > button:focus-visible,
                .stButton > button:active {
                    color: var(--agent-accent);
                    background: color-mix(in srgb, var(--agent-accent) 10%, transparent);
                    border-radius: 12px !important;
                    outline: none !important;
                    box-shadow: none !important;
                }
                .stButton > button p {
                    color: var(--agent-accent-strong);
                    font-weight: 600;
                    white-space: nowrap;
                }
                .stButton > button:hover p {
                    color: var(--agent-accent);
                }
                """,
            ):
                if st.button(_sidebar_button_title("+ 新建会话"), use_container_width=True, type="secondary"):
                    _create_new_session()
                    st.rerun()

        with gutter_right:
            st.markdown("&nbsp;", unsafe_allow_html=True)

        sessions = _get_recent_sessions(current_session_id)
        if sessions:
            for session in sessions:
                _render_sidebar_session_item(session, current_session_id)
        else:
            st.caption("当前还没有历史会话，新建后会自动显示在这里。")

        st.markdown("---")
        _render_task_history(current_session_id)

_ensure_page_state()
session_id = _ensure_session()
current_session = _get_session(session_id)

_render_sidebar(session_id)
_render_page_styles()
_render_page_header()

last_response = st.session_state.get(LAST_RESPONSE_KEY)
session_response = last_response if last_response is not None and getattr(last_response, "session_id", None) == session_id else None

_render_conversation(session_id, session_response)
_render_input_area(session_id)
