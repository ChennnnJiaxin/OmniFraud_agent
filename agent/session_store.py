from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any

from clients.storage_client import JsonStorageClient

from agent.session_models import (
    AgentMessage,
    AgentSession,
    AgentTaskRecord,
    AgentToolTraceRecord,
    new_id,
    utc_now_iso,
)

DEFAULT_AGENT_SESSION_DIR = Path("data/agent_sessions")
_DEFAULT_STORE: AgentSessionStore | None = None
logger = logging.getLogger(__name__)


class AgentSessionStore:
    def __init__(self, root_dir: str | Path = DEFAULT_AGENT_SESSION_DIR):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, title: str | None = None, metadata: dict[str, Any] | None = None) -> AgentSession:
        now = utc_now_iso()
        session = AgentSession(
            session_id=new_id(),
            created_at=now,
            updated_at=now,
            title=title,
            metadata=dict(metadata or {}),
        )
        self._write_session_payload(session, {"messages": [], "tasks": []})
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        payload = self._load_session_payload(session_id)
        if payload is None:
            return None
        return self._session_from_dict(payload.get("session", {}))

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        payload = self._load_session_payload(session_id)
        if payload is None:
            raise ValueError(f"session not found: {session_id}")

        message = AgentMessage(
            message_id=new_id(),
            session_id=session_id,
            role=role,
            content=content,
            created_at=utc_now_iso(),
            metadata=dict(metadata or {}),
        )
        payload.setdefault("messages", []).append(asdict(message))
        self._touch_session_payload(payload)
        self._save_session_payload(session_id, payload)
        return message

    def list_messages(self, session_id: str, limit: int = 20) -> list[AgentMessage]:
        payload = self._load_session_payload(session_id)
        if payload is None:
            return []
        messages = [self._message_from_dict(item) for item in payload.get("messages", [])]
        return messages[-max(1, limit) :]

    def append_task_record(self, record: AgentTaskRecord) -> AgentTaskRecord:
        payload = self._load_session_payload(record.session_id)
        if payload is None:
            raise ValueError(f"session not found: {record.session_id}")
        payload.setdefault("tasks", []).append(self._task_to_dict(record))
        self._touch_session_payload(payload)
        self._save_session_payload(record.session_id, payload)
        return record

    def list_sessions(self, limit: int = 20) -> list[AgentSession]:
        sessions: list[AgentSession] = []
        for file_path in self.root_dir.glob("*.json"):
            try:
                payload = self._storage_for_file(file_path).load()
                session_data = payload.get("session")
                if not session_data:
                    continue
                sessions.append(self._session_from_dict(session_data))
            except Exception as exc:
                logger.warning("failed to read agent session file %s: %s", file_path, exc)

        sessions.sort(
            key=lambda session: (session.updated_at or session.created_at or "", session.created_at or ""),
            reverse=True,
        )
        return sessions[: max(1, limit)]

    def update_session_title(self, session_id: str, title: str) -> AgentSession | None:
        payload = self._load_session_payload(session_id)
        if payload is None:
            return None
        normalized_title = (title or "").strip() or None
        session_payload = payload.setdefault("session", {})
        session_payload["title"] = normalized_title
        self._touch_session_payload(payload)
        self._save_session_payload(session_id, payload)
        return self._session_from_dict(session_payload)

    def list_task_records(self, session_id: str, limit: int = 20) -> list[AgentTaskRecord]:
        payload = self._load_session_payload(session_id)
        if payload is None:
            return []
        tasks = [self._task_from_dict(item) for item in payload.get("tasks", [])]
        return tasks[-max(1, limit) :]

    def get_task_record(self, task_id: str) -> AgentTaskRecord | None:
        for file_path in self.root_dir.glob("*.json"):
            payload = self._storage_for_file(file_path).load()
            for item in payload.get("tasks", []):
                if item.get("task_id") == task_id:
                    return self._task_from_dict(item)
        return None

    def list_tool_traces(self, session_id: str, limit: int = 50) -> list[AgentToolTraceRecord]:
        traces: list[AgentToolTraceRecord] = []
        for task in self.list_task_records(session_id, limit=max(1, limit)):
            traces.extend(task.tool_trace)
        return traces[-max(1, limit) :]

    def clear_session(self, session_id: str) -> None:
        file_path = self._session_file(session_id)
        if file_path.exists():
            file_path.unlink()

    def touch_session(self, session_id: str) -> AgentSession | None:
        payload = self._load_session_payload(session_id)
        if payload is None:
            return None
        self._touch_session_payload(payload)
        self._save_session_payload(session_id, payload)
        return self._session_from_dict(payload.get("session", {}))

    def get_or_create_session(
        self,
        session_id: str | None,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentSession:
        if session_id:
            session = self.get_session(session_id)
            if session is not None:
                return session

        session = self.create_session(title=title, metadata=metadata)
        if session_id and session.session_id != session_id:
            created_payload = self._load_session_payload(session.session_id) or {}
            created_payload.setdefault("session", {})["session_id"] = session_id
            created_payload["session"]["updated_at"] = utc_now_iso()
            target_path = self._session_file(session_id)
            self._storage_for_file(target_path).save(created_payload)
            temp_path = self._session_file(session.session_id)
            if temp_path.exists():
                temp_path.unlink()
            session.session_id = session_id
            return session
        return session

    def _session_file(self, session_id: str) -> Path:
        return self.root_dir / f"{session_id}.json"

    def _storage_for_file(self, file_path: Path) -> JsonStorageClient:
        return JsonStorageClient(file_path)

    def _load_session_payload(self, session_id: str) -> dict[str, Any] | None:
        file_path = self._session_file(session_id)
        if not file_path.exists():
            return None
        storage = self._storage_for_file(file_path)
        storage.ensure_exists(default={})
        return storage.load()

    def _save_session_payload(self, session_id: str, payload: dict[str, Any]) -> None:
        storage = self._storage_for_file(self._session_file(session_id))
        storage.save(payload)

    def _write_session_payload(self, session: AgentSession, data: dict[str, Any]) -> None:
        payload = {
            "session": asdict(session),
            "messages": list(data.get("messages", [])),
            "tasks": list(data.get("tasks", [])),
        }
        self._save_session_payload(session.session_id, payload)

    def _touch_session_payload(self, payload: dict[str, Any]) -> None:
        payload.setdefault("session", {})["updated_at"] = utc_now_iso()

    def _session_from_dict(self, data: dict[str, Any]) -> AgentSession:
        return AgentSession(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            title=data.get("title"),
            metadata=dict(data.get("metadata") or {}),
        )

    def _message_from_dict(self, data: dict[str, Any]) -> AgentMessage:
        return AgentMessage(
            message_id=data.get("message_id", ""),
            session_id=data.get("session_id", ""),
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            created_at=data.get("created_at", utc_now_iso()),
            metadata=dict(data.get("metadata") or {}),
        )

    def _task_from_dict(self, data: dict[str, Any]) -> AgentTaskRecord:
        return AgentTaskRecord(
            task_id=data.get("task_id", ""),
            session_id=data.get("session_id", ""),
            user_input=data.get("user_input", ""),
            task_type=data.get("task_type", "unknown"),
            success=bool(data.get("success", False)),
            risk_level=data.get("risk_level", "unknown"),
            fraud_type=data.get("fraud_type"),
            conclusion=data.get("conclusion", ""),
            suggestions=list(data.get("suggestions") or []),
            tool_trace=[self._trace_from_dict(item) for item in data.get("tool_trace", [])],
            fallback_message=data.get("fallback_message"),
            error=data.get("error"),
            created_at=data.get("created_at", utc_now_iso()),
            output_text=data.get("output_text"),
            answer=data.get("answer"),
            evidence=list(data.get("evidence") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def _trace_from_dict(self, data: dict[str, Any]) -> AgentToolTraceRecord:
        return AgentToolTraceRecord(
            trace_id=data.get("trace_id", ""),
            task_id=data.get("task_id", ""),
            session_id=data.get("session_id", ""),
            tool_name=data.get("tool_name", ""),
            success=bool(data.get("success", False)),
            summary=data.get("summary", ""),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
            error_type=data.get("error_type"),
            handler_name=data.get("handler_name"),
            metadata=dict(data.get("metadata") or {}),
            created_at=data.get("created_at", utc_now_iso()),
        )

    def _task_to_dict(self, record: AgentTaskRecord) -> dict[str, Any]:
        return {
            **asdict(record),
            "tool_trace": [asdict(trace) for trace in record.tool_trace],
        }


def get_agent_session_store(root_dir: str | Path | None = None) -> AgentSessionStore:
    global _DEFAULT_STORE
    if root_dir is not None:
        return AgentSessionStore(root_dir)
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = AgentSessionStore(DEFAULT_AGENT_SESSION_DIR)
    return _DEFAULT_STORE
