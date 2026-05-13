from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AppConfig:
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    openai_vl_model: str | None = None
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None


def _get_streamlit_module():
    try:
        import streamlit as st  # type: ignore

        return st
    except Exception:
        return None


def _get_streamlit_value(key: str) -> Any:
    st = _get_streamlit_module()
    if st is None:
        return None

    if key in st.session_state:
        return st.session_state.get(key)

    try:
        return st.secrets.get(key)
    except Exception:
        return None


def load_app_config() -> AppConfig:
    streamlit_values = {
        "openai_api_key": _get_streamlit_value("openai_api_key") or _get_streamlit_value("OPENAI_API_KEY"),
        "openai_model": _get_streamlit_value("openai_model") or _get_streamlit_value("OPENAI_MODEL"),
        "openai_base_url": _get_streamlit_value("openai_base_url") or _get_streamlit_value("OPENAI_BASE_URL"),
        "openai_vl_model": _get_streamlit_value("openai_vl_model") or _get_streamlit_value("OPENAI_VL_MODEL"),
        "neo4j_uri": _get_streamlit_value("neo4j_uri") or _get_streamlit_value("NEO4J_URI"),
        "neo4j_username": _get_streamlit_value("neo4j_username") or _get_streamlit_value("NEO4J_USERNAME"),
        "neo4j_password": _get_streamlit_value("neo4j_password") or _get_streamlit_value("NEO4J_PASSWORD"),
        "neo4j_database": _get_streamlit_value("neo4j_database") or _get_streamlit_value("NEO4J_DATABASE"),
    }
    return AppConfig(
        openai_api_key=streamlit_values["openai_api_key"] or os.getenv("OPENAI_API_KEY"),
        openai_model=streamlit_values["openai_model"] or os.getenv("OPENAI_MODEL"),
        openai_base_url=streamlit_values["openai_base_url"] or os.getenv("OPENAI_BASE_URL"),
        openai_vl_model=streamlit_values["openai_vl_model"] or os.getenv("OPENAI_VL_MODEL"),
        neo4j_uri=streamlit_values["neo4j_uri"] or os.getenv("NEO4J_URI"),
        neo4j_username=streamlit_values["neo4j_username"] or os.getenv("NEO4J_USERNAME"),
        neo4j_password=streamlit_values["neo4j_password"] or os.getenv("NEO4J_PASSWORD"),
        neo4j_database=streamlit_values["neo4j_database"] or os.getenv("NEO4J_DATABASE"),
    )


def load_streamlit_app_config() -> AppConfig:
    return load_app_config()


def ensure_streamlit_session_config(config: AppConfig | None = None) -> AppConfig:
    config = config or load_app_config()
    st = _get_streamlit_module()
    if st is None:
        return config

    values = {
        "openai_api_key": config.openai_api_key,
        "openai_model": config.openai_model,
        "openai_base_url": config.openai_base_url,
        "openai_vl_model": config.openai_vl_model or config.openai_model,
        "neo4j_uri": config.neo4j_uri,
        "neo4j_username": config.neo4j_username,
        "neo4j_password": config.neo4j_password,
        "neo4j_database": config.neo4j_database,
    }
    for key, value in values.items():
        if value is not None:
            st.session_state[key] = value
    return config
