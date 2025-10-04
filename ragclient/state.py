from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from ragclient.logger import logger

VIEW_MAIN = "main"
VIEW_INGEST = "ingest"
VIEW_SEARCH = "search"
VIEW_RAGSEARCH = "ragsearch"
VIEW_ADMIN = "admin"

_FEEDBACK_KEYS = [
    "ingest_files_feedback",
    "ingest_url_feedback",
    "ingest_url_list_feedback",
    "search_text_feedback",
    "search_multi_feedback",
    "search_image_feedback",
    "admin_path_feedback",
    "admin_path_list_feedback",
    "admin_reload_feedback",
]

_SEARCH_RESULT_KEYS = [
    "text_search_result",
    "multi_search_result",
    "image_search_result",
]

__all__ = [
    "VIEW_MAIN",
    "VIEW_INGEST",
    "VIEW_SEARCH",
    "VIEW_RAGSEARCH",
    "VIEW_ADMIN",
    "ensure_session_state",
    "set_view",
    "set_feedback",
    "clear_feedback",
    "display_feedback",
    "set_search_result",
    "clear_search_result",
]


def ensure_session_state() -> None:
    """Streamlit のセッション状態を初期化する。"""
    logger.debug("trace")

    _DEFAULT_STATUS_TEXT = "不明"
    if "view" not in st.session_state:
        st.session_state["view"] = VIEW_MAIN

    if "status_texts" not in st.session_state:
        st.session_state["status_texts"] = {
            "ragserver": _DEFAULT_STATUS_TEXT,
            "embed": _DEFAULT_STATUS_TEXT,
            "rerank": _DEFAULT_STATUS_TEXT,
        }

    if "status_dirty" not in st.session_state:
        st.session_state["status_dirty"] = True

    for key in _FEEDBACK_KEYS:
        st.session_state.setdefault(key, None)

    for key in _SEARCH_RESULT_KEYS:
        st.session_state.setdefault(key, None)


def set_view(view: str) -> None:
    """表示する画面を更新する。

    Args:
        view (str): 遷移先ビュー識別子
    """
    logger.debug("trace")

    st.session_state["view"] = view
    if view == VIEW_MAIN:
        st.session_state["status_dirty"] = True


def set_feedback(key: str, category: str, message: str) -> None:
    """フィードバックメッセージをセッションに設定する。

    Args:
        key (str): セッションステートのキー
        category (str): 表示カテゴリ
        message (str): 表示メッセージ
    """
    logger.debug("trace")

    st.session_state[key] = {"category": category, "message": message}


def clear_feedback(key: str) -> None:
    """指定キーのフィードバックメッセージを消去する。

    Args:
        key (str): セッションステートのキー
    """
    logger.debug("trace")

    st.session_state[key] = None


def display_feedback(key: str) -> None:
    """保持しているフィードバックメッセージを Streamlit 上に表示する。

    Args:
        key (str): セッションステートのキー
    """
    logger.debug("trace")

    payload = st.session_state.get(key)
    if not payload:
        return

    category = payload.get("category", "")
    message = payload.get("message", "")

    if category == "success":
        st.success(message)
    elif category == "error":
        st.error(message)
    elif category == "warning":
        st.warning(message)
    elif category == "info":
        st.info(message)
    else:
        logger.warning(f"undefined category: {category}")


def set_search_result(key: str, result: Optional[dict[str, Any]]) -> None:
    """検索結果をセッションに保存する。

    Args:
        key (str): セッションステートのキー
        result (Optional[dict[str, Any]]): 保存する検索結果
    """
    logger.debug("trace")

    st.session_state[key] = result


def clear_search_result(key: str) -> None:
    """保持している検索結果を消去する。

    Args:
        key (str): セッションステートのキー
    """
    logger.debug("trace")

    st.session_state[key] = None
