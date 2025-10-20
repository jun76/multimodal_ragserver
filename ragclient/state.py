from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Optional

import streamlit as st

from .logger import logger

__all__ = [
    "View",
    "FeedBack",
    "SearchResult",
    "ensure_session_state",
    "set_view",
    "set_feedback",
    "clear_feedback",
    "display_feedback",
    "set_search_result",
    "clear_search_result",
]


class View(StrEnum):
    MAIN = auto()
    INGEST = auto()
    SEARCH = auto()
    RAGSEARCH = auto()
    ADMIN = auto()


class FeedBack(StrEnum):
    # Ingest
    FB_INGEST_FILES = auto()
    FB_INGEST_URL = auto()
    FB_INGEST_URL_LIST = auto()
    # Search
    FB_SEARCH_TEXT_TEXT = auto()
    FB_SEARCH_TEXT_IMAGE = auto()
    FB_SEARCH_IMAGE_IMAGE = auto()
    FB_SEARCH_TEXT_AUDIO = auto()
    FB_SEARCH_AUDIO_AUDIO = auto()
    # RAG Search
    FB_RAGSEARCH_TEXT_TEXT = auto()
    FB_RAGSEARCH_TEXT_IMAGE = auto()
    FB_RAGSEARCH_IMAGE_IMAGE = auto()
    FB_RAGSEARCH_TEXT_AUDIO = auto()
    FB_RAGSEARCH_AUDIO_AUDIO = auto()
    # Admin
    FB_ADMIN_PATH = auto()
    FB_ADMIN_PATH_LIST = auto()


class SearchResult(StrEnum):
    # Search
    SR_SEARCH_TEXT_TEXT = auto()
    SR_SEARCH_TEXT_IMAGE = auto()
    SR_SEARCH_IMAGE_IMAGE = auto()
    SR_SEARCH_TEXT_AUDIO = auto()
    SR_SEARCH_AUDIO_AUDIO = auto()
    # RAG Search
    SR_RAGSEARCH_TEXT_TEXT = auto()
    SR_RAGSEARCH_TEXT_IMAGE = auto()
    SR_RAGSEARCH_IMAGE_IMAGE = auto()
    SR_RAGSEARCH_TEXT_AUDIO = auto()
    SR_RAGSEARCH_AUDIO_AUDIO = auto()


def ensure_session_state() -> None:
    """Streamlit のセッション状態を初期化する。"""

    _DEFAULT_STATUS_TEXT = "不明"
    current_view = st.session_state.get("view")
    if current_view is None:
        st.session_state["view"] = View.MAIN
    elif not isinstance(current_view, View):
        try:
            st.session_state["view"] = View[str(current_view).upper()]
        except KeyError:
            st.session_state["view"] = View.MAIN

    if "status_texts" not in st.session_state:
        st.session_state["status_texts"] = {
            "ragserver": _DEFAULT_STATUS_TEXT,
            "embed": _DEFAULT_STATUS_TEXT,
            "rerank": _DEFAULT_STATUS_TEXT,
        }

    if "status_dirty" not in st.session_state:
        st.session_state["status_dirty"] = True

    for key in FeedBack:
        st.session_state.setdefault(key, None)

    for key in SearchResult:
        st.session_state.setdefault(key, None)


def set_view(view: View) -> None:
    """表示する画面を更新する。

    Args:
        view (View): 遷移先ビュー識別子
    """

    st.session_state["view"] = view
    if view == View.MAIN:
        st.session_state["status_dirty"] = True


def set_feedback(key: FeedBack | str, category: str, message: str) -> None:
    """フィードバックメッセージをセッションに設定する。

    Args:
        key (FeedBack | str): セッションステートのキー
        category (str): 表示カテゴリ
        message (str): 表示メッセージ
    """

    st.session_state[key] = {"category": category, "message": message}


def clear_feedback(key: FeedBack | str) -> None:
    """指定キーのフィードバックメッセージを消去する。

    Args:
        key (FeedBack | str): セッションステートのキー
    """

    st.session_state[key] = None


def display_feedback(key: FeedBack | str) -> None:
    """保持しているフィードバックメッセージを Streamlit 上に表示する。

    Args:
        key (FeedBack | str): セッションステートのキー
    """

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


def set_search_result(
    key: SearchResult | str, result: Optional[dict[str, Any]]
) -> None:
    """検索結果をセッションに保存する。

    Args:
        key (SearchResult | str): セッションステートのキー
        result (Optional[dict[str, Any]]): 保存する検索結果
    """

    st.session_state[key] = result


def clear_search_result(key: SearchResult | str) -> None:
    """保持している検索結果を消去する。

    Args:
        key (SearchResult | str): セッションステートのキー
    """

    st.session_state[key] = None
