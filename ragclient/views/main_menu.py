from __future__ import annotations

from typing import Any, Optional

import requests
import streamlit as st

from ragclient.logger import logger
from ragclient.state import View, set_view
from ragclient.views.common import emojify_robot

__all__ = ["render_main_menu"]


def _check_service_health(url: str) -> Optional[dict[str, Any]]:
    """ヘルスチェックエンドポイントへアクセスし、サービス稼働状況を取得する。

    Args:
        url (str): ヘルスチェック URL

    Returns:
        Optional[dict[str, Any]]: 応答 JSON（失敗時は None）
    """
    logger.debug("trace")

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        logger.exception("health check failed for %s: %s", url, e)
        return None

    try:
        data = res.json()
    except ValueError as e:
        logger.exception("health check response is not json for %s: %s", url, e)
        return None

    if not isinstance(data, dict):
        logger.warning("health check response is not a dict for %s", url)
        return None

    return data


def _summarize_status(
    ragserver_stat: Optional[dict[str, Any]],
) -> dict[str, str]:
    """ヘルスチェック結果を表示用テキストへまとめる。

    Args:
        ragserver_stat (Optional[dict[str, Any]]): ragserver の状態

    Returns:
        dict[str, str]: サービスの状態表示テキスト
    """
    logger.debug("trace")

    return {
        "ragserver": (
            "✅ Online ("
            + ", ".join(
                [
                    f"store: {ragserver_stat.get('store', 'N/A')}",
                    f"embed: {ragserver_stat.get('embed', 'N/A')}",
                    f"rerank: {ragserver_stat.get('rerank', 'N/A')}",
                ]
            )
            + ")"
            if ragserver_stat and ragserver_stat.get("status") == "ok"
            else "🛑 Offline"
        )
    }


def _refresh_status(ragserver_health: str) -> None:
    """サービス状態を再取得し、セッションステートへ保存する。

    Args:
        ragserver_health (str): ragserver のヘルスチェック URL
    """
    logger.debug("trace")

    try:
        ragserver_stat = _check_service_health(ragserver_health)
        texts = _summarize_status(ragserver_stat)
        st.session_state["status_texts"] = texts
        st.session_state["status_dirty"] = False
    except Exception as e:
        logger.exception(e)

        _DEFAULT_STATUS_TEXT = "不明"
        st.session_state["status_texts"] = {"ragserver": _DEFAULT_STATUS_TEXT}


def _render_status_section(ragserver_health: str) -> None:
    """メインメニューに表示するステータスセクションを描画する。

    Args:
        ragserver_health (str): ragserver のヘルスチェック URL
    """
    logger.debug("trace")

    if st.session_state.get("status_dirty", False):
        _refresh_status(ragserver_health)

    st.subheader("🩺 サービスステータス")
    texts = st.session_state["status_texts"]
    st.write(f"RAG サーバー: {texts['ragserver']}")
    st.button(
        "🔄 最新情報を取得",
        on_click=_refresh_status,
        args=(ragserver_health,),
    )


def render_main_menu(ragserver_health: str) -> None:
    """メインメニュー画面を描画する。

    Args:
        ragserver_health (str): ragserver のヘルスチェック URL
    """
    logger.debug("trace")

    st.title("📚 RAG Client")
    _render_status_section(ragserver_health)

    st.subheader("🧭 メニュー")
    st.button("📝 ナレッジ登録へ", on_click=set_view, args=(View.INGEST,))
    st.button("🔍 検索画面へ", on_click=set_view, args=(View.SEARCH,))
    st.button(
        emojify_robot("🤖 RAG 検索画面へ"), on_click=set_view, args=(View.RAGSEARCH,)
    )
    st.button("🛠️ 管理メニューへ", on_click=set_view, args=(View.ADMIN,))
