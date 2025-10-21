from __future__ import annotations

from typing import Any, Optional

import requests
import streamlit as st

from ..logger import logger
from ..state import View, set_view
from .common import emojify_robot

__all__ = ["render_main_menu"]


def _check_service_health(url: str) -> Optional[dict[str, Any]]:
    """ヘルスチェックエンドポイントへアクセスし、サービス稼働状況を取得する。

    Args:
        url (str): ヘルスチェック URL

    Returns:
        Optional[dict[str, Any]]: 応答 JSON（失敗時は None）
    """

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception:
        logger.warning("no response from ragserver")
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

    try:
        ragserver_stat = _check_service_health(ragserver_health)
        texts = _summarize_status(ragserver_stat)
        st.session_state["status_texts"] = texts
        st.session_state["status_dirty"] = False
    except Exception:
        logger.warning("ragserver is not ready")

        _DEFAULT_STATUS_TEXT = "不明"
        st.session_state["status_texts"] = {"ragserver": _DEFAULT_STATUS_TEXT}


def _render_status_section(ragserver_health: str) -> None:
    """メインメニューに表示するステータスセクションを描画する。

    Args:
        ragserver_health (str): ragserver のヘルスチェック URL
    """

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

    st.title("📚 RAG Client")
    _render_status_section(ragserver_health)

    st.subheader("🧭 メニュー")
    st.button("📝 ナレッジ登録へ", on_click=set_view, args=(View.INGEST,))
    st.button("🔍 ＤＢ検索画面へ", on_click=set_view, args=(View.SEARCH,))
    st.button(
        emojify_robot("🤖 RAG 検索画面へ"), on_click=set_view, args=(View.RAGSEARCH,)
    )
    st.button("🛠️ 管理メニューへ", on_click=set_view, args=(View.ADMIN,))
