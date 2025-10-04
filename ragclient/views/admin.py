from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from ragclient.api_client import RagServerClient
from ragclient.logger import logger
from ragclient.state import (
    VIEW_MAIN,
    clear_feedback,
    display_feedback,
    set_feedback,
    set_view,
)
from ragclient.views.common import save_uploaded_files

__all__ = [
    "register_local_path_callback",
    "register_path_list_callback",
    "reload_server_callback",
    "render_admin_view",
]


def _pick_single_selection(values: Optional[list[str]], label: str) -> str:
    """マルチセレクトの選択値から 1 件を取得する。

    Args:
        values (Optional[list[str]]): セレクトされた値
        label (str): エラーメッセージ用の名称

    Raises:
        ValueError: 選択が 0 件または複数件の場合

    Returns:
        str: 選択された 1 件
    """
    logger.debug("trace")

    if not values:
        raise ValueError(f"{label} を選択してください")
    if len(values) != 1:
        raise ValueError(f"{label} は 1 つのみ選択してください")
    return values[0]


def register_local_path_callback(
    client: RagServerClient, path_value: str, feedback_key: str
) -> None:
    """ローカルパス取り込みを実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        path_value (str): 取り込み対象パス
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    clear_feedback(feedback_key)
    path = (path_value or "").strip()
    if not path:
        set_feedback(feedback_key, "warning", "パスを入力してください")
        return

    try:
        with st.spinner("パスを取り込み中です..."):
            client.ingest_path(path)
    except Exception as e:
        set_feedback(feedback_key, "error", f"パスの取り込みに失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "パスの取り込みが完了しました")


def register_path_list_callback(
    client: RagServerClient,
    file_obj: Any,
    feedback_key: str,
) -> None:
    """ローカルパスリスト取り込みを実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされたパスリストファイル
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    clear_feedback(feedback_key)
    if file_obj is None:
        set_feedback(feedback_key, "warning", "パスリストが選択されていません")
        return

    try:
        with st.spinner("パスリストを取り込み中です..."):
            saved = save_uploaded_files(client, [file_obj])[0]
            client.ingest_path_list(saved)
    except Exception as e:
        set_feedback(feedback_key, "error", f"パスリストの取り込みに失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "パスリストの取り込みが完了しました")


def reload_server_callback(client: RagServerClient, feedback_key: str) -> None:
    """リロード API を順次呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    clear_feedback(feedback_key)

    try:
        store_name = _pick_single_selection(
            st.session_state.get("admin_vs"), "ベクトルストア"
        )
        embed_name = _pick_single_selection(
            st.session_state.get("admin_embed"), "埋め込みプロバイダ"
        )
        rerank_name = _pick_single_selection(
            st.session_state.get("admin_rerank"), "リランクプロバイダ"
        )
    except ValueError as e:
        set_feedback(feedback_key, "warning", str(e))
        return

    try:
        with st.spinner("リロード要求を送信中です..."):
            client.reload("store", store_name)
            client.reload("embed", embed_name)
            client.reload("rerank", rerank_name)
    except Exception as e:
        set_feedback(feedback_key, "error", f"リロード要求に失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "リロード要求を送信しました")


def render_admin_view(client: RagServerClient) -> None:
    """管理者メニュー画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """
    logger.debug("trace")

    st.title("🛠️ 管理メニュー")
    st.button(
        "⬅️ メニューに戻る", key="admin_back", on_click=set_view, args=(VIEW_MAIN,)
    )

    st.divider()
    st.subheader("🗂️ ragserver ローカルパスを指定して登録")
    st.caption("ragserver 側に配置済みのファイルやフォルダからナレッジ登録します。")
    path_value = st.text_input("対象パス", key="admin_path")
    st.button(
        "🗂️ 登録",
        on_click=register_local_path_callback,
        args=(client, path_value, "admin_path_feedback"),
    )
    display_feedback("admin_path_feedback")

    st.divider()
    st.subheader("📄 ragserver ローカルパスリストをアップロードして登録")
    st.caption(
        "ragserver 側に配置済みのファイルやフォルダ名のリスト（*.txt）からナレッジ登録します。"
    )
    path_list = st.file_uploader("パスリストを選択", key="admin_path_list")
    st.button(
        "📄 登録",
        on_click=register_path_list_callback,
        args=(client, path_list, "admin_path_list_feedback"),
    )
    display_feedback("admin_path_list_feedback")

    st.divider()
    st.subheader("🔁 サーバ設定リロード")
    st.caption("ragserver へリロード要求を送信します")
    st.multiselect("ベクトルストア", options=["chroma", "pgvector"], key="admin_vs")
    st.multiselect(
        "埋め込みプロバイダ", options=["hfclip", "openai", "cohere"], key="admin_embed"
    )
    st.multiselect(
        "リランクプロバイダ", options=["hf", "cohere", "none"], key="admin_rerank"
    )
    st.button(
        "🔁 サーバをリロード",
        on_click=reload_server_callback,
        args=(client, "admin_reload_feedback"),
    )
    display_feedback("admin_reload_feedback")
