from __future__ import annotations

from typing import Any

import streamlit as st

from ragclient.api_client import RagServerClient
from ragclient.logger import logger
from ragclient.state import (
    FeedBack,
    View,
    clear_feedback,
    display_feedback,
    set_feedback,
    set_view,
)
from .common import save_uploaded_files

__all__ = [
    "register_local_path_callback",
    "register_path_list_callback",
    "render_admin_view",
]


def register_local_path_callback(
    client: RagServerClient, path_value: str, feedback_key: FeedBack
) -> None:
    """ローカルパス取り込みを実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        path_value (str): 取り込み対象パス
        feedback_key (FeedBack): フィードバック表示用キー
    """

    clear_feedback(feedback_key)
    path = (path_value or "").strip()
    if not path:
        set_feedback(feedback_key, "warning", "パスを入力してください")
        return

    try:
        with st.spinner("パスを取り込み中です..."):
            client.ingest_path(path)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"パスの取り込みに失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "パスの取り込みが完了しました")


def register_path_list_callback(
    client: RagServerClient,
    file_obj: Any,
    feedback_key: FeedBack,
) -> None:
    """ローカルパスリスト取り込みを実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされたパスリストファイル
        feedback_key (FeedBack): フィードバック表示用キー
    """

    clear_feedback(feedback_key)
    if file_obj is None:
        set_feedback(feedback_key, "warning", "パスリストが選択されていません")
        return

    try:
        with st.spinner("パスリストを取り込み中です..."):
            saved = save_uploaded_files(client, [file_obj])[0]
            client.ingest_path_list(saved)
    except Exception as e:
        logger.exception(e)
        set_feedback(feedback_key, "error", f"パスリストの取り込みに失敗しました: {e}")
    else:
        set_feedback(feedback_key, "success", "パスリストの取り込みが完了しました")


def render_admin_view(client: RagServerClient) -> None:
    """管理者メニュー画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """

    st.title("🛠️ 管理メニュー")
    st.button(
        "⬅️ メニューに戻る", key="admin_back", on_click=set_view, args=(View.MAIN,)
    )

    st.divider()
    st.subheader("🗂️ ragserver ローカルパスを指定して登録")
    st.caption("ragserver 側に配置済みのファイルやフォルダからナレッジ登録します。")
    path_value = st.text_input("対象パス", key="admin_path")
    st.button(
        "🗂️ 登録",
        on_click=register_local_path_callback,
        args=(client, path_value, FeedBack.FB_ADMIN_PATH),
    )
    display_feedback(FeedBack.FB_ADMIN_PATH)

    st.divider()
    st.subheader("📄 ragserver ローカルパスリストをアップロードして登録")
    st.caption(
        "ragserver 側に配置済みのファイルやフォルダ名のリスト（*.txt）からナレッジ登録します。"
    )
    path_list = st.file_uploader("パスリストを選択", key="admin_path_list")
    st.button(
        "📄 登録",
        on_click=register_path_list_callback,
        args=(client, path_list, FeedBack.FB_ADMIN_PATH_LIST),
    )
    display_feedback(FeedBack.FB_ADMIN_PATH_LIST)
