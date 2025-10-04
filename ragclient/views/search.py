from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from ragclient.api_client import RagServerClient
from ragclient.logger import logger
from ragclient.state import (
    VIEW_MAIN,
    clear_feedback,
    clear_search_result,
    display_feedback,
    set_feedback,
    set_search_result,
    set_view,
)
from ragclient.views.common import save_uploaded_files

__all__ = [
    "run_text_search_callback",
    "run_multimodal_search_callback",
    "run_image_search_callback",
    "render_search_view",
]


def _run_text_search(
    func: Callable[[str], dict[str, Any]],
    query: str,
    result_key: str,
    feedback_key: str,
) -> None:
    """テキスト系検索を実行する。

    Args:
        func (Callable[[str], dict[str, Any]]): query_text または query_text_multi
        query (str): 検索クエリ
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    clear_feedback(feedback_key)
    clear_search_result(result_key)

    text = (query or "").strip()
    if not text:
        set_feedback(feedback_key, "warning", "クエリを入力してください")
        return

    try:
        with st.spinner("検索中です..."):
            result = func(text)
    except Exception as e:
        logger.error(e)
        set_feedback(feedback_key, "error", f"検索に失敗しました: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "検索が完了しました")


def run_text_search_callback(
    client: RagServerClient,
    query: str,
    result_key: str,
    feedback_key: str,
) -> None:
    """テキスト検索を実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    _run_text_search(
        func=client.query_text,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_multimodal_search_callback(
    client: RagServerClient,
    query: str,
    result_key: str,
    feedback_key: str,
) -> None:
    """マルチモーダル検索を実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    _run_text_search(
        func=client.query_text_multi,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_image_search_callback(
    client: RagServerClient,
    file_obj: Any,
    result_key: str,
    feedback_key: str,
) -> None:
    """画像検索を実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされた画像ファイル
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    clear_feedback(feedback_key)
    clear_search_result(result_key)

    if file_obj is None:
        set_feedback(feedback_key, "warning", "画像が選択されていません")
        return

    try:
        with st.spinner("画像検索中です..."):
            saved = save_uploaded_files(client, [file_obj])[0]
            result = client.query_image(saved)
    except Exception as e:
        set_feedback(feedback_key, "error", f"画像検索に失敗しました: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "検索が完了しました")


def _render_query_results_text(title: str, result: dict[str, Any]) -> None:
    """テキスト検索結果を描画する。

    Args:
        title (str): セクションタイトル
        result (dict[str, Any]): ragserver からの検索結果
    """
    logger.debug("trace")

    st.subheader(title)
    documents = result.get("documents") if isinstance(result, dict) else None
    if not documents:
        st.info("該当するドキュメントはありません")
        return

    for item in documents:
        content = item.get("page_content", "")
        metadata = item.get("metadata") or {}
        source = metadata.get("source", "不明")

        st.divider()
        st.markdown("#### 本文")
        st.write(content)
        st.markdown("##### ソース")
        st.write(source)


def _render_query_results_image(title: str, result: dict[str, Any]) -> None:
    """画像検索結果を描画する。

    Args:
        title (str): セクションタイトル
        result (dict[str, Any]): ragserver からの検索結果
    """
    logger.debug("trace")

    st.subheader(title)
    documents = result.get("documents") if isinstance(result, dict) else None
    if not documents:
        st.info("該当する画像はありません")
        return

    for item in documents:
        image_url = item.get("page_content", "")
        metadata = item.get("metadata") or {}

        st.divider()
        if image_url:
            st.image(image_url, use_container_width=False)
        st.markdown("##### ソース")

        source = metadata.get("source", "不明")
        st.write(source)

        base_source = metadata.get("base_source", "不明")
        if base_source != source:
            st.write(f"出典：{base_source}")


def render_search_view(client: RagServerClient) -> None:
    """検索画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """
    logger.debug("trace")

    st.title("🔎 検索")
    st.button(
        "⬅️ メニューに戻る", key="search_back", on_click=set_view, args=(VIEW_MAIN,)
    )
    st.divider()

    choice_text_text = "📝→📝"
    choice_text_image = "📝→🖼️"
    choice_image_image = "🖼️→🖼️"
    options = [choice_text_text, choice_text_image, choice_image_image]
    choice = st.sidebar.selectbox("検索オプションを選択して下さい。", options)

    if choice == choice_text_text:
        st.subheader(f"{choice_text_text} テキストでテキストを検索")
        st.caption("検索ワードに似た文脈を検索します。 例：「就業規則　一覧」")
        text_query = st.text_input("検索ワード", key="text_query")
        st.button(
            "🔎 検索",
            on_click=run_text_search_callback,
            args=(client, text_query, "text_search_result", "search_text_feedback"),
        )
        display_feedback("search_text_feedback")
        text_result = st.session_state.get("text_search_result")
        if text_result is not None:
            _render_query_results_text("📝 検索結果", text_result)

    elif choice == choice_text_image:
        st.subheader(f"{choice_text_image} テキストで画像を検索")
        st.caption("検索ワードに似た画像を検索します。 例：「談笑している男女」")
        multi_query = st.text_input("検索ワード", key="multi_query")
        st.button(
            "🔎 検索",
            on_click=run_multimodal_search_callback,
            args=(client, multi_query, "multi_search_result", "search_multi_feedback"),
        )
        display_feedback("search_multi_feedback")
        multi_result = st.session_state.get("multi_search_result")
        if multi_result is not None:
            _render_query_results_image("🖼️ 検索結果", multi_result)

    elif choice == choice_image_image:
        st.subheader(f"{choice_image_image} 画像で画像を検索")
        st.caption("アップロードした画像に似た画像を検索します。")
        image_file = st.file_uploader(
            "検索したい画像を選択", key="image_query_uploader"
        )
        st.button(
            "🔎 検索",
            on_click=run_image_search_callback,
            args=(
                client,
                image_file,
                "image_search_result",
                "search_image_feedback",
            ),
        )
        display_feedback("search_image_feedback")
        image_result = st.session_state.get("image_search_result")
        if image_result is not None:
            _render_query_results_image("🖼️ 検索結果", image_result)
