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
    "run_text_text_search_callback",
    "run_text_image_search_callback",
    "run_image_image_search_callback",
    "run_text_audio_search_callback",
    "run_audio_audio_search_callback",
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


def run_text_text_search_callback(
    client: RagServerClient,
    query: str,
    result_key: str,
    feedback_key: str,
) -> None:
    """クエリ文字列によるテキストドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    _run_text_search(
        func=client.query_text_text,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_text_image_search_callback(
    client: RagServerClient,
    query: str,
    result_key: str,
    feedback_key: str,
) -> None:
    """クエリ文字列による画像ドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    _run_text_search(
        func=client.query_text_image,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_image_image_search_callback(
    client: RagServerClient,
    file_obj: Any,
    result_key: str,
    feedback_key: str,
) -> None:
    """クエリ画像による画像ドキュメント検索 API を呼び出す。

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
            result = client.query_image_image(saved)
    except Exception as e:
        set_feedback(feedback_key, "error", f"画像検索に失敗しました: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "検索が完了しました")


def run_text_audio_search_callback(
    client: RagServerClient,
    query: str,
    result_key: str,
    feedback_key: str,
) -> None:
    """クエリ文字列による音声ドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    _run_text_search(
        func=client.query_text_audio,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_audio_audio_search_callback(
    client: RagServerClient,
    file_obj: Any,
    result_key: str,
    feedback_key: str,
) -> None:
    """クエリ音声による音声ドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされた音声ファイル
        result_key (str): 検索結果を保持するセッションキー
        feedback_key (str): フィードバック表示用キー
    """
    logger.debug("trace")

    clear_feedback(feedback_key)
    clear_search_result(result_key)

    if file_obj is None:
        set_feedback(feedback_key, "warning", "音声が選択されていません")
        return

    try:
        with st.spinner("音声検索中です..."):
            saved = save_uploaded_files(client, [file_obj])[0]
            result = client.query_audio_audio(saved)
    except Exception as e:
        set_feedback(feedback_key, "error", f"音声検索に失敗しました: {e}")
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

    for doc in documents:
        metadata = doc.get("metadata", {})
        content = doc.get("text", "")
        source = metadata.get("file_path", "") or metadata.get("url", "")  # 空でない方

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

    for doc in documents:
        metadata = doc.get("metadata", {})
        source = metadata.get("file_path", "") or metadata.get("url", "")  # 空でない方

        st.divider()
        try:
            st.image(source, width="content")
        except:
            st.warning("ファイル埋め込み画像等のため、表示できません。")

        st.markdown("##### ソース")
        st.write(source)

        base_source = metadata.get("base_source", "")
        if base_source and source != base_source:
            st.write(f"出典：{base_source}")


def _render_query_results_audio(title: str, result: dict[str, Any]) -> None:
    """音声検索結果を描画する。

    Args:
        title (str): セクションタイトル
        result (dict[str, Any]): ragserver からの検索結果
    """
    logger.debug("trace")

    st.subheader(title)
    documents = result.get("documents") if isinstance(result, dict) else None
    if not documents:
        st.info("該当する音声はありません")
        return

    for doc in documents:
        metadata = doc.get("metadata", {})
        source = metadata.get("file_path", "") or metadata.get("url", "")  # 空でない方

        st.divider()
        try:
            # FIXME: フォーマット決め打ち
            st.audio(data=source, format="audio/mp3")
        except:
            st.warning("ファイル埋め込み音声等のため、表示できません。")

        st.markdown("##### ソース")
        st.write(source)

        base_source = metadata.get("base_source", "")
        if base_source and source != base_source:
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
    choice_text_audio = "📝→🎤"
    choice_audio_audio = "🎤→🎤"
    options = [
        choice_text_text,
        choice_text_image,
        choice_image_image,
        choice_text_audio,
        choice_audio_audio,
    ]
    choice = st.sidebar.selectbox("検索オプションを選択して下さい。", options)

    if choice == choice_text_text:
        st.subheader(f"{choice_text_text} テキストでテキストを検索")
        st.caption("検索ワードに似た文脈を検索します。 例：「就業規則　一覧」")
        text_text_query = st.text_input("検索ワード", key="text_text_query")
        st.button(
            "🔎 検索",
            on_click=run_text_text_search_callback,
            args=(
                client,
                text_text_query,
                "text_text_search_result",
                "search_text_text_feedback",
            ),
        )
        display_feedback("search_text_text_feedback")
        text_text_result = st.session_state.get("text_text_search_result")
        if text_text_result is not None:
            _render_query_results_text("📝 検索結果", text_text_result)

    elif choice == choice_text_image:
        st.subheader(f"{choice_text_image} テキストで画像を検索")
        st.caption("検索ワードに似た画像を検索します。 例：「談笑している男女」")
        text_image_query = st.text_input("検索ワード", key="text_image_query")
        st.button(
            "🔎 検索",
            on_click=run_text_image_search_callback,
            args=(
                client,
                text_image_query,
                "text_image_search_result",
                "search_text_image_feedback",
            ),
        )
        display_feedback("search_text_image_feedback")
        text_image_result = st.session_state.get("text_image_search_result")
        if text_image_result is not None:
            _render_query_results_image("🖼️ 検索結果", text_image_result)

    elif choice == choice_image_image:
        st.subheader(f"{choice_image_image} 画像で画像を検索")
        st.caption("アップロードした画像に似た画像を検索します。")
        image_file = st.file_uploader(
            "検索したい画像を選択", key="image_query_uploader"
        )
        st.button(
            "🔎 検索",
            on_click=run_image_image_search_callback,
            args=(
                client,
                image_file,
                "image_image_search_result",
                "search_image_image_feedback",
            ),
        )
        display_feedback("search_image_image_feedback")
        image_image_result = st.session_state.get("image_image_search_result")
        if image_image_result is not None:
            _render_query_results_image("🖼️ 検索結果", image_image_result)

    elif choice == choice_text_audio:
        st.subheader(f"{choice_text_audio} テキストで音声を検索")
        st.caption("検索ワードに似た音声を検索します。 例：「車のクラクション」")
        text_audio_query = st.text_input("検索ワード", key="text_audio_query")
        st.button(
            "🔎 検索",
            on_click=run_text_audio_search_callback,
            args=(
                client,
                text_audio_query,
                "text_audio_search_result",
                "search_text_audio_feedback",
            ),
        )
        display_feedback("search_text_audio_feedback")
        text_audio_result = st.session_state.get("text_audio_search_result")
        if text_audio_result is not None:
            _render_query_results_audio("🎤 検索結果", text_audio_result)

    elif choice == choice_audio_audio:
        st.subheader(f"{choice_audio_audio} 音声で音声を検索")
        st.caption("アップロードした音声に似た音声を検索します。")
        audio_file = st.file_uploader(
            "検索したい音声を選択", key="audio_query_uploader"
        )
        st.button(
            "🔎 検索",
            on_click=run_audio_audio_search_callback,
            args=(
                client,
                audio_file,
                "audio_audio_search_result",
                "search_audio_audio_feedback",
            ),
        )
        display_feedback("search_audio_audio_feedback")
        audio_audio_result = st.session_state.get("audio_audio_search_result")
        if audio_audio_result is not None:
            _render_query_results_audio("🎤 検索結果", audio_audio_result)
