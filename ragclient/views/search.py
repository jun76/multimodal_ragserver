from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from ragclient.api_client import RagServerClient
from ragclient.logger import logger
from ragclient.state import (
    FeedBack,
    SearchResult,
    View,
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


def _render_search_section(
    *,
    title: str,
    caption: str,
    input_func: Callable[[], Any],
    button_label: str,
    button_callback: Callable[..., None],
    button_args: Callable[[Any], tuple],
    feedback_key: FeedBack,
    result_key: SearchResult,
    result_renderer: Callable[[dict[str, Any]], None],
) -> None:
    """共通の検索フォーム描画処理を実行する。

    Args:
        title (str): セクションタイトル
        caption (str): 入力補足テキスト
        input_func (Callable[[], Any]): 入力ウィジェット生成関数
        button_label (str): 検索ボタンのラベル
        button_callback (Callable[..., None]): 検索コールバック
        button_args (Callable[[Any], tuple]): コールバックへ渡す引数生成関数
        feedback_key (FeedBack): フィードバック表示用キー
        result_key (SearchResult): 検索結果格納キー
        result_renderer (Callable[[dict[str, Any]], None]): 検索結果描画関数
    """

    st.subheader(title)
    if caption:
        st.caption(caption)

    value = input_func()
    st.button(
        button_label,
        on_click=button_callback,
        args=button_args(value),
    )

    display_feedback(feedback_key)
    result = st.session_state.get(result_key)
    if result is not None:
        result_renderer(result)


def _render_search_view_text_text(client: RagServerClient) -> None:
    """テキスト→テキスト検索を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """

    _render_search_section(
        title="📝→📝 テキストでテキストを検索",
        caption="検索ワードに似た文脈を検索します。 例：「就業規則　一覧」",
        input_func=lambda: st.text_input("検索ワード", key="text_text_query"),
        button_label="🔎 検索",
        button_callback=run_text_text_search_callback,
        button_args=lambda query: (
            client,
            query,
            SearchResult.SR_SEARCH_TEXT_TEXT,
            FeedBack.FB_SEARCH_TEXT_TEXT,
        ),
        feedback_key=FeedBack.FB_SEARCH_TEXT_TEXT,
        result_key=SearchResult.SR_SEARCH_TEXT_TEXT,
        result_renderer=lambda data: _render_query_results_text("📝 検索結果", data),
    )


def _render_search_view_text_image(client: RagServerClient) -> None:
    """テキスト→画像検索を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """

    _render_search_section(
        title="📝→🖼️ テキストで画像を検索",
        caption="検索ワードに似た画像を検索します。 例：「談笑している男女」",
        input_func=lambda: st.text_input("検索ワード", key="text_image_query"),
        button_label="🔎 検索",
        button_callback=run_text_image_search_callback,
        button_args=lambda query: (
            client,
            query,
            SearchResult.SR_SEARCH_TEXT_IMAGE,
            FeedBack.FB_SEARCH_TEXT_IMAGE,
        ),
        feedback_key=FeedBack.FB_SEARCH_TEXT_IMAGE,
        result_key=SearchResult.SR_SEARCH_TEXT_IMAGE,
        result_renderer=lambda data: _render_query_results_image("🖼️ 検索結果", data),
    )


def _render_search_view_image_image(client: RagServerClient) -> None:
    """画像→画像検索を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """

    _render_search_section(
        title="🖼️→🖼️ 画像で画像を検索",
        caption="アップロードした画像に似た画像を検索します。",
        input_func=lambda: st.file_uploader(
            "検索したい画像を選択", key="image_query_uploader"
        ),
        button_label="🔎 検索",
        button_callback=run_image_image_search_callback,
        button_args=lambda file_obj: (
            client,
            file_obj,
            SearchResult.SR_SEARCH_IMAGE_IMAGE,
            FeedBack.FB_SEARCH_IMAGE_IMAGE,
        ),
        feedback_key=FeedBack.FB_SEARCH_IMAGE_IMAGE,
        result_key=SearchResult.SR_SEARCH_IMAGE_IMAGE,
        result_renderer=lambda data: _render_query_results_image("🖼️ 検索結果", data),
    )


def _render_search_view_text_audio(client: RagServerClient) -> None:
    """テキスト→音声検索を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """

    _render_search_section(
        title="📝→🎤 テキストで音声を検索",
        caption="検索ワードに似た音声を検索します。 例：「車のクラクション」",
        input_func=lambda: st.text_input("検索ワード", key="text_audio_query"),
        button_label="🔎 検索",
        button_callback=run_text_audio_search_callback,
        button_args=lambda query: (
            client,
            query,
            SearchResult.SR_SEARCH_TEXT_AUDIO,
            FeedBack.FB_SEARCH_TEXT_AUDIO,
        ),
        feedback_key=FeedBack.FB_SEARCH_TEXT_AUDIO,
        result_key=SearchResult.SR_SEARCH_TEXT_AUDIO,
        result_renderer=lambda data: _render_query_results_audio("🎤 検索結果", data),
    )


def _render_search_view_audio_audio(client: RagServerClient) -> None:
    """音声→音声検索を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """

    _render_search_section(
        title="🎤→🎤 音声で音声を検索",
        caption="アップロードした音声に似た音声を検索します。",
        input_func=lambda: st.file_uploader(
            "検索したい音声を選択", key="audio_query_uploader"
        ),
        button_label="🔎 検索",
        button_callback=run_audio_audio_search_callback,
        button_args=lambda file_obj: (
            client,
            file_obj,
            SearchResult.SR_SEARCH_AUDIO_AUDIO,
            FeedBack.FB_SEARCH_AUDIO_AUDIO,
        ),
        feedback_key=FeedBack.FB_SEARCH_AUDIO_AUDIO,
        result_key=SearchResult.SR_SEARCH_AUDIO_AUDIO,
        result_renderer=lambda data: _render_query_results_audio("🎤 検索結果", data),
    )


def _run_text_search(
    func: Callable[[str], dict[str, Any]],
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """テキスト系検索を実行する。

    Args:
        func (Callable[[str], dict[str, Any]]): query_text または query_text_multi
        query (str): 検索クエリ
        result_key (SearchResult): 検索結果を保持するセッションキー
        feedback_key (FeedBack): フィードバック表示用キー
    """

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
        logger.exception(e)
        set_feedback(feedback_key, "error", f"検索に失敗しました: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "検索が完了しました")


def run_text_text_search_callback(
    client: RagServerClient,
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """クエリ文字列によるテキストドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (SearchResult): 検索結果を保持するセッションキー
        feedback_key (FeedBack): フィードバック表示用キー
    """

    _run_text_search(
        func=client.query_text_text,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_text_image_search_callback(
    client: RagServerClient,
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """クエリ文字列による画像ドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (SearchResult): 検索結果を保持するセッションキー
        feedback_key (FeedBack): フィードバック表示用キー
    """

    _run_text_search(
        func=client.query_text_image,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_image_image_search_callback(
    client: RagServerClient,
    file_obj: Any,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """クエリ画像による画像ドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされた画像ファイル
        result_key (SearchResult): 検索結果を保持するセッションキー
        feedback_key (FeedBack): フィードバック表示用キー
    """

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
        logger.exception(e)
        set_feedback(feedback_key, "error", f"画像検索に失敗しました: {e}")
    else:
        set_search_result(result_key, result)
        set_feedback(feedback_key, "success", "検索が完了しました")


def run_text_audio_search_callback(
    client: RagServerClient,
    query: str,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """クエリ文字列による音声ドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        query (str): 検索クエリ
        result_key (SearchResult): 検索結果を保持するセッションキー
        feedback_key (FeedBack): フィードバック表示用キー
    """

    _run_text_search(
        func=client.query_text_audio,
        query=query,
        result_key=result_key,
        feedback_key=feedback_key,
    )


def run_audio_audio_search_callback(
    client: RagServerClient,
    file_obj: Any,
    result_key: SearchResult,
    feedback_key: FeedBack,
) -> None:
    """クエリ音声による音声ドキュメント検索 API を呼び出す。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされた音声ファイル
        result_key (SearchResult): 検索結果を保持するセッションキー
        feedback_key (FeedBack): フィードバック表示用キー
    """

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
        logger.exception(e)
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
        except Exception as e:
            logger.exception(e)
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
        except Exception as e:
            logger.exception(e)
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

    st.title("🔎 検索")
    st.button(
        "⬅️ メニューに戻る", key="search_back", on_click=set_view, args=(View.MAIN,)
    )
    st.divider()

    choice_map: dict[str, Callable[[RagServerClient], None]] = {
        "ﾃｷｽﾄ📝 → ﾃｷｽﾄ📝": _render_search_view_text_text,
        "ﾃｷｽﾄ📝 → 画像🖼️": _render_search_view_text_image,
        "画像🖼️ → 画像🖼️": _render_search_view_image_image,
        "ﾃｷｽﾄ📝 → 音声🎤": _render_search_view_text_audio,
        "音声🎤 → 音声🎤": _render_search_view_audio_audio,
    }
    choice = st.sidebar.selectbox(
        "検索オプションを選択して下さい。", list(choice_map.keys())
    )

    renderer = choice_map.get(choice)
    if renderer is not None:
        renderer(client)
