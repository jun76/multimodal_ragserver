from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any, Optional

import requests
import streamlit as st

from ragclient.api_client import RagServerClient
from ragclient.config import get_config
from ragclient.logger import logger

VIEW_MAIN = "main"
VIEW_INGEST = "ingest"
VIEW_SEARCH = "search"
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


def _check_service_health(url: str) -> Optional[dict[str, Any]]:
    """ヘルスチェック用エンドポイントへアクセスし、サービス稼働可否を返す。

    Args:
        url (str): ヘルスチェック URL

    Returns:
        Optional[dict[str, Any]]: 応答 json 辞書（取得失敗時は None）
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
    embed_stat: Optional[dict[str, Any]],
    rerank_stat: Optional[dict[str, Any]],
) -> dict[str, str]:
    """取得したヘルスチェック結果を文字列表現へまとめる。

    Args:
        ragserver_stat (Optional[dict[str, Any]]): ragserver の状態
        embed_stat (Optional[dict[str, Any]]): ローカル埋め込みサービスの状態
        rerank_stat (Optional[dict[str, Any]]): ローカルリランクサービスの状態

    Returns:
        dict[str, str]: それぞれの状態表示用テキスト
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
        ),
        "embed": (
            "✅ Online"
            if embed_stat and embed_stat.get("status") == "ok"
            else "🛑 Offline"
        ),
        "rerank": (
            "✅ Online"
            if rerank_stat and rerank_stat.get("status") == "ok"
            else "🛑 Offline"
        ),
    }


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


def _init_services() -> tuple[RagServerClient, str, str, str]:
    """設定を読み込み、API クライアントとヘルスチェック用 URL を初期化する。

    Returns:
        tuple[RagServerClient, str, str, str]: API クライアント、
          ragserver, embed, rerank のヘルスチェック URL
    """
    logger.debug("trace")

    cfg = get_config()
    client = RagServerClient(cfg.ragserver_base_url)
    ragserver_health = cfg.ragserver_base_url.rstrip("/") + "/health"
    embed_health = cfg.local_embed_base_url.rstrip("/") + "/health"
    rerank_health = cfg.local_rerank_base_url.rstrip("/") + "/health"
    return client, ragserver_health, embed_health, rerank_health


def _ensure_session_state() -> None:
    """セッション状態を初期化する。

    Returns:
        None
    """
    logger.debug("trace")

    if "view" not in st.session_state:
        st.session_state["view"] = VIEW_MAIN
    if "status_texts" not in st.session_state:
        st.session_state["status_texts"] = {
            "ragserver": "不明",
            "embed": "不明",
            "rerank": "不明",
        }
    if "status_dirty" not in st.session_state:
        st.session_state["status_dirty"] = True
    for key in _FEEDBACK_KEYS:
        st.session_state.setdefault(key, None)
    for key in _SEARCH_RESULT_KEYS:
        st.session_state.setdefault(key, None)


def _set_view(view: str) -> None:
    """表示中の画面を更新する。

    Args:
        view (str): 遷移先ビュー識別子

    Returns:
        None
    """
    logger.debug("trace")

    st.session_state["view"] = view
    if view == VIEW_MAIN:
        st.session_state["status_dirty"] = True


def _set_feedback(key: str, category: str, message: str) -> None:
    """フィードバックメッセージを設定する。

    Args:
        key (str): セッションステートのキー
        category (str): 表示カテゴリ（success|error|warning|info など）
        message (str): 表示メッセージ

    Returns:
        None
    """
    logger.debug("trace")

    st.session_state[key] = {"category": category, "message": message}


def _clear_feedback(key: str) -> None:
    """フィードバックメッセージをクリアする。

    Args:
        key (str): セッションステートのキー

    Returns:
        None
    """
    logger.debug("trace")

    st.session_state[key] = None


def _display_feedback(key: str) -> None:
    """フィードバックメッセージを表示する。

    Args:
        key (str): セッションステートのキー

    Returns:
        None
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


def _set_search_result(key: str, result: Optional[dict[str, Any]]) -> None:
    """検索結果をセッションステートに保存する。

    Args:
        key (str): セッションステートのキー
        result (Optional[dict[str, Any]]): 保存する検索結果

    Returns:
        None
    """
    logger.debug("trace")

    st.session_state[key] = result


def _clear_search_result(key: str) -> None:
    """検索結果をクリアする。

    Args:
        key (str): セッションステートのキー

    Returns:
        None
    """
    logger.debug("trace")

    st.session_state[key] = None


def _refresh_status(
    ragserver_health: str, embed_health: str, rerank_health: str
) -> None:
    """各サービスの状態を更新する。

    Args:
        ragserver_health (str): ragserver のヘルスチェック URL
        embed_health (str): ローカル埋め込みサービスのヘルスチェック URL
        rerank_health (str): ローカルリランクサービスのヘルスチェック URL

    Returns:
        None
    """
    logger.debug("trace")

    try:
        ragserver_stat = _check_service_health(ragserver_health)
        embed_stat = _check_service_health(embed_health)
        rerank_stat = _check_service_health(rerank_health)
        texts = _summarize_status(ragserver_stat, embed_stat, rerank_stat)
        st.session_state["status_texts"] = texts
        st.session_state["status_dirty"] = False
    except Exception as e:
        logger.exception(e)
        # 既定表示にフォールバック（再取得できるよう status_dirty は変更しない）
        st.session_state["status_texts"] = {
            "ragserver": "不明",
            "embed": "不明",
            "rerank": "不明",
        }


def _render_status_section(
    ragserver_health: str, embed_health: str, rerank_health: str
) -> None:
    """メインメニューのステータス表示セクションを描画する。

    Args:
        ragserver_health (str): ragserver のヘルスチェック URL
        embed_health (str): ローカル埋め込みサービスのヘルスチェック URL
        rerank_health (str): ローカルリランクサービスのヘルスチェック URL

    Returns:
        None
    """
    logger.debug("trace")

    if st.session_state.get("status_dirty", False):
        _refresh_status(ragserver_health, embed_health, rerank_health)

    st.subheader("🩺 サービスステータス")
    texts = st.session_state["status_texts"]
    st.write(f"RAG サーバー: {texts['ragserver']}")
    st.write(f"ローカル埋め込みサービス: {texts['embed']}")
    st.write(f"ローカルリランクサービス: {texts['rerank']}")
    st.button(
        "🔄 最新情報を取得",
        on_click=_refresh_status,
        args=(ragserver_health, embed_health, rerank_health),
    )


def _register_uploaded_files_callback(
    client: RagServerClient,
    files: Optional[list[Any]],
    feedback_key: str,
) -> None:
    """アップロード済みファイルを取り込み登録する。

    Args:
        client (RagServerClient): ragserver API クライアント
        files (Optional[list[Any]]): アップロードファイルのリスト
        feedback_key (str): フィードバック表示用キー

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    if not files:
        _set_feedback(feedback_key, "warning", "アップロードされたファイルがありません")
        return

    try:
        with st.spinner("ファイルを取り込み中です..."):
            saved_paths = _save_uploaded_files(client, files)
            for path in saved_paths:
                client.ingest_path(path)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"ファイルの取り込みに失敗しました: {e}")
    else:
        _set_feedback(feedback_key, "success", "取り込みが完了しました")


def _register_url_callback(
    client: RagServerClient, url_value: str, feedback_key: str
) -> None:
    """URL を取り込み登録する。

    Args:
        client (RagServerClient): ragserver API クライアント
        url_value (str): 取り込み対象の URL
        feedback_key (str): フィードバック表示用キー

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    url = (url_value or "").strip()
    if not url:
        _set_feedback(feedback_key, "warning", "URL を入力してください")
        return

    try:
        with st.spinner("URL を取り込み中です..."):
            client.ingest_url(url)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"URL の取り込みに失敗しました: {e}")
    else:
        _set_feedback(feedback_key, "success", "URL の取り込みが完了しました")


def _register_url_list_callback(
    client: RagServerClient,
    file_obj: Any,
    feedback_key: str,
) -> None:
    """URL リストファイルを取り込み登録する。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされた URL リストファイル
        feedback_key (str): フィードバック表示用キー

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    if file_obj is None:
        _set_feedback(feedback_key, "warning", "URL リストが選択されていません")
        return

    try:
        with st.spinner("URL リストを取り込み中です..."):
            saved = _save_uploaded_files(client, [file_obj])[0]
            client.ingest_url_list(saved)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"URL リストの取り込みに失敗しました: {e}")
    else:
        _set_feedback(feedback_key, "success", "URL リストの取り込みが完了しました")


def _run_text_search_callback(
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

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    _clear_search_result(result_key)

    text = (query or "").strip()
    if not text:
        _set_feedback(feedback_key, "warning", "クエリを入力してください")
        return

    try:
        with st.spinner("検索中です..."):
            result = client.query_text(text)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"検索に失敗しました: {e}")
    else:
        _set_search_result(result_key, result)
        _set_feedback(feedback_key, "success", "検索が完了しました")


def _run_multimodal_search_callback(
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

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    _clear_search_result(result_key)

    text = (query or "").strip()
    if not text:
        _set_feedback(feedback_key, "warning", "クエリを入力してください")
        return

    try:
        with st.spinner("検索中です..."):
            result = client.query_text_multi(text)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"マルチモーダル検索に失敗しました: {e}")
    else:
        _set_search_result(result_key, result)
        _set_feedback(feedback_key, "success", "検索が完了しました")


def _run_image_search_callback(
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

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    _clear_search_result(result_key)

    if file_obj is None:
        _set_feedback(feedback_key, "warning", "画像が選択されていません")
        return

    try:
        with st.spinner("画像検索中です..."):
            saved = _save_uploaded_files(client, [file_obj])[0]
            result = client.query_image(saved)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"画像検索に失敗しました: {e}")
    else:
        _set_search_result(result_key, result)
        _set_feedback(feedback_key, "success", "検索が完了しました")


def _register_local_path_callback(
    client: RagServerClient, path_value: str, feedback_key: str
) -> None:
    """ローカルパス取り込みを実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        path_value (str): 取り込み対象パス
        feedback_key (str): フィードバック表示用キー

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    path = (path_value or "").strip()
    if not path:
        _set_feedback(feedback_key, "warning", "パスを入力してください")
        return

    try:
        with st.spinner("パスを取り込み中です..."):
            client.ingest_path(path)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"パスの取り込みに失敗しました: {e}")
    else:
        _set_feedback(feedback_key, "success", "パスの取り込みが完了しました")


def _register_path_list_callback(
    client: RagServerClient,
    file_obj: Any,
    feedback_key: str,
) -> None:
    """ローカルパスリスト取り込みを実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Any): アップロードされたパスリストファイル
        feedback_key (str): フィードバック表示用キー

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)
    if file_obj is None:
        _set_feedback(feedback_key, "warning", "パスリストが選択されていません")
        return

    try:
        with st.spinner("パスリストを取り込み中です..."):
            saved = _save_uploaded_files(client, [file_obj])[0]
            client.ingest_path_list(saved)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"パスリストの取り込みに失敗しました: {e}")
    else:
        _set_feedback(feedback_key, "success", "パスリストの取り込みが完了しました")


def _reload_server_callback(client: RagServerClient, feedback_key: str) -> None:
    """リロード API を順次実行する。

    Args:
        client (RagServerClient): ragserver API クライアント
        feedback_key (str): フィードバック表示用キー

    Returns:
        None
    """
    logger.debug("trace")

    _clear_feedback(feedback_key)

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
        _set_feedback(feedback_key, "warning", str(e))
        return

    try:
        with st.spinner("リロード要求を送信中です..."):
            client.reload("store", store_name)
            client.reload("embed", embed_name)
            client.reload("rerank", rerank_name)
    except Exception as e:
        _set_feedback(feedback_key, "error", f"リロード要求に失敗しました: {e}")
    else:
        _set_feedback(feedback_key, "success", "リロード要求を送信しました")


def _render_main_menu(
    ragserver_health: str, embed_health: str, rerank_health: str
) -> None:
    """メインメニュー画面を描画する。

    Args:
        ragserver_health (str): ragserver のヘルスチェック URL
        embed_health (str): 埋め込みサービスのヘルスチェック URL
        rerank_health (str): リランクサービスのヘルスチェック URL

    Returns:
        None
    """
    logger.debug("trace")

    st.title("📚 RAG Client")
    _render_status_section(ragserver_health, embed_health, rerank_health)

    st.subheader("🧭 メニュー")
    st.button("📝 ナレッジ登録へ", on_click=_set_view, args=(VIEW_INGEST,))
    st.button("🔍 検索画面へ", on_click=_set_view, args=(VIEW_SEARCH,))
    st.button("🛠️ 管理メニューへ", on_click=_set_view, args=(VIEW_ADMIN,))


def _save_uploaded_files(client: RagServerClient, files: list[Any]) -> list[str]:
    """アップロード済みファイルを保存（ragserver 側にアップロード）する。

    Args:
        client (RagServerClient): ragserver API クライアント
        files (list[Any]): Streamlit のアップロードファイルオブジェクト

    Returns:
        list[str]: 保存したファイルパス一覧
    """
    logger.debug("trace")

    payload: list[tuple[str, bytes, Optional[str]]] = []
    for uploaded in files:
        data = uploaded.getvalue()
        payload.append((uploaded.name, data, getattr(uploaded, "type", None)))

    if not payload:
        return []

    response = client.upload(payload)
    entries = response.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("ragserver upload response is invalid")

    saved: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError("ragserver upload response item is invalid")
        save_path = item.get("save_path")
        if not isinstance(save_path, str) or save_path == "":
            raise RuntimeError("ragserver upload save_path is invalid")
        saved.append(save_path)

    if len(saved) != len(payload):
        raise RuntimeError("ragserver upload file count mismatch")

    return saved


def _render_ingest_view(client: RagServerClient) -> None:
    """ナレッジ登録画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント

    Returns:
        None
    """
    logger.debug("trace")

    st.title("📝 ナレッジ登録")
    st.button("⬅️ メニューに戻る", on_click=_set_view, args=(VIEW_MAIN,))

    st.divider()
    st.subheader("📁 ファイルをアップロードして登録")
    st.caption("ファイルを ragserver に送信し、その内容を登録します。")
    files = st.file_uploader("ファイルを選択", accept_multiple_files=True)
    st.button(
        "📁 登録",
        on_click=_register_uploaded_files_callback,
        args=(client, files, "ingest_files_feedback"),
    )
    _display_feedback("ingest_files_feedback")

    st.divider()
    st.subheader("🌐 URL を指定して登録")
    st.caption("URL を ragserver に通知し、その内容を登録します。")
    url_value = st.text_input("対象 URL", key="ingest_url_input")
    st.button(
        "🌐 登録",
        on_click=_register_url_callback,
        args=(client, url_value, "ingest_url_feedback"),
    )
    _display_feedback("ingest_url_feedback")

    st.divider()
    st.subheader("📚 URL リストをアップロードして登録")
    st.caption(
        "URL を列挙したテキストファイル（*.txt）を ragserver に送信し、その内容を登録します。"
    )
    url_list = st.file_uploader("URL リストを選択", key="url_list_uploader")
    st.button(
        "📚 登録",
        on_click=_register_url_list_callback,
        args=(client, url_list, "ingest_url_list_feedback"),
    )
    _display_feedback("ingest_url_list_feedback")


def _render_query_results_text(title: str, result: dict[str, Any]) -> None:
    """テキスト検索結果を描画する。

    Args:
        title (str): セクションタイトル
        result (dict[str, Any]): ragserver からの検索結果

    Returns:
        None
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

    Returns:
        None
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


def _render_search_view(client: RagServerClient) -> None:
    """検索画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント

    Returns:
        None
    """
    logger.debug("trace")

    st.title("🔎 検索")
    st.button(
        "⬅️ メニューに戻る", key="search_back", on_click=_set_view, args=(VIEW_MAIN,)
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
            on_click=_run_text_search_callback,
            args=(client, text_query, "text_search_result", "search_text_feedback"),
        )
        _display_feedback("search_text_feedback")
        text_result = st.session_state.get("text_search_result")
        if text_result is not None:
            _render_query_results_text("📝 検索結果", text_result)

    elif choice == choice_text_image:
        st.subheader(f"{choice_text_image} テキストで画像を検索")
        st.caption("検索ワードに似た画像を検索します。 例：「談笑している男女」")
        multi_query = st.text_input("検索ワード", key="multi_query")
        st.button(
            "🔎 検索",
            on_click=_run_multimodal_search_callback,
            args=(client, multi_query, "multi_search_result", "search_multi_feedback"),
        )
        _display_feedback("search_multi_feedback")
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
            on_click=_run_image_search_callback,
            args=(
                client,
                image_file,
                "image_search_result",
                "search_image_feedback",
            ),
        )
        _display_feedback("search_image_feedback")
        image_result = st.session_state.get("image_search_result")
        if image_result is not None:
            _render_query_results_image("🖼️ 検索結果", image_result)


def _render_admin_view(client: RagServerClient) -> None:
    """管理者メニュー画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント

    Returns:
        None
    """
    logger.debug("trace")

    st.title("🛠️ 管理メニュー")
    st.button(
        "⬅️ メニューに戻る", key="admin_back", on_click=_set_view, args=(VIEW_MAIN,)
    )

    st.divider()
    st.subheader("🗂️ ragserver ローカルパスを指定して登録")
    st.caption("ragserver 側に配置済みのファイルやフォルダからナレッジ登録します。")
    path_value = st.text_input("対象パス", key="admin_path")
    st.button(
        "🗂️ 登録",
        on_click=_register_local_path_callback,
        args=(client, path_value, "admin_path_feedback"),
    )
    _display_feedback("admin_path_feedback")

    st.divider()
    st.subheader("📄 ragserver ローカルパスリストをアップロードして登録")
    st.caption(
        "ragserver 側に配置済みのファイルやフォルダ名のリスト（*.txt）からナレッジ登録します。"
    )
    path_list = st.file_uploader("パスリストを選択", key="admin_path_list")
    st.button(
        "📄 登録",
        on_click=_register_path_list_callback,
        args=(client, path_list, "admin_path_list_feedback"),
    )
    _display_feedback("admin_path_list_feedback")

    st.divider()
    st.subheader("🔁 サーバ設定リロード")
    st.caption("ragserver へリロード要求を送信します")
    st.multiselect("ベクトルストア", options=["chroma", "pgvector"], key="admin_vs")
    st.multiselect(
        "埋め込みプロバイダ", options=["local", "openai", "cohere"], key="admin_embed"
    )
    st.multiselect(
        "リランクプロバイダ", options=["local", "cohere", "none"], key="admin_rerank"
    )
    st.button(
        "🔁 サーバをリロード",
        on_click=_reload_server_callback,
        args=(client, "admin_reload_feedback"),
    )
    _display_feedback("admin_reload_feedback")


def main() -> None:
    """Streamlit アプリのエントリポイント。

    Returns:
        None
    """
    logger.debug("trace")

    st.set_page_config(page_title="RAG Client", page_icon="🧠", layout="wide")
    _ensure_session_state()

    client, ragserver_health, embed_health, rerank_health = _init_services()

    view = st.session_state.get("view", VIEW_MAIN)
    if view == VIEW_MAIN:
        _render_main_menu(ragserver_health, embed_health, rerank_health)
    elif view == VIEW_INGEST:
        _render_ingest_view(client)
    elif view == VIEW_SEARCH:
        _render_search_view(client)
    elif view == VIEW_ADMIN:
        _render_admin_view(client)
    else:
        st.error("未定義の画面です")


if __name__ == "__main__":
    main()
