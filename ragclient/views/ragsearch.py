from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Optional

import streamlit as st

from ..agent import AgentExecutionError, RagAgentManager
from ..api_client import RagServerClient
from ..config.config import Config
from ..config.settings import LLMProvider
from ..state import View, set_view
from .common import emojify_robot, save_uploaded_files

__all__ = ["render_ragsearch_view"]


class RagSearchSessionKey(StrEnum):
    ANSWER = auto()
    IMAGE_PATH = auto()
    AUDIO_PATH = auto()


def _resolve_provider_selection(selected: list[str]) -> LLMProvider:
    """マルチセレクトで選択されたプロバイダを 1 件へ確定する。

    Args:
        selected (list[str]): 選択されたプロバイダ名のリスト

    Raises:
        ValueError: 選択が 0 件、または複数件、あるいは未対応プロバイダの場合

    Returns:
        LLMProvider: 利用する LLM プロバイダ
    """

    if not selected:
        raise ValueError("provider must be selected")

    if len(selected) != 1:
        raise ValueError("select exactly one provider")

    key = selected[0].upper()
    try:
        return LLMProvider[key]
    except KeyError as exc:
        raise ValueError(f"{selected[0]!r} is not a supported provider") from exc


def _save_reference_file(
    client: RagServerClient,
    file_obj: Optional[Any],
    session_key: RagSearchSessionKey,
) -> Optional[str]:
    """アップロードファイルを ragserver に保存しパスを返す。

    Args:
        client (RagServerClient): ragserver API クライアント
        file_obj (Optional[Any]): アップロードされたファイルオブジェクト
        session_key (RagSearchSessionKey): セッションステートに保存するキー

    Raises:
        AgentExecutionError: ファイルのアップロードに失敗した場合

    Returns:
        Optional[str]: 保存されたファイルパス。アップロードが無ければ None
    """

    if file_obj is None:
        st.session_state[session_key] = None
        return None

    try:
        saved = save_uploaded_files(client, [file_obj])
    except Exception as exc:  # pragma: no cover - upload side effects
        st.session_state[session_key] = None
        raise AgentExecutionError(f"failed to upload reference file: {exc}") from exc

    path = saved[0] if saved else None
    st.session_state[session_key] = path
    return path


def render_ragsearch_view(client: RagServerClient) -> None:
    """RAG 検索画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """

    st.title(emojify_robot("🤖 RAG 検索"))
    st.button(
        "⬅️ メニューに戻る", key="ragsearch_back", on_click=set_view, args=(View.MAIN,)
    )
    st.divider()

    default_provider = Config.llm_provider
    provider_options = [provider.value.lower() for provider in LLMProvider]
    selected = st.multiselect(
        "使用する LLM プロバイダを選択してください。",
        options=provider_options,
        default=[default_provider.value.lower()],
        key="ragsearch_provider",
    )

    question = st.text_area("質問文", key="ragsearch_question")

    image_file = st.file_uploader(
        "参考画像（任意）",
        type=["png", "jpg", "jpeg", "gif", "bmp"],
        key="ragsearch_image",
    )
    audio_file = st.file_uploader(
        "参考音声（任意）", type=["wav", "mp3", "flac", "ogg"], key="ragsearch_audio"
    )

    if RagSearchSessionKey.ANSWER not in st.session_state:
        st.session_state[RagSearchSessionKey.ANSWER] = None

    if st.button("送信", key="ragsearch_submit"):
        if not question.strip():
            st.warning("質問文を入力してください")
        else:
            try:
                provider = _resolve_provider_selection(selected)
            except ValueError as error:
                st.warning(str(error))
            else:
                image_path = None
                audio_path = None
                try:
                    image_path = _save_reference_file(
                        client, image_file, RagSearchSessionKey.IMAGE_PATH
                    )
                    audio_path = _save_reference_file(
                        client, audio_file, RagSearchSessionKey.AUDIO_PATH
                    )
                except AgentExecutionError as exc:
                    st.error(str(exc))
                    st.session_state[RagSearchSessionKey.ANSWER] = None
                else:
                    manager = RagAgentManager(client=client, provider=provider)
                    try:
                        with st.spinner("RAG 検索を実行しています..."):
                            answer = manager.run(
                                question=question,
                                image_path=image_path,
                                audio_path=audio_path,
                            )
                    except AgentExecutionError as exc:
                        st.session_state[RagSearchSessionKey.ANSWER] = None
                        st.error(f"RAG 検索に失敗しました: {exc}")
                    else:
                        st.session_state[RagSearchSessionKey.ANSWER] = emojify_robot(
                            answer
                        )
                        st.success("RAG 検索が完了しました")

    final_answer: Optional[str] = st.session_state.get(RagSearchSessionKey.ANSWER)
    if final_answer:
        st.divider()
        st.header("🧠 最終回答")
        st.write(final_answer, unsafe_allow_html=True)
