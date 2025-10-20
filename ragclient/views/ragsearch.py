from __future__ import annotations

import streamlit as st

from ragclient.agent import configure_agent_context, execute_rag_search
from ragclient.api_client import RagServerClient
from ragclient.config.config import Config
from ragclient.config.settings import LLMProvider
from ragclient.logger import logger
from ragclient.state import View, set_view
from ragclient.views.common import emojify_robot

__all__ = ["render_ragsearch_view"]


def _resolve_provider_selection(selected: list[str]) -> LLMProvider:
    """マルチセレクトで選択されたプロバイダを 1 件へ確定する。

    Args:
        selected (list[str]): ユーザが選択したプロバイダ一覧

    Raises:
        ValueError: 選択数が 1 件でない場合

    Returns:
        LLMProvider: 利用するプロバイダ
    """
    logger.debug("trace")

    if not selected:
        raise ValueError("provider must be selected")
    if len(selected) != 1:
        raise ValueError("select exactly one provider")

    return LLMProvider(selected[0])


def render_ragsearch_view(client: RagServerClient) -> None:
    """RAG 検索画面を描画する。

    Args:
        client (RagServerClient): ragserver API クライアント
    """
    logger.debug("trace")

    st.title(emojify_robot("🤖 RAG 検索"))
    st.button(
        "⬅️ メニューに戻る", key="ragsearch_back", on_click=set_view, args=(View.MAIN,)
    )
    st.divider()

    default_selection = Config.llm_provider
    provider_options = ["local", "openai"]
    selected_providers = st.multiselect(
        "使用する LLM プロバイダを選択してください。",
        options=provider_options,
        default=default_selection,
        key="ragsearch_provider",
    )

    question = st.text_area("質問文", key="ragsearch_question")
    image_file = st.file_uploader("参考画像", key="ragsearch_image")
    st.session_state.setdefault("ragsearch_answer", None)

    if st.button("送信", key="ragsearch_submit"):
        if not question.strip():
            st.warning("質問文を入力してください")
        else:
            try:
                provider = _resolve_provider_selection(selected_providers)
                logger.info(f"llm provider = {provider}")
            except ValueError as e:
                logger.exception(e)
                st.warning("LLM プロバイダは 1 件のみ選択してください")
            else:
                configure_agent_context(client, image_file)
                try:
                    with st.spinner("RAG 検索を実行しています..."):
                        answer = execute_rag_search(question, provider)
                except Exception as e:
                    logger.exception(e)
                    st.error(f"RAG 検索に失敗しました: {e}")
                else:
                    st.session_state["ragsearch_answer"] = emojify_robot(answer)
                    st.success("RAG 検索が完了しました")

    if st.session_state.get("ragsearch_answer"):
        st.divider()
        st.header("🧠 最終回答")
        st.write(st.session_state["ragsearch_answer"], unsafe_allow_html=True)
