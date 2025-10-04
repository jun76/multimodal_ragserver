from __future__ import annotations

import json
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from ragclient.api_client import RagServerClient
from ragclient.llm import get_chat_model
from ragclient.logger import logger
from ragclient.views.search import (
    run_image_search_callback,
    run_multimodal_search_callback,
    run_text_search_callback,
)

__all__ = ["configure_agent_context", "execute_rag_search"]

_TEXT_RESULT_KEY = "rag_agent_text_result"
_TEXT_FEEDBACK_KEY = "rag_agent_text_feedback"
_MULTI_RESULT_KEY = "rag_agent_multi_result"
_MULTI_FEEDBACK_KEY = "rag_agent_multi_feedback"
_IMAGE_RESULT_KEY = "rag_agent_image_result"
_IMAGE_FEEDBACK_KEY = "rag_agent_image_feedback"

_TOOL_RESULT_KEY_MAP = {
    "text_search_tool": _TEXT_RESULT_KEY,
    "multimodal_search_tool": _MULTI_RESULT_KEY,
    "image_search_tool": _IMAGE_RESULT_KEY,
}

_ACTIVE_CLIENT: RagServerClient | None = None
_ACTIVE_IMAGE: Any | None = None


def configure_agent_context(client: RagServerClient, image: Any | None) -> None:
    """エージェントが利用するクライアントと画像情報を設定する。

    Args:
        client (RagServerClient): 検索 API へ接続するクライアント
        image (Any | None): ユーザがアップロードした画像（未指定可）
    """
    logger.debug("trace")

    # FIXME: 複数クライアントだと混線
    global _ACTIVE_CLIENT
    global _ACTIVE_IMAGE

    _ACTIVE_CLIENT = client
    _ACTIVE_IMAGE = image


def _ensure_session_key(key: str) -> None:
    """セッションステートに指定キーが存在しない場合は初期化する。

    Args:
        key (str): セッションステートのキー
    """
    logger.debug("trace")

    if key not in st.session_state:
        st.session_state[key] = None


def _require_client() -> RagServerClient:
    """設定済みクライアントを取得する。

    Returns:
        RagServerClient: 検索 API クライアント

    Raises:
        RuntimeError: クライアントが未設定の場合
    """
    logger.debug("trace")

    if _ACTIVE_CLIENT is None:
        raise RuntimeError("rag agent client is not configured")
    return _ACTIVE_CLIENT


@tool
def text_search_tool(query: str) -> str:
    """テキスト検索を実行する。

    Args:
        query (str): 検索クエリ

    Returns:
        str: 結果 JSON
    """
    logger.debug("trace")

    client = _require_client()
    _ensure_session_key(_TEXT_RESULT_KEY)
    _ensure_session_key(_TEXT_FEEDBACK_KEY)
    run_text_search_callback(
        client,
        query,
        _TEXT_RESULT_KEY,
        _TEXT_FEEDBACK_KEY,
    )
    payload = st.session_state.get(_TEXT_RESULT_KEY) or {}

    return json.dumps(payload, ensure_ascii=False)


@tool
def multimodal_search_tool(query: str) -> str:
    """テキストを基点としたマルチモーダル検索を実行する。

    Args:
        query (str): 検索クエリ

    Returns:
        str: 結果 JSON
    """
    logger.debug("trace")

    client = _require_client()
    _ensure_session_key(_MULTI_RESULT_KEY)
    _ensure_session_key(_MULTI_FEEDBACK_KEY)
    run_multimodal_search_callback(
        client,
        query,
        _MULTI_RESULT_KEY,
        _MULTI_FEEDBACK_KEY,
    )
    payload = st.session_state.get(_MULTI_RESULT_KEY) or {}

    return json.dumps(payload, ensure_ascii=False)


@tool
def image_search_tool(query: str) -> str:
    """アップロード済み画像を用いた画像検索を実行する。

    Args:
        query (str): 検索クエリ

    Raises:
        ValueError: 画像無し

    Returns:
        str: 結果 JSON
    """
    logger.debug("trace")

    client = _require_client()
    if _ACTIVE_IMAGE is None:
        raise ValueError("no image uploaded")

    _ensure_session_key(_IMAGE_RESULT_KEY)
    _ensure_session_key(_IMAGE_FEEDBACK_KEY)
    run_image_search_callback(
        client,
        _ACTIVE_IMAGE,
        _IMAGE_RESULT_KEY,
        _IMAGE_FEEDBACK_KEY,
    )
    payload = st.session_state.get(_IMAGE_RESULT_KEY) or {}

    return json.dumps(payload, ensure_ascii=False)


def _build_planner_prompt() -> ChatPromptTemplate:
    """検索方針策定用プロンプトを生成する。

    Returns:
        ChatPromptTemplate: LangChain プロンプト
    """
    logger.debug("trace")

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a search planner. Use tools to gather knowledge for the user's question. "
                "Only call image_search_tool when an image is available. "
                "Provide a short english summary after using the tool. "
                "After you, responder agent will use your summary to make final answer",
            ),
            (
                "human",
                "Question: {question}\nImage hint: {image_hint}",
            ),
        ]
    )


def _build_responder_prompt() -> ChatPromptTemplate:
    """最終回答生成用プロンプトを生成する。

    Returns:
        ChatPromptTemplate: LangChain プロンプト
    """
    logger.debug("trace")

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the provided knowledge to answer the question in Japanese. "
                "Use tables and emoji to ensure your HTML text is easy to read. "
                "Please ensure that your answers utilize 'knowledge' and do not rely solely on "
                "information you already know or information obtained through web searches. "
                "So, if no relevant 'knowledge' is available, explain that no related information was found, "
                "then base your answers on information you already know or information obtained through web searches. "
                "Please also provide the reference URLs or local paths used in your response.",
            ),
            (
                "human",
                "Question: {question}\nKnowledge:\n{knowledge}\nPlanner summary: {planner_summary}\nImage hint: {image_hint}",
            ),
        ]
    )


def _collect_tool_outputs(
    tool_calls: list[Any], available_tools: list[Any]
) -> tuple[list[str], list[str]]:
    """ツール呼び出し結果を集約し、ナレッジとして返す。

    Args:
        tool_calls (list[Any]): LLM から返されたツール呼び出し定義
        available_tools (list[Any]): 利用可能なツール一覧

    Returns:
        tuple[list[str], list[str]]: 使用したツール名とナレッジ
    """
    logger.debug("trace")

    tool_map = {tool.name: tool for tool in available_tools}
    used_names: list[str] = []
    knowledge: list[str] = []

    for call in tool_calls or []:
        name = getattr(call, "name", None) or getattr(call, "tool", None)
        if isinstance(call, dict):
            name = call.get("name") or call.get("tool")
        if not isinstance(name, str):
            logger.warning("planner returned invalid tool call: %s", call)
            continue

        tool = tool_map.get(name)
        if tool is None:
            logger.warning("planner requested unknown tool: %s", name)
            continue

        raw_args = getattr(call, "args", None)
        if isinstance(call, dict):
            raw_args = call.get("args")
        try:
            args = _normalize_tool_args(raw_args)
        except Exception as e:
            logger.exception("tool %s arguments are invalid: %s", name, e)
            continue

        try:
            result = tool.invoke(args)
        except Exception as e:
            logger.exception("tool %s failed: %s", name, e)
            continue

        used_names.append(name)
        if result:
            knowledge.append(str(result))

    session_payloads = _extract_session_payloads(used_names)
    for payload in session_payloads:
        if payload not in knowledge:
            knowledge.append(payload)

    return used_names, knowledge


def _normalize_tool_args(raw: Any) -> dict[str, Any]:
    """ツール呼び出し引数を辞書へ正規化する。

    Args:
        raw (Any): ツール呼び出し引数

    Raises:
        ValueError: 無効な引数形式

    Returns:
        dict[str, Any]: 辞書形式の引数
    """
    logger.debug("trace")

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        return json.loads(text) if text else {}

    raise ValueError("tool arguments must be dict or json string")


def _extract_session_payloads(tool_names: list[str]) -> list[str]:
    """セッションステートの最新結果を JSON 文字列として抽出する。

    Args:
        tool_names (list[str]): ツール名のリスト

    Returns:
        list[str]: 抽出した JSON 文字列
    """
    logger.debug("trace")

    outputs: list[str] = []
    for name in tool_names:
        key = _TOOL_RESULT_KEY_MAP.get(name)
        if key is None:
            continue

        payload = st.session_state.get(key)
        if payload is None:
            continue

        outputs.append(json.dumps(payload, ensure_ascii=True))

    return outputs


def execute_rag_search(question: str, provider: str) -> str:
    """RAG エージェントを実行し、最終回答を生成する。

    Args:
        question (str): ユーザからの質問文
        provider (str): 利用する LLM プロバイダ

    Returns:
        str: エージェントが生成した最終回答
    """
    logger.debug("trace")

    if question.strip() == "":
        raise ValueError("question must not be empty")

    llm = get_chat_model(provider)

    # 検索プランナーエージェントのセットアップ
    planner_prompt = _build_planner_prompt()

    available_tools = [text_search_tool, multimodal_search_tool]
    image_hint = "image_available" if _ACTIVE_IMAGE is not None else "no_image"
    if _ACTIVE_IMAGE is not None:
        available_tools.append(image_search_tool)

    planner_messages = planner_prompt.format_messages(
        question=question,
        image_hint=image_hint,
    )
    planner_llm = llm.bind_tools(available_tools)

    # 検索実行
    planner_response = planner_llm.invoke(planner_messages)

    if not isinstance(planner_response, AIMessage):
        raise RuntimeError("planner did not return AIMessage")

    logger.info(planner_response)

    tool_calls = getattr(planner_response, "tool_calls", None)
    used_tool_names, knowledge_chunks = _collect_tool_outputs(
        tool_calls or [], available_tools
    )
    logger.info(f"used tools: {', '.join(used_tool_names)}")

    planner_summary = (planner_response.content or "").strip()  # type: ignore

    knowledge = "\n".join(knowledge_chunks).strip()

    # 回答エージェントのセットアップ
    responder_prompt = _build_responder_prompt()
    messages = responder_prompt.format_messages(
        question=question,
        knowledge=knowledge or "",
        planner_summary=planner_summary,
        image_hint=image_hint,
    )

    # 回答生成
    final_response = llm.invoke(messages)
    logger.info(final_response)

    if not isinstance(final_response.content, str):
        raise ValueError("unexpected response shape")

    return final_response.content
