from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from .api_client import RagServerClient
from .logger import logger

__all__ = ["AgentExecutionError", "RagAgentManager"]


class AgentExecutionError(RuntimeError):
    """openai-agents 実行時の例外ラッパー"""


class _TextSearchArgs(TypedDict, total=False):
    query: str
    topk: int


class _MultiModalSearchArgs(TypedDict, total=False):
    topk: int


class _RagAgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: RagServerClient
    file_path: Optional[str] = None


def _format_documents(payload: dict[str, Any]) -> str:
    """検索結果ドキュメントを短い文字列へ整形する。

    Args:
        payload (dict[str, Any]): 検索 API の応答ペイロード

    Returns:
        str: 各ドキュメントの概要をまとめた文字列
    """

    docs = payload.get("documents") or []
    if not docs:
        return "No documents were retrieved."

    lines: list[str] = []
    for idx, doc in enumerate(docs[:5], start=1):
        metadata = doc.get("metadata") or {}
        source = metadata.get("file_path") or metadata.get("url") or "unknown source"
        text = (doc.get("text") or "").strip().replace("\n", " ")
        score = doc.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
        lines.append(f"{idx}. score={score_text} source={source}\n{text[:200]}")

    return "\n".join(lines)


def _format_response(title: str, payload: dict[str, Any]) -> str:
    """検索結果を JSON 文字列としてまとめる。

    Args:
        title (str): 結果種別を示すタイトル
        payload (dict[str, Any]): 検索 API の応答ペイロード

    Returns:
        str: まとめられた検索結果 JSON 文字列
    """

    summary = _format_documents(payload)
    result = json.dumps(
        {"title": title, "summary": summary, "raw": payload},
        ensure_ascii=False,
        indent=2,
    )
    logger.debug(result)

    return result


@function_tool
async def tool_search_text_text(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _TextSearchArgs,
) -> str:
    """テキストクエリでテキストドキュメントを検索する。

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): 実行時コンテキスト
        args (_TextSearchArgs): 検索パラメータ

    Raises:
        ValueError: クエリ文字列が指定されていない場合

    Returns:
        str: 検索結果要約を含む JSON 文字列
    """

    query = args.get("query")
    if not query:
        raise ValueError("query is required")

    topk = args.get("topk")
    response = ctx.context.client.query_text_text(query, topk)
    return _format_response("text_text", response)


@function_tool
async def tool_search_text_image(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _TextSearchArgs,
) -> str:
    """テキストクエリで画像ドキュメントを検索する。

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): 実行時コンテキスト
        args (_TextSearchArgs): 検索パラメータ

    Raises:
        ValueError: クエリ文字列が指定されていない場合

    Returns:
        str: 検索結果要約を含む JSON 文字列
    """

    query = args.get("query")
    if not query:
        raise ValueError("query is required")

    topk = args.get("topk")
    response = ctx.context.client.query_text_image(query, topk)
    return _format_response("text_image", response)


@function_tool
async def tool_search_image_image(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _MultiModalSearchArgs,
) -> str:
    """アップロード済み画像を基に画像ドキュメントを検索する。

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): 実行時コンテキスト
        args (_MultiModalSearchArgs): 検索パラメータ

    Raises:
        ValueError: 参照画像が未登録の場合

    Returns:
        str: 検索結果要約を含む JSON 文字列
    """

    if not ctx.context.file_path:
        raise ValueError("file_path is not provided in context")

    topk = args.get("topk")
    response = ctx.context.client.query_image_image(ctx.context.file_path, topk)
    return _format_response("image_image", response)


@function_tool
async def tool_search_text_audio(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _TextSearchArgs,
) -> str:
    """テキストクエリで音声ドキュメントを検索する。

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): 実行時コンテキスト
        args (_TextSearchArgs): 検索パラメータ

    Raises:
        ValueError: クエリ文字列が指定されていない場合

    Returns:
        str: 検索結果要約を含む JSON 文字列
    """

    query = args.get("query")
    if not query:
        raise ValueError("query is required")

    topk = args.get("topk")
    response = ctx.context.client.query_text_audio(query, topk)
    return _format_response("text_audio", response)


@function_tool
async def tool_search_audio_audio(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _MultiModalSearchArgs,
) -> str:
    """アップロード済み音声を基に音声ドキュメントを検索する。

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): 実行時コンテキスト
        args (_MultiModalSearchArgs): 検索パラメータ

    Raises:
        ValueError: 参照音声が未登録の場合

    Returns:
        str: 検索結果要約を含む JSON 文字列
    """

    if not ctx.context.file_path:
        raise ValueError("file_path is not provided in context")

    topk = args.get("topk")
    response = ctx.context.client.query_audio_audio(ctx.context.file_path, topk)
    return _format_response("audio_audio", response)


_TOOLSET = [
    tool_search_text_text,
    tool_search_text_image,
    tool_search_image_image,
    tool_search_text_audio,
    tool_search_audio_audio,
]


@dataclass
class RagAgentManager:
    """openai-agents を用いた RAG 検索の実行を管理するクラス。"""

    client: RagServerClient
    model: str

    def run(
        self,
        *,
        question: str,
        file_path: Optional[str] = None,
        max_turns: int = 5,
    ) -> str:
        """エージェントを実行し最終回答を返す。

        Args:
            question (str): ユーザからの質問文
            file_path (Optional[str]): 参照画像ファイルの保存パス
            max_turns (int): エージェントの最大ターン数

        Raises:
            ValueError: 質問文が空の場合
            AgentExecutionError: エージェント実行に失敗した場合

        Returns:
            str: エージェントが生成した最終回答
        """

        if question.strip() == "":
            raise ValueError("question must not be empty")

        logger.debug([tool.name for tool in _TOOLSET])
        agent = Agent(
            name="rag_assistant",
            instructions=(
                "あなたは検索エージェントです。"
                "ユーザからの質問に対し、日本語で回答して下さい。"
                "回答する前に、提供されているツールを使用して必ずナレッジベースを検索して下さい。"
                "検索の際に使用できる参考画像や音声がある場合は file_path に格納されています。"
                "関連文書が見つかった場合は、ファイルパスを回答に含めて下さい。"
                "ただし、スコアは回答に含めないで下さい。"
                "その他、関連文書が見つからない場合やエラー時は"
                "「該当するドキュメントが見つかりませんでした。」とだけ回答下さい。"
            ),
            tools=_TOOLSET,  # type: ignore
            model=self.model,
        )

        logger.debug(f"file path = {file_path}")
        context = _RagAgentContext(
            client=self.client,
            file_path=file_path,
        )

        async def _run() -> str:
            result = await Runner.run(
                agent,
                input=question,
                max_turns=max_turns,
                context=context,
            )

            final = getattr(result, "final_output", None)
            if isinstance(final, str):
                return final
            if final is not None:
                return str(final)
            return ""

        try:
            return asyncio.run(_run())
        except Exception as e:
            logger.exception(e)
            raise AgentExecutionError(str(e)) from e
