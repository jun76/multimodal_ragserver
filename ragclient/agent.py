from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from .api_client import RagServerClient
from .config.config import Config
from .config.settings import LLMProvider
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
    image_path: Optional[str] = None
    audio_path: Optional[str] = None


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
    return json.dumps(
        {"title": title, "summary": summary, "raw": payload}, ensure_ascii=False
    )


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

    if not ctx.context.image_path:
        raise ValueError("image_path is not provided in context")

    topk = args.get("topk")
    response = ctx.context.client.query_image_image(ctx.context.image_path, topk)
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

    if not ctx.context.audio_path:
        raise ValueError("audio_path is not provided in context")

    topk = args.get("topk")
    response = ctx.context.client.query_audio_audio(ctx.context.audio_path, topk)
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
    provider: LLMProvider

    def run(
        self,
        *,
        question: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_turns: int = 8,
    ) -> str:
        """エージェントを実行し最終回答を返す。

        Args:
            question (str): ユーザからの質問文
            image_path (Optional[str]): 参照画像ファイルの保存パス
            audio_path (Optional[str]): 参照音声ファイルの保存パス
            max_turns (int): エージェントの最大ターン数

        Raises:
            ValueError: 質問文が空の場合
            AgentExecutionError: エージェント実行に失敗した場合

        Returns:
            str: エージェントが生成した最終回答
        """

        if question.strip() == "":
            raise ValueError("question must not be empty")

        agent = Agent(
            name="rag_assistant",
            instructions=(
                "あなたは検索アシスタントです。"
                "回答する前に、ツールを使用して必ずナレッジベースを調べてください。"
                "ツールが関連文書を返さない場合は、裏付けとなる証拠が見つからないことを明記してください。"
                "日本語で回答してください。"
                "最終的な回答を出す前に、提供されているツールを少なくとも１つ呼び出して下さい。"
            ),
            tools=_TOOLSET,  # type: ignore
            model=self._resolve_model(),
        )

        context = _RagAgentContext(
            client=self.client,
            image_path=image_path,
            audio_path=audio_path,
        )
        logger.info(f"image path = {image_path}, audio path = {audio_path}")

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

    def _resolve_model(self) -> str:
        """LLM プロバイダ設定から使用するモデル名を取得する。

        Raises:
            AgentExecutionError: 未対応のプロバイダが指定された場合

        Returns:
            str: 利用するモデル名
        """

        if self.provider is LLMProvider.OPENAI:
            return Config.llm_openai_model
        if self.provider is LLMProvider.LOCAL:
            return Config.llm_local_model
        raise AgentExecutionError(f"unsupported llm provider: {self.provider}")
