from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ragclient.api_client import RagServerClient
from ragclient.config import Config, get_config
from ragclient.logger import logger
from ragclient.state import (
    VIEW_ADMIN,
    VIEW_INGEST,
    VIEW_MAIN,
    VIEW_RAGSEARCH,
    VIEW_SEARCH,
    ensure_session_state,
)
from ragclient.views.admin import render_admin_view
from ragclient.views.ingest import render_ingest_view
from ragclient.views.main_menu import render_main_menu
from ragclient.views.ragsearch import render_ragsearch_view
from ragclient.views.search import render_search_view


def _init_services() -> tuple[RagServerClient, str, str, str]:
    """設定を読み込み、API クライアントとヘルスチェック用 URL を初期化する。

    Returns:
        tuple[RagServerClient, str, str, str]: 設定オブジェクトと API クライアント、
        ragserver・埋め込み・リランク各サービスのヘルスチェック URL
    """
    logger.debug("trace")

    cfg = get_config()
    client = RagServerClient(cfg.ragserver_base_url)
    ragserver_health = cfg.ragserver_base_url.rstrip("/") + "/health"
    embed_health = cfg.hfclip_embed_base_url.rstrip("/") + "/health"
    rerank_health = cfg.hf_rerank_base_url.rstrip("/") + "/health"
    return client, ragserver_health, embed_health, rerank_health


def main() -> None:
    """Streamlit アプリのエントリポイント。"""
    logger.debug("trace")

    st.set_page_config(page_title="RAG Client", page_icon="🧠", layout="wide")
    ensure_session_state()

    client, ragserver_health, embed_health, rerank_health = _init_services()

    view = st.session_state.get("view", VIEW_MAIN)
    if view == VIEW_MAIN:
        render_main_menu(ragserver_health, embed_health, rerank_health)
    elif view == VIEW_INGEST:
        render_ingest_view(client)
    elif view == VIEW_SEARCH:
        render_search_view(client)
    elif view == VIEW_RAGSEARCH:
        render_ragsearch_view(client)
    elif view == VIEW_ADMIN:
        render_admin_view(client)
    else:
        st.error("未定義の画面です")


if __name__ == "__main__":
    main()
