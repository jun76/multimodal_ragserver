from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Sequence

import aiosqlite

from ragserver.config.general_config import GeneralConfig
from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import BasicMetaData
from ragserver.logger import logger
from ragserver.meta_store.structured.structured import Structured

# メタデータ管理テーブルの create 用
# カラムを追加する場合、PRIMARY KEY への追加が必要なら足し忘れに注意
# PRIMARY KEY に追加する場合、fingerprint 計算側（_get_lazy_fp）の修正も忘れずに
DDL_CREATE_METADATA = """
CREATE TABLE IF NOT EXISTS {table_name} (
  {file_path}        TEXT    NOT NULL DEFAULT '',   -- 取得元ファイルパス
  {file_type}        TEXT    NOT NULL DEFAULT '',   -- mimetype 等
  {file_size}        INTEGER NOT NULL DEFAULT 0,    -- バイト
  {file_created_at}  TEXT    NOT NULL DEFAULT '',   -- ファイル作成日時（ISO文字列等）
  {file_lastmod_at}  TEXT    NOT NULL DEFAULT '',   -- ファイル最終更新日時（ISO文字列等）
  {chunk_no}         INTEGER NOT NULL DEFAULT 0,    -- テキストのチャンク番号
  {url}              TEXT    NOT NULL DEFAULT '',   -- 取得元 URL（無ければ空）
  {base_source}      TEXT    NOT NULL DEFAULT '',   -- 出典（直リンク画像の親ページ等）
  {node_lastmod_at}  REAL    NOT NULL DEFAULT 0,    -- ノードの最終更新時刻（epoch 秒）
  {page_no}          INTEGER NOT NULL DEFAULT 0,    -- ページ番号
  {asset_no}         INTEGER NOT NULL DEFAULT 0,    -- アセット番号（同一ページ内の画像等）
  {fingerprint}      TEXT    NOT NULL DEFAULT '',   -- fingerprint 文字列
  PRIMARY KEY ({file_path}, {url}, {chunk_no}, {page_no}, {asset_no})
);
"""

# 同じソース塊（ファイル x URL x チャンク）を常に 1 行に保つ
# 内容が変われば同じ行を上書き（fingerprint も更新）
DDL_IDX_FINGERPRINT = """
CREATE UNIQUE INDEX IF NOT EXISTS
  idx_{table_name}_fingerprint ON {table_name}({fingerprint});
"""

# システム起動時の fingerprint キャッシュロード時に効く
DDL_IDX_NODE_LASTMOD_AT = """
CREATE INDEX IF NOT EXISTS 
  idx_{table_name}_{node_lastmod_at} ON {table_name}({node_lastmod_at} DESC);
"""

# DELETE FROM table WHERE base_source = ?; 等に効く
DDL_IDX_BASE_SOURCE = """
CREATE INDEX IF NOT EXISTS
  idx_{table_name}_{base_source} ON {table_name}({base_source});
"""

# メタデータ管理テーブルの upsert 用
# カラムを追加する場合、?, の足し忘れに注意
DML_UPSERT_METADATA = """
INSERT INTO {table_name} (
  {file_path},
  {file_type},
  {file_size},
  {file_created_at},
  {file_lastmod_at},
  {chunk_no},
  {url},
  {base_source},
  {node_lastmod_at},
  {page_no},
  {asset_no},
  {fingerprint}
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT({file_path},{url},{chunk_no},{page_no},{asset_no}) DO UPDATE SET
  {file_type}       = excluded.{file_type},
  {file_size}       = excluded.{file_size},
  {file_lastmod_at} = excluded.{file_lastmod_at},
  {base_source}     = excluded.{base_source},
  {node_lastmod_at} = excluded.{node_lastmod_at},
  {fingerprint}     = excluded.{fingerprint}
"""

# select 用
DML_SELECT = "SELECT {col_csv}, {order_col} AS _ord FROM {table}"
DML_SELECT_MULTI = (
    "SELECT {col_csv} FROM ({union_all}) AS _u ORDER BY _ord DESC LIMIT {limit}"
)
UNION_ALL = " UNION ALL "


class SQLiteStructured(Structured):
    def __init__(self) -> None:
        """SQLite3 管理クラス

        Raises:
            RuntimeError: 初期化失敗
        """
        logger.debug("trace")

        self._db_path = f"{GeneralConfig.project_name}_metas.db"

        try:
            self._sync_db = sqlite3.connect(self._db_path)
        except Exception as e:
            raise RuntimeError("failed to initialize") from e

        self._created: list[str] = []

    def _prepare_with(self, table_name: str) -> None:
        """指定のテーブルが存在しない場合、予め作成する。

        Args:
            table_name (str): テーブル名

        Raises:
            RuntimeError: テーブル作成失敗
        """
        logger.debug("trace")

        try:
            self._sync_db.execute("BEGIN")
            self._sync_db.execute(
                DDL_CREATE_METADATA.format(
                    table_name=table_name,
                    file_path=MK.FILE_PATH,
                    file_type=MK.FILE_TYPE,
                    file_size=MK.FILE_SIZE,
                    file_created_at=MK.FILE_CREATED_AT,
                    file_lastmod_at=MK.FILE_LASTMOD_AT,
                    chunk_no=MK.CHUNK_NO,
                    url=MK.URL,
                    base_source=MK.BASE_SOURCE,
                    node_lastmod_at=MK.NODE_LASTMOD_AT,
                    page_no=MK.PAGE_NO,
                    asset_no=MK.ASSET_NO,
                    fingerprint=MK.FINGERPRINT,
                )
            )
            self._sync_db.execute(
                DDL_IDX_FINGERPRINT.format(
                    table_name=table_name, fingerprint=MK.FINGERPRINT
                )
            )
            self._sync_db.execute(
                DDL_IDX_NODE_LASTMOD_AT.format(
                    table_name=table_name, node_lastmod_at=MK.NODE_LASTMOD_AT
                )
            )
            self._sync_db.execute(
                DDL_IDX_BASE_SOURCE.format(
                    table_name=table_name,
                    base_source=MK.BASE_SOURCE,
                )
            )
            self._sync_db.commit()
        except Exception as e:
            self._sync_db.rollback()
            raise RuntimeError("failed to exec DDL queries") from e

        self._created.append(table_name)

    async def _aupsert_batch(
        self,
        table_name: str,
        rows: Iterable[Sequence[Any]],
        chunk_size: int = 1000,
    ) -> None:
        """メタデータのバッチ upsert。

        Args:
            table_name (str): テーブル名
            rows (Iterable[Sequence[Any]]): メタデータ（複数レコード）
            chunk_size (int): バッチ数が多すぎる場合の分割用

        Raises:
            RuntimeError: upsert 失敗
        """
        logger.debug("trace")

        sql = DML_UPSERT_METADATA.format(
            table_name=table_name,
            file_path=MK.FILE_PATH,
            file_type=MK.FILE_TYPE,
            file_size=MK.FILE_SIZE,
            file_created_at=MK.FILE_CREATED_AT,
            file_lastmod_at=MK.FILE_LASTMOD_AT,
            chunk_no=MK.CHUNK_NO,
            url=MK.URL,
            base_source=MK.BASE_SOURCE,
            node_lastmod_at=MK.NODE_LASTMOD_AT,
            page_no=MK.PAGE_NO,
            asset_no=MK.ASSET_NO,
            fingerprint=MK.FINGERPRINT,
        )

        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute("BEGIN")
                batch: list[Sequence[Any]] = []
                for row in rows:
                    batch.append(row)
                    if len(batch) >= chunk_size:
                        await db.executemany(sql, batch)
                        batch.clear()

                if batch:
                    await db.executemany(sql, batch)

                await db.commit()
            except Exception as e:
                await db.rollback()
                raise RuntimeError("failed to upsert batch") from e

    async def aupsert(
        self, metas: list[BasicMetaData], fingerprints: list[str], table_name: str
    ) -> None:
        """メタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
            table_name (str): テーブル名

        Raises:
            RuntimeError: upsert 失敗
        """
        logger.debug("trace")

        if table_name not in self._created:
            self._prepare_with(table_name)

        rows = []
        for i, meta in enumerate(metas):
            row = (
                meta.file_path,
                meta.file_type,
                meta.file_size,
                meta.file_created_at,
                meta.file_lastmod_at,
                meta.chunk_no,
                meta.url,
                meta.base_source,
                meta.node_lastmod_at,
                meta.page_no,
                meta.asset_no,
                fingerprints[i],
            )
            rows.append(row)

        await self._aupsert_batch(table_name=table_name, rows=rows)

    def select(
        self, cols: list[str], table_names: list[str], limit: int
    ) -> list[tuple]:
        """select 文を実行する。

        Args:
            cols (list[str]): 取得する列
            table_names (list[str]): テーブル名のリスト
            limit (int): 件数上限

        Returns:
            list[tuple]: 取得したレコード群
        """
        logger.debug("trace")

        for table_name in table_names:
            if table_name not in self._created:
                self._prepare_with(table_name)

        col_csv = ", ".join(cols)
        parts = [
            DML_SELECT.format(
                col_csv=col_csv, order_col=MK.NODE_LASTMOD_AT, table=table_name
            )
            for table_name in table_names
        ]
        query = DML_SELECT_MULTI.format(
            col_csv=col_csv, union_all=UNION_ALL.join(parts), limit=limit
        )

        try:
            with self._sync_db:
                cur = self._sync_db.cursor()
                try:
                    cur.execute(query)
                    res = cur.fetchall()
                finally:
                    cur.close()
        except Exception as e:
            raise RuntimeError("failed to exec query") from e

        return res
