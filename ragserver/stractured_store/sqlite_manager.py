from __future__ import annotations

import sqlite3

from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.names import PROJECT_NAME
from ragserver.logger import logger
from ragserver.stractured_store.stractured_store_manager import StructuredStoreManager

# DDL（{table_name} を format で埋める）
DDL_METADATA = """
CREATE TABLE IF NOT EXISTS {table_name} (
  {file_path}        TEXT    NOT NULL DEFAULT '',   -- 取得元ファイルパス
  {file_type}        TEXT    NOT NULL DEFAULT '',   -- mimetype 等
  {file_size}        INTEGER NOT NULL DEFAULT 0,    -- バイト
  {file_created_at}  TEXT    NOT NULL DEFAULT '',   -- ファイル作成日時（ISO文字列等）
  {file_lastmod_at}  TEXT    NOT NULL DEFAULT '',   -- ファイル最終更新日時（ISO文字列等）
  {chunk_no}         INTEGER NOT NULL DEFAULT 0,    -- テキストのチャンク番号
  {url}              TEXT    NOT NULL DEFAULT '',   -- 取得元 URL（無ければ空）
  {base_source}      TEXT    NOT NULL DEFAULT '',   -- 出典（直リンク画像の親ページ等）
  {node_lastmod_at}  REAL    NOT NULL DEFAULT 0,    -- ノードの最終更新（epoch秒 推奨）
  PRIMARY KEY ({file_path}, {url}, {chunk_no})
);
"""

# システム起動時の fingerprint キャッシュロード時に効く
DDL_IDX_NODE_LASTMOD_AT = "CREATE INDEX IF NOT EXISTS idx_{table}_node_lastmod_at ON {table}({node_lastmod_at} DESC);"

# DELETE FROM table WHERE base_source = ?; 等に効く
DDL_IDX_BASE_SOURCE = (
    "CREATE INDEX IF NOT EXISTS idx_{table}_base_source ON {table}({base_source});"
)


class SQLiteManager(StructuredStoreManager):
    def __init__(
        self,
        knowledgebase_name: str = "default",
    ) -> None:
        """SQLite3 管理クラス

        Args:
            knowledgebase_name (str, optional): ナレッジベース（用途）名。Defaults to "default".

        Raises:
            RuntimeError: 初期化失敗
        """
        logger.debug("trace")

        super().__init__(knowledgebase_name)

        try:
            self._db = sqlite3.connect(f"{PROJECT_NAME}")
        except Exception as e:
            raise RuntimeError("failed to initialize") from e

    def __del__(self):
        """デストラクタ"""
        logger.debug("trace")

        self._db.close()

    def _exec_query(self, query: str, params: list[str]) -> dict[str, str]:
        """クエリを実行する。

        Args:
            query (str): クエリ
            params (list[str]): パラメータのリスト

        Returns:
            dict[str, str]: 取得したレコード群
        """

        logger.debug("trace")

        try:
            cur = self._db.cursor()
            cur.execute(query, params)
            self._db.commit()
            res = dict(cur.fetchall())
            cur.close()
        except Exception as e:
            raise RuntimeError("failed to exec query") from e

        return res

    def activate_with(self, space_key: str):
        """空間キーに合わせてストアを初期化する。

        Args:
            space_key (str): 空間キー

        Raises:
            RuntimeError: ストア初期化失敗
        """
        logger.debug("trace")

        table_name = f"{PROJECT_NAME}_{self._knowledgebase_name}_{space_key}"
        query = DDL_METADATA.format(
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
        )

        try:
            self._exec_query(query, [])
            self._exec_query(
                DDL_IDX_NODE_LASTMOD_AT.format(
                    table=table_name, node_lastmod_at=MK.NODE_LASTMOD_AT
                ),
                [],
            )
            self._exec_query(
                DDL_IDX_BASE_SOURCE.format(
                    table=table_name,
                    base_source=MK.BASE_SOURCE,
                ),
                [],
            )
        except Exception as e:
            raise RuntimeError("failed to exec DDL query") from e
