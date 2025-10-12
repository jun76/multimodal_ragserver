from __future__ import annotations

# TODO: aiosqlite による非同期対応
import sqlite3
from typing import Any, Iterable, Optional, Sequence

from ragserver.core.metadata import META_KEYS as MK
from ragserver.core.metadata import BasicMetaData
from ragserver.core.names import PROJECT_NAME
from ragserver.logger import logger
from ragserver.stractured_store.structured_store_manager import StructuredStoreManager

# メタデータ管理テーブルの create 用
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
  {fingerprint}      TEXT    NOT NULL DEFAULT '',   -- fingerprint 文字列
  PRIMARY KEY ({file_path}, {url}, {chunk_no})
);
"""

# 同じソース塊（ファイル x URL x チャンク）を常に 1 行に保つ
# 内容が変われば同じ行を上書き（fingerprint も更新）
DDL_IDX_FINGERPRINT = "CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_fingerprint ON {table_name}({fingerprint});"

# システム起動時の fingerprint キャッシュロード時に効く
DDL_IDX_NODE_LASTMOD_AT = "CREATE INDEX IF NOT EXISTS idx_{table_name}_{node_lastmod_at} ON {table_name}({node_lastmod_at} DESC);"

# DELETE FROM table WHERE base_source = ?; 等に効く
DDL_IDX_BASE_SOURCE = "CREATE INDEX IF NOT EXISTS idx_{table_name}_{base_source} ON {table_name}({base_source});"

# メタデータ管理テーブルの upsert 用
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
  {fingerprint}
) VALUES (?,?,?,?,?,?,?,?,?,?)
ON CONFLICT({file_path},{url},{chunk_no}) DO UPDATE SET
  {file_type}       = excluded.{file_type},
  {file_size}       = excluded.{file_size},
  {file_lastmod_at} = excluded.{file_lastmod_at},
  {base_source}     = excluded.{base_source},
  {node_lastmod_at} = excluded.{node_lastmod_at},
  {fingerprint}     = excluded.{fingerprint}
"""


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
            self._db = sqlite3.connect(f"{PROJECT_NAME}.db")
        except Exception as e:
            raise RuntimeError("failed to initialize") from e

    def __del__(self):
        """デストラクタ"""
        logger.debug("trace")

        self._db.close()

    def _exec_query(self, query: str) -> list[tuple]:
        """クエリを実行する。

        Args:
            query (str): クエリ

        Raises:
            RuntimeError: クエリ実行失敗

        Returns:
            list[tuple]: 取得したレコード群
        """
        logger.debug("trace")
        logger.info(query)

        try:
            cur = self._db.cursor()
            cur.execute(query)
            self._db.commit()
            res = cur.fetchall()
            cur.close()
        except Exception as e:
            raise RuntimeError("failed to exec query") from e

        return res

    def _prepare_with(self, space_key: str) -> None:
        """空間キーに合わせてストアを初期化する。

        Args:
            space_key (str): 空間キー

        Raises:
            RuntimeError: ストア初期化失敗
        """
        logger.debug("trace")

        table_name = f"{PROJECT_NAME}_{self._knowledgebase_name}_{space_key}"
        try:
            self._exec_query(
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
                    fingerprint=MK.FINGERPRINT,
                )
            )
            self._exec_query(
                DDL_IDX_FINGERPRINT.format(
                    table_name=table_name, fingerprint=MK.FINGERPRINT
                )
            )
            self._exec_query(
                DDL_IDX_NODE_LASTMOD_AT.format(
                    table_name=table_name, node_lastmod_at=MK.NODE_LASTMOD_AT
                )
            )
            self._exec_query(
                DDL_IDX_BASE_SOURCE.format(
                    table_name=table_name,
                    base_source=MK.BASE_SOURCE,
                )
            )
        except Exception as e:
            raise RuntimeError("failed to exec DDL queries") from e

    def prepare_with(
        self, space_key_text: str, space_key_multi: Optional[str] = None
    ) -> None:
        """空間キーに合わせてストアを初期化する。

        Args:
            space_key_text (str): テキストベクトルの空間キー
            space_key_multi (Optional[str], optional): 画像ベクトルの空間キー。Defaults to None.

        Raises:
            RuntimeError: ストア初期化失敗
        """
        logger.debug("trace")

        self._space_key_text = space_key_text
        self._prepare_with(space_key_text)

        if space_key_multi:
            self._space_key_multi = space_key_multi
            self._prepare_with(space_key_multi)

    def _upsert_metadata_batch(
        self,
        table_name: str,
        rows: Iterable[Sequence[Any]],  # パラメタ順は下に記載
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
            fingerprint=MK.FINGERPRINT,
        )

        cur = self._db.cursor()
        try:
            self._db.execute("BEGIN")
            batch = []
            for row in rows:
                batch.append(row)
                if len(batch) >= chunk_size:
                    cur.executemany(sql, batch)
                    batch.clear()

            if batch:
                cur.executemany(sql, batch)

            self._db.commit()
        except Exception as e:
            self._db.rollback()
            cur.close()
            raise RuntimeError("failed to upsert batch") from e
        finally:
            cur.close()

    def _upsert(
        self, metas: list[BasicMetaData], fingerprints: list[str], space_key: str
    ) -> None:
        """メタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
            space_key (str): 空間キー
        """
        logger.debug("trace")

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
                fingerprints[i],
            )
            rows.append(row)

        table_name = f"{PROJECT_NAME}_{self._knowledgebase_name}_{space_key}"
        self._upsert_metadata_batch(table_name=table_name, rows=rows)

    def upsert_text_metas(
        self, metas: list[BasicMetaData], fingerprints: list[str]
    ) -> None:
        """テキストノードのメタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
        """
        logger.debug("trace")

        if self._space_key_text is None:
            logger.warning("space key(text) is not initialized")
            return

        self._upsert(
            metas=metas, fingerprints=fingerprints, space_key=self._space_key_text
        )

    def upsert_image_metas(
        self, metas: list[BasicMetaData], fingerprints: list[str]
    ) -> None:
        """画像ノードのメタデータをストアに格納する。

        Args:
            metas (list[BasicMetaData]): メタデータ
            fingerprints (list[str]): fingerprint 文字列
        """
        logger.debug("trace")

        if self._space_key_multi is None:
            logger.warning("space key(multi) is not initialized")
            return

        self._upsert(
            metas=metas, fingerprints=fingerprints, space_key=self._space_key_multi
        )
