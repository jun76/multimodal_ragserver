from __future__ import annotations

import os
from urllib.parse import urlparse


class Exts:
    # 個別参照用
    PNG: str = ".png"
    PDF: str = ".pdf"

    # 基本的に reader(llama_index.core.readers.file.base._try_loading_included_file_formats)
    # のサポートする拡張子に追従する。
    # ただし、reader はその他の拡張子もフォールバックとしてテキストファイル扱いで
    # 読み込もうとするため、逆に .txt 等のテキストファイルの拡張子は明記されていない点に注意。

    # base64 エンコーディングしてマルチモーダル（画像）の埋め込みモデルに渡せる拡張子
    IMAGE: set[str] = {".gif", ".jpg", PNG, ".jpeg", ".webp"}

    # マルチモーダル（音声）の埋め込みモデルに渡せる拡張子
    AUDIO: set[str] = {".wav", ".mp3", ".flac", ".ogg"}

    # サイトマップの抽出判定に使用する拡張子
    SITEMAP: set[str] = {".xml"}

    ## Web ページから予想外のファイルや巨大な動画ファイルをフェッチしてこないように絞る
    # 専用の reader が存在するもの
    _DEFAULT_FETCH_TARGET: set[str] = {
        ".hwp",
        PDF,
        ".docx",
        ".pptx",
        ".ppt",
        ".pptm",
        ".csv",
        ".epub",
        ".mbox",
        ".ipynb",
        ".xls",
        ".xlsx",
    }

    # その他にフェッチしたいもの
    _ADDITIONAL_FETCH_TARGET: set[str] = {
        ".txt",
        ".text",
        ".md",
        ".json",
    }

    FETCH_TARGET: set[str] = (
        IMAGE | AUDIO | SITEMAP | _DEFAULT_FETCH_TARGET | _ADDITIONAL_FETCH_TARGET
    )

    @classmethod
    def endswith_exts(cls, s: str, exts: set[str]) -> bool:
        """文字列の末尾に指定の拡張子が含まれるか。

        Args:
            s (str): 文字列
            exts (set[str]): チェック対象の拡張子セット

        Returns:
            bool: 含まれる場合 True
        """
        return any(s.lower().endswith(ext) for ext in exts)

    @classmethod
    def endswith_ext(cls, s: str, ext: str) -> bool:
        """文字列の末尾に指定の拡張子が含まれるか。

        Args:
            s (str): 文字列
            ext (str): チェック対象の拡張子

        Returns:
            bool: 含まれる場合 True
        """
        return s.lower().endswith(ext.lower())

    @classmethod
    def get_ext(cls, uri: str) -> str:
        """ファイルパスまたは URL 文字列から拡張子を取得する。

        Args:
            uri (str): ファイルパスまたは URL

        Returns:
            str: 拡張子
        """
        parsed = urlparse(uri)

        return os.path.splitext(parsed.path)[1].lower()
