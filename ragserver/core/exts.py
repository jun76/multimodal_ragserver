from __future__ import annotations

from typing import ClassVar, FrozenSet


class Exts:
    # 基本的に reader(llama_index.core.readers.file.base._try_loading_included_file_formats)
    # のサポートする拡張子に追従する。
    # ただし、reader はその他の拡張子もフォールバックとしてテキストファイル扱いで
    # 読み込もうとするため、逆に .txt 等のテキストファイルの拡張子は明記されていない点に注意。

    # base64 エンコーディングしてマルチモーダル（画像）の埋め込みモデルに渡せる拡張子
    IMAGE: ClassVar[FrozenSet[str]] = frozenset(
        {".gif", ".jpg", ".png", ".jpeg", ".webp"}
    )

    # サイトマップの抽出判定に使用する拡張子
    SITEMAP: ClassVar[FrozenSet[str]] = frozenset({".xml"})

    # Web ページから予想外のファイルや巨大な動画ファイルをフェッチしてこないように絞る
    # 専用の reader が存在するもの
    _DEFAULT_FETCH_TARGET: ClassVar[FrozenSet[str]] = frozenset(
        {
            ".hwp",
            ".pdf",
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
    )

    # その他にフェッチしたいもの
    _ADDITIONAL_FETCH_TARGET: ClassVar[FrozenSet[str]] = frozenset(
        {
            ".txt",
            ".text",
            ".md",
            ".json",
        }
    )

    FETCH_TARGET: ClassVar[FrozenSet[str]] = (
        IMAGE | SITEMAP | _DEFAULT_FETCH_TARGET | _ADDITIONAL_FETCH_TARGET
    )

    @classmethod
    def endswith_exts(cls, s: str, exts: frozenset[str]) -> bool:
        """文字列の末尾に指定の拡張子が含まれるか。

        Args:
            s (str): 文字列
            exts (frozenset[str]): チェック対象の拡張子セット

        Returns:
            bool: 含まれる場合 True
        """
        return any(s.lower().endswith(ext) for ext in exts)

    @classmethod
    def is_image_file(cls, uri: str) -> bool:
        """ファイルパスまたは URL が指すのは画像ファイルか。

        Args:
            uri (str): ファイルパスまたは URL

        Returns:
            bool: 対象ファイルなら True
        """
        return cls.endswith_exts(uri, cls.IMAGE)

    @classmethod
    def is_sitemap_file(cls, url: str) -> bool:
        """URL が指すのはサイトマップファイルか。

        Args:
            url (str): URL

        Returns:
            bool: サイトマップファイルなら True
        """
        return cls.endswith_exts(url, cls.SITEMAP)
