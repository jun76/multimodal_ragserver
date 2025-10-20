from __future__ import annotations

from enum import StrEnum, auto

from llama_index.core.schema import TextNode


# モダリティ
# ! 字列を変更すると空間キーの字列が変わって別空間（ingest やり直し）になるので注意 !
class Modality(StrEnum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()


class AudioNode(TextNode):
    """音声モダリティのノード実装用クラス"""

    pass
