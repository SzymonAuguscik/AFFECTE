from enum import EnumMeta
from typing import Any


class ContainerizedEnum(EnumMeta):
    def __contains__(cls: Any, item: object) -> bool:
        return item in cls.__members__
