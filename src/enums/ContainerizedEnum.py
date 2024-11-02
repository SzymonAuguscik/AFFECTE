from typing import Any
from enum import EnumMeta


class ContainerizedEnum(EnumMeta):
    def __contains__(cls: Any, item: object) -> bool:
        return item in cls.__members__

