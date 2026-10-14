from enum import Enum

from src.enums.ContainerizedEnum import ContainerizedEnum


class AugmentationMode(Enum, metaclass=ContainerizedEnum):
    APPEND = "APPEND"
    MODIFY = "MODIFY"
