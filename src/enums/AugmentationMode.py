from src.enums.ContainerizedEnum import ContainerizedEnum
from enum import Enum

class AugmentationMode(Enum, metaclass=ContainerizedEnum):
    APPEND = "APPEND"
    MODIFY = "MODIFY"

