from src.enums.ContainerizedEnum import ContainerizedEnum
from enum import Enum

class AugmentationStrategy(Enum, metaclass=ContainerizedEnum):
    ADD = "ADD"
    NOISE = "NOISE"

