from enum import Enum

from src.enums.ContainerizedEnum import ContainerizedEnum


class AugmentationStrategy(Enum, metaclass=ContainerizedEnum):
    ADD = "ADD"
    NOISE = "NOISE"
