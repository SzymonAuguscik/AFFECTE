from typing import Any

import logging
import torch


class TensorManager:
    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)

    def load(self, path: str) -> Any:
        self._logger.debug(f"Loading tensor from {path}")
        return torch.load(path)

    def save(self, tensor: Any, path: str) -> None:
        self._logger.debug(f"Saving tensor to {path}")
        torch.save(tensor, path)

