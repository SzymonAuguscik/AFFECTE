from typing import Any

import logging
import torch


class TensorManager:
    """
    TensorManager is used to save and load Tensors.

    Attributes
    ----------
    _logger : logging.Logger
        Used for logging purposes.

    Examples
    --------
    manager = TensorManager()
    X = manager.load("/path/to/features")
    y = manager.load("/path/to/labels")
    ...
    y = y.long()
    manager.save(y, "/new/path/to/labels")

    """
    def __init__(self) -> None:
        """Initiate TensorManager with default logger"""
        self._logger: logging.Logger = logging.getLogger(__name__)

    def load(self, path: str) -> Any:
        """
        Load Tensor.

        Parameters
        ----------
        path : str
            Path to load Tensor from.

        Returns
        -------
        Any
            Loaded Tensor.

        """
        self._logger.debug(f"Loading tensor from {path}")
        return torch.load(path)

    def save(self, tensor: Any, path: str) -> None:
        """
        Save Tensor.

        Parameters
        ----------
        tensor : Any
            Tensor to be saved.
        path : str
            Path to save Tensor.

        """
        self._logger.debug(f"Saving tensor to {path}")
        torch.save(tensor, path)

