from constants import Hyperparameters

import logging
import torch


class FeedForwardClassifier(torch.nn.Module):
    def __init__(self, in_features: int) -> None:
        super(FeedForwardClassifier, self).__init__()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._linear1: torch.nn.Linear = torch.nn.Linear(in_features=in_features, out_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION)
        self._drop1: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.FeedForwardClassifier.Layer1.DROPOUT)
        self._relu: torch.nn.ReLU = torch.nn.ReLU()
        self._linear2: torch.nn.Linear = torch.nn.Linear(in_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION, out_features=1)
        self._drop2: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.FeedForwardClassifier.Layer2.DROPOUT)
        self._sigmoid: torch.nn.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._logger.debug("Starting forward pass!")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._relu(self._drop1(self._linear1(x)))
        self._logger.debug("relu(drop(linear(x))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._sigmoid(self._drop2(self._linear2(x)))
        self._logger.debug("sigmoid(drop(linear(x))) done")
        self._logger.debug(f"Data size: {x.size()}")
        return x.mean(1).reshape(-1, 1)

