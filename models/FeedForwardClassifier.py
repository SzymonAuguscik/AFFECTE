from torch.nn import Module, Linear, Sigmoid, ReLU, Dropout
from constants import Hyperparameters

import logging
import torch


class FeedForwardClassifier(Module):
    def __init__(self, in_features: int) -> None:
        super(FeedForwardClassifier, self).__init__()
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.linear1: Linear = Linear(in_features=in_features, out_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION)
        self.drop1: Dropout = Dropout(Hyperparameters.FeedForwardClassifier.Layer1.DROPOUT)
        self.relu: ReLU = ReLU()
        self.linear2: Linear = Linear(in_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION, out_features=1)
        self.drop2: Dropout = Dropout(Hyperparameters.FeedForwardClassifier.Layer2.DROPOUT)
        self.sigmoid: Sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.debug("Starting forward pass!")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.relu(self.drop1(self.linear1(x)))
        self.logger.debug("relu(drop(linear(x))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.sigmoid(self.drop2(self.linear2(x)))
        self.logger.debug("sigmoid(drop(linear(x))) done")
        self.logger.debug(f"Data size: {x.size()}")
        return x.mean(1).reshape(-1, 1)

