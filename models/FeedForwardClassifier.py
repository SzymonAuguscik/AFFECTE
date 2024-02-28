from constants import Hyperparameters

import logging
import torch


class FeedForwardClassifier(torch.nn.Module):
    def __init__(self, in_features: int) -> None:
        super(FeedForwardClassifier, self).__init__()
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.linear1: torch.nn.Linear = torch.nn.Linear(in_features=in_features, out_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION)
        self.drop1: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.FeedForwardClassifier.Layer1.DROPOUT)
        self.relu: torch.nn.ReLU = torch.nn.ReLU()
        self.linear2: torch.nn.Linear = torch.nn.Linear(in_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION, out_features=1)
        self.drop2: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.FeedForwardClassifier.Layer2.DROPOUT)
        self.sigmoid: torch.nn.Sigmoid = torch.nn.Sigmoid()

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

