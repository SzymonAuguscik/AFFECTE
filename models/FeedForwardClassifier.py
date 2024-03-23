from constants import Hyperparameters

import logging
import torch


class FeedForwardClassifier(torch.nn.Module):
    """
    FeedForwardClassifier is a 2-layer neural network with sigmoid head.
    Besides it contains ReLU as 1st activation and 2 dropouts after each layer.

    Attributes
    ----------
    _logger : logging.Logger
        Used for logging purposes.
    _linear1 : torch.nn.Linear
        First linear layer. Features: in_layers -> 1024.
    _drop1 : torch.nn.Dropout
        First dropout layer. Drops neurons with probability 0.03.
    _relu : torch.nn.ReLU
        First activation layer.
    _linear2 : torch.nn.Linear
        First linear layer. Features: 1024 -> 1.
    _drop2 : torch.nn.Dropout
        Second dropout layer. Drops neurons with probability 0.03.
    _sigmoid : torch.nn.Sigmoid
        Second activation layer. The output of the model.

    Examples
    --------
    X = <load features e.g. from EcgSignalLoader>
    model = FeedForwardClassifier(in_features=200)
    predictions = model(X)

    """
    def __init__(self, in_features: int) -> None:
        """
        Initiate FeedForwardClassifier with default logger and trainable layers.

        Parameters
        ----------
        in_features : int
            The number of the input features to the model.

        """
        super(FeedForwardClassifier, self).__init__()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._linear1: torch.nn.Linear = torch.nn.Linear(in_features=in_features, out_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION)
        self._drop1: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.FeedForwardClassifier.Layer1.DROPOUT)
        self._relu: torch.nn.ReLU = torch.nn.ReLU()
        self._linear2: torch.nn.Linear = torch.nn.Linear(in_features=Hyperparameters.FeedForwardClassifier.HIDDEN_LAYER_DIMENSION, out_features=1)
        self._drop2: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.FeedForwardClassifier.Layer2.DROPOUT)
        self._sigmoid: torch.nn.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The batch of the data to be propagated.

        Returns
        -------
        torch.Tensor
            The average values of rows along 1st dimension from propagated tensor.

        """
        self._logger.debug("Starting forward pass!")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._relu(self._drop1(self._linear1(x)))
        self._logger.debug("relu(drop(linear(x))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._sigmoid(self._drop2(self._linear2(x)))
        self._logger.debug("sigmoid(drop(linear(x))) done")
        self._logger.debug(f"Data size: {x.size()}")
        return x.mean(1).reshape(-1, 1)

