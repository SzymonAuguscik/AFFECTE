from constants import Hyperparameters

import logging
import torch


class Transformer(torch.nn.Module):
    """
    Transformer is a transformer encoder which consists of encoder layers only.
    Each encoder layer is the same (e.g. has fixed layer normalization coefficient and activation for FFNs).

    Attributes
    ----------
    _logger : logging.Logger
        Used for logging purposes.
    _d_model : int
        The expected number of features. In other words, it is a number of all possible feature values
        after embedding.
    _encoder : torch.nn.TransformerEncoder
        The stack of encoding layers.

    Examples
    --------
    X = <load features e.g. from EcgSignalLoader>
    model = Transformer(d_model=64, hidden_dimension=128, heads_number=16, encoder_layers=4)
    predictions = model(X)

    See also
    --------
    https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
    https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

    """
    def __init__(self, d_model: int, hidden_dimension: int, heads_number: int, encoder_layers: int) -> None:
        """
        Build a Transformer encoder from individual layers and initialize weights with Xavier method.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        hidden_dimension : int
            The dimension of feed forward networks in encoder layers.
        heads_number : int
            The number of heads for multi-head attention per one encoder.
        encoder_layers : int
            The number of encoder layers in Transformer encoder.

        """
        super(Transformer, self).__init__()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._d_model: int = d_model
        encoder_layer: torch.nn.TransformerEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self._d_model,
                                                                                           nhead=heads_number,
                                                                                           dim_feedforward=hidden_dimension,
                                                                                           activation=torch.nn.LeakyReLU(),
                                                                                           batch_first=True,
                                                                                           norm_first=True,
                                                                                           layer_norm_eps=Hyperparameters.Transformer.LAYER_NORM_EPS)

        self._init_weights(encoder_layer.linear1, encoder_layer.linear2)
        self._encoder: torch.nn.TransformerEncoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                                                 num_layers=encoder_layers)

    def _init_weights(self, *layers: torch.nn.Linear) -> None:
        """
        Initialize each layer weights and biases with Xavier method.

        Parameters
        ----------
        *layers : torch.nn.Linear
            The feed forward layers from encoder layer.

        """
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(Hyperparameters.INITIAL_BIAS)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The batch of the data to be propagated.

        Returns
        -------
        x : torch.Tensor
            The batch of the data after propagation.

        """
        self._logger.debug("Starting forward pass!")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._encoder.forward(x)
        self._logger.debug("TransformerEncoder(x) done")
        self._logger.debug(f"Data size: {x.size()}")
        return x

