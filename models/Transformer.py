from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, LeakyReLU, Linear
from torch.nn.init import xavier_uniform_
from constants import Hyperparameters

import logging
import torch


class Transformer(Module):
    def __init__(self, d_model: int, hidden_dimension: int, heads_number: int, encoder_layers: int) -> None:
        super(Transformer, self).__init__()
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.d_model: int = d_model
        encoder_layer: TransformerEncoderLayer = TransformerEncoderLayer(d_model=self.d_model,
                                                                         nhead=heads_number,
                                                                         dim_feedforward=hidden_dimension,
                                                                         activation=LeakyReLU(),
                                                                         batch_first=True,
                                                                         norm_first=True,
                                                                         layer_norm_eps=Hyperparameters.Transformer.LAYER_NORM_EPS)

        self._init_weights(encoder_layer.linear1, encoder_layer.linear2)
        self.transformer: TransformerEncoder = TransformerEncoder(encoder_layer=encoder_layer,
                                                                  num_layers=encoder_layers)

    def _init_weights(self, *layers: Linear) -> None:
        for layer in layers:
            xavier_uniform_(layer.weight)
            layer.bias.data.fill_(Hyperparameters.INITIAL_BIAS)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"Data size: {x.size()}")
        self.logger.debug("Starting forward pass!")
        self.logger.debug(f"Data size: {x.size()}")
        self.logger.debug("positional_encoding(x) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.transformer.forward(x)
        self.logger.debug("TransformerEncoder(x) done")
        self.logger.debug(f"Data size: {x.size()}")
        return x

