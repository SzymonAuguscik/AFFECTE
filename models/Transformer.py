from constants import Hyperparameters

import logging
import torch


class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dimension: int, heads_number: int, encoder_layers: int) -> None:
        super(Transformer, self).__init__()
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.d_model: int = d_model
        encoder_layer: torch.nn.TransformerEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.d_model,
                                                                                           nhead=heads_number,
                                                                                           dim_feedforward=hidden_dimension,
                                                                                           activation=torch.nn.LeakyReLU(),
                                                                                           batch_first=True,
                                                                                           norm_first=True,
                                                                                           layer_norm_eps=Hyperparameters.Transformer.LAYER_NORM_EPS)

        self._init_weights(encoder_layer.linear1, encoder_layer.linear2)
        self.transformer: torch.nn.TransformerEncoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                                                    num_layers=encoder_layers)

    def _init_weights(self, *layers: torch.nn.Linear) -> None:
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
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

