from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, LeakyReLU
from torch.nn.init import xavier_uniform_

import logging
# import torch
# import math


class Transformer(Module):
    def __init__(self, d_model, hidden_dimension):
        super(Transformer, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.d_model = d_model
        # self.positional_encoding = self.create_positional_encoding()

        encoder_layer = encoder_layer=TransformerEncoderLayer(d_model=self.d_model,
                                                              nhead=16,
                                                              dim_feedforward=hidden_dimension,
                                                              activation=LeakyReLU(),
                                                              batch_first=True,
                                                              norm_first=True,
                                                              layer_norm_eps=1e-6)

        self._init_weights(encoder_layer.linear1, encoder_layer.linear2)
        self.transformer = TransformerEncoder(encoder_layer=encoder_layer,
                                              num_layers=8)

    def _init_weights(self, *layers):
        for layer in layers:
            xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    # def create_positional_encoding(self, max_length=5000):
    #     position = torch.arange(0, max_length).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
    #     positional_encoding = torch.zeros(max_length, self.d_model)
    #     positional_encoding[:, 0::2] = torch.sin(position * div_term)
    #     positional_encoding[:, 1::2] = torch.cos(position * div_term)
    #     return positional_encoding.unsqueeze(1)
    
    def forward(self, x):
        self.logger.debug(f"Data size: {x.size()}")
        self.logger.debug("Starting forward pass!")
        self.logger.debug(f"Data size: {x.size()}")
        # self.logger.debug(f"Positonal encoding size: {self.positional_encoding.size()}")
        # self.logger.debug(f"self.positional_encoding[:x.size(0), :] size = {self.positional_encoding[:x.size(0), :].size()}")
        # x = x + self.positional_encoding[:x.size(0), :]
        self.logger.debug("positional_encoding(x) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.transformer.forward(x)
        self.logger.debug("TransformerEncoder(x) done")
        self.logger.debug(f"Data size: {x.size()}")
        return x

