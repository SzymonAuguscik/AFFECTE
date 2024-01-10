from .FeedForwardClassifier import FeedForwardClassifier
from constants import Hyperparameters
from .Transformer import Transformer
from torch.nn import Module, Linear
from .Cnn import Cnn

import logging


class AtrialFibrillationDetector(Module):
    def __init__(self, transformer_dimension, ecg_channels, window_length,
                 transformer_hidden_dimension=Hyperparameters.Transformer.HIDDEN_LAYER,
                 use_cnn=False, use_transformer=False):
        super(AtrialFibrillationDetector, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        self.name = f"AFD({'CNN,' if self.use_cnn else ''}{'TN,' if self.use_transformer else ''}FFN)"
        self.logger.info(f"Model {self.name}")

        self.cnn = Cnn(ecg_channels, transformer_dimension, window_length)
        self.linear = Linear(window_length, transformer_dimension)
        self.transformer = Transformer(transformer_dimension, hidden_dimension=transformer_hidden_dimension)
        
        ffc_input_dimension = transformer_dimension if self.use_cnn or self.use_transformer else window_length
        self.logger.debug(f"FeedForwardClassifier input dimension = {ffc_input_dimension}")
        self.ff = FeedForwardClassifier(ffc_input_dimension)

        self.hyperparameters = {
            Hyperparameters.Names.TRANSFORMER_DIMENSION        : transformer_dimension,
            Hyperparameters.Names.ECG_CHANNELS                 : ecg_channels,
            Hyperparameters.Names.WINDOW_LENGTH                : window_length,
            Hyperparameters.Names.TRANSFORMER_HIDDEN_DIMENSION : transformer_hidden_dimension,
        }

        self.embedded = None

    def get_embedded(self):
        return self.embedded

    def get_hyperparameters(self):
        return self.hyperparameters
    
    def forward(self, x):
        self.logger.debug("Starting forward pass!")
        self.logger.debug(f"Data size: {x.size()}")
        # TODO potential refactor - booleans
        if self.use_cnn:
            x = self.cnn(x)
            self.logger.debug("cnn(x) done")
            self.logger.debug(f"Data size: {x.size()}")
        if self.use_transformer:
            if not self.use_cnn:
                x = self.linear(x)
                self.logger.debug("linear(x) done")
                self.logger.debug(f"Data size: {x.size()}")    
            x = self.transformer(x)
            self.embedded = x
            self.logger.debug("transformer(x) done")
            self.logger.debug(f"Data size: {x.size()}")
        x = self.ff(x)
        self.logger.debug("ff(x) done")
        self.logger.debug(f"Data size: {x.size()}")
        return x

