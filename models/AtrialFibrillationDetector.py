from .FeedForwardClassifier import FeedForwardClassifier
from constants import Hyperparameters
from .Transformer import Transformer
from typing import Dict, Optional
from .Cnn import Cnn

import logging
import torch


class AtrialFibrillationDetector(torch.nn.Module):
    def __init__(self, transformer_dimension: int, ecg_channels: int, window_length: int,
                 transformer_hidden_dimension: int, transformer_heads: int, transformer_encoder_layers: int,
                 use_cnn: bool = False, use_transformer: bool = False) -> None:
        super(AtrialFibrillationDetector, self).__init__()
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.use_cnn: bool = use_cnn
        self.use_transformer: bool = use_transformer
        self.name: str = f"AFD({'CNN,' if self.use_cnn else ''}{'TN,' if self.use_transformer else ''}FFN)"
        self.logger.info(f"Model {self.name}")

        self.cnn: Cnn = Cnn(ecg_channels, transformer_dimension, window_length) if self.use_cnn else None
        self.linear: torch.nn.Linear = torch.nn.Linear(window_length, transformer_dimension) if self.use_transformer and not self.use_cnn else None
        self.transformer: Transformer = Transformer(d_model=transformer_dimension,
                                                    hidden_dimension=transformer_hidden_dimension,
                                                    heads_number=transformer_heads,
                                                    encoder_layers=transformer_encoder_layers) if self.use_transformer else None
        
        ffc_input_dimension: int = transformer_dimension if self.use_cnn or self.use_transformer else window_length
        self.logger.debug(f"FeedForwardClassifier input dimension = {ffc_input_dimension}")
        self.ff: FeedForwardClassifier = FeedForwardClassifier(ffc_input_dimension)

        self.hyperparameters: Dict[str, float] = {
            Hyperparameters.Names.TRANSFORMER_DIMENSION        : transformer_dimension,
            Hyperparameters.Names.ECG_CHANNELS                 : ecg_channels,
            Hyperparameters.Names.WINDOW_LENGTH                : window_length,
            Hyperparameters.Names.TRANSFORMER_HIDDEN_DIMENSION : transformer_hidden_dimension,
            Hyperparameters.Names.TRANSFORMER_HEADS            : transformer_heads,
            Hyperparameters.Names.TRANSFORMER_ENCODER_LAYERS   : transformer_encoder_layers,
        }

        self.embedded: Optional[torch.Tensor] = None

    def get_embedded(self) -> Optional[torch.Tensor]:
        return self.embedded

    def get_hyperparameters(self) -> Dict[str, int]:
        return self.hyperparameters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

