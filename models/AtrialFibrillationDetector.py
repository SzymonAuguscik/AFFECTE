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
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._use_cnn: bool = use_cnn
        self._use_transformer: bool = use_transformer
        self._name: str = f"AFD({'CNN,' if self._use_cnn else ''}{'TN,' if self._use_transformer else ''}FFN)"
        self._logger.info(f"Model {self._name}")

        self._cnn: Cnn = Cnn(ecg_channels, transformer_dimension, window_length) if self._use_cnn else None
        self._linear: torch.nn.Linear = torch.nn.Linear(window_length, transformer_dimension) if self._use_transformer and not self._use_cnn else None
        self._transformer: Transformer = Transformer(d_model=transformer_dimension,
                                                    hidden_dimension=transformer_hidden_dimension,
                                                    heads_number=transformer_heads,
                                                    encoder_layers=transformer_encoder_layers) if self._use_transformer else None
        
        ffc_input_dimension: int = transformer_dimension if self._use_cnn or self._use_transformer else window_length
        self._logger.debug(f"FeedForwardClassifier input dimension = {ffc_input_dimension}")
        self._ff: FeedForwardClassifier = FeedForwardClassifier(ffc_input_dimension)

        self._hyperparameters: Dict[str, float] = {
            Hyperparameters.Names.TRANSFORMER_DIMENSION        : transformer_dimension,
            Hyperparameters.Names.ECG_CHANNELS                 : ecg_channels,
            Hyperparameters.Names.WINDOW_LENGTH                : window_length,
            Hyperparameters.Names.TRANSFORMER_HIDDEN_DIMENSION : transformer_hidden_dimension,
            Hyperparameters.Names.TRANSFORMER_HEADS            : transformer_heads,
            Hyperparameters.Names.TRANSFORMER_ENCODER_LAYERS   : transformer_encoder_layers,
        }

        self._embedded: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def hyperparameters(self) -> Dict[str, int]:
        return self._hyperparameters

    @property
    def embedded(self) -> Optional[torch.Tensor]:
        return self._embedded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._logger.debug("Starting forward pass!")
        self._logger.debug(f"Data size: {x.size()}")
        # TODO potential refactor - booleans
        if self._use_cnn:
            x = self._cnn(x)
            self._logger.debug("cnn(x) done")
            self._logger.debug(f"Data size: {x.size()}")
        if self._use_transformer:
            if not self._use_cnn:
                x = self._linear(x)
                self._logger.debug("linear(x) done")
                self._logger.debug(f"Data size: {x.size()}")    
            x = self._transformer(x)
            self._embedded = x
            self._logger.debug("transformer(x) done")
            self._logger.debug(f"Data size: {x.size()}")
        x = self._ff(x)
        self._logger.debug("ff(x) done")
        self._logger.debug(f"Data size: {x.size()}")
        return x

