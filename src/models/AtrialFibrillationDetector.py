from .FeedForwardClassifier import FeedForwardClassifier
from src.constants import Hyperparameters
from .Transformer import Transformer
from typing import Dict, Optional
from .Cnn import Cnn

import logging
import torch


class AtrialFibrillationDetector(torch.nn.Module):
    """
    AtrialFibrillationDetector is a model that can consist of several simpler subnetworks. It can be specified if Cnn or Transformer should be used.
    The only mandatory subnetwork is Feed Forward Network (FFN) which serves as a classification head. The model provides a necessary "connector" layers
    which will automatically adapt the flow of data between subnetworks. This architecture is inspired by the paper mentioned in `See also` section.

    Attributes
    ----------
    _logger : logging.Logger
        Used for logging purposes.
    _use_cnn : bool
        It indicates if Cnn network should be used in the model.
    _use_transformer : bool
        It indicates if Transformer network should be used in the model.
    _name : str
        The model name built with respect to used subnetworks.
    _cnn : Cnn
        The Cnn subnetwork which can be used for data embedding.
    _linear : torch.nn.Linear
        The linear layer that is used when only Cnn subnetwork is not used (in order to fit multichannel data to the Transformer).
    _transformer : Transformer
        The Transformer subnetwork that can be used to catch long-term dependencies in ECG signal.
    _ffc : FeedForwardClassifier
        Feed Forward Network subnetwork which is a classification head.
    _hyperparameters : Dict[str, float]
        The names and values of the model hyperparameters.
    _embedded : Optional[torch.Tensor]
        The output of the Transformer subnetwork. It is present only if _use_transformer is set to True.

    Examples
    --------
    X = <load features e.g. from EcgSignalLoader>
    model = AtrialFibrillationDetector(transformer_dimension=64, ecg_channels=2, window_length=128,
                                       transformer_hidden_dimension=256, transformer_heads=16, transformer_encoder_layers=8,
                                       use_cnn=False, use_transformer=True)
    predictions = model(X)

    for name, value in model.hyperparameters.items():
        print(f"{name} : {value}")

    See also
    --------
    https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01546-2

    """
    def __init__(self, transformer_dimension: int, ecg_channels: int, window_length: int,
                 transformer_hidden_dimension: int, transformer_heads: int, transformer_encoder_layers: int,
                 use_cnn: bool = False, use_transformer: bool = False) -> None:
        """
        Initiates AtrialFibrillationDetector with respect to the hyperparameters and chosen subnetworks.

        Parameters
        ----------
        transformer_dimension : int
            The expected number of input features to the Transformer.
            It is also the output size of the Cnn and input size of the FFN.
        ecg_channels : int
            The number of channels for the input data.
        window_length : int
            The length of the sliding window that where used to split the original signal into chunks.
            It is related to the number of seconds chosen during dataset preparations.
        transformer_hidden_dimension : int
            The dimension for Transformer feed forward layers.
        transformer_heads : int
            The number of heads for Transformer multi-head attention.
        transformer_encoder_layers : int
            The number of encoder layers in Transformer.
        use_cnn : bool, optional
            It indicates if Cnn network should be used in the model.
        use_transformer : bool, optional
            It indicates if Transformer network should be used in the model.

        """
        super(AtrialFibrillationDetector, self).__init__()
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._use_cnn: bool = use_cnn
        self._use_transformer: bool = use_transformer
        self._name: str = f"AFD({'CNN,' if self._use_cnn else ''}{'TN,' if self._use_transformer else ''}FFN)"
        self._logger.info(f"Model {self._name}")

        self._cnn: Cnn = Cnn(ecg_channels, window_length, transformer_dimension) if self._use_cnn else None
        self._linear: torch.nn.Linear = torch.nn.Linear(window_length, transformer_dimension) if self._use_transformer and not self._use_cnn else None
        self._transformer: Transformer = Transformer(d_model=transformer_dimension,
                                                     hidden_dimension=transformer_hidden_dimension,
                                                     heads_number=transformer_heads,
                                                     encoder_layers=transformer_encoder_layers) if self._use_transformer else None
        
        ffc_input_dimension: int = transformer_dimension if self._use_cnn or self._use_transformer else window_length
        self._logger.debug(f"FeedForwardClassifier input dimension = {ffc_input_dimension}")
        self._ffc: FeedForwardClassifier = FeedForwardClassifier(ffc_input_dimension)

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
        """
        Return the model name.

        Returns
        -------
        _name : str
            The model name.

        """
        return self._name

    @property
    def hyperparameters(self) -> Dict[str, int]:
        """
        Return the model hyperparameters.

        Returns
        -------
        _hyperparameters : Dict[str, int]
            The model hyperparameters.

        """
        return self._hyperparameters

    @property
    def embedded(self) -> Optional[torch.Tensor]:
        """
        Return the embedded vectors from Transformer.

        Returns
        -------
        _embedded : Optional[torch.Tensor]
            The embedded vectors from Transformer.

        """
        return self._embedded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.
        It automatically adapts to the chosen subnetworks.

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
        x = self._ffc(x)
        self._logger.debug("ff(x) done")
        self._logger.debug(f"Data size: {x.size()}")
        return x

