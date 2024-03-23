from constants import Hyperparameters
from typing import List, Tuple

import logging
import torch


class Cnn(torch.nn.Module):
    """
    Cnn is a model built from 7 similar layers and linear head. Each layer consists of 5 sublayers: 1D convolution, 1D batch normalization, activation (ReLU),
    1D max pooling, and dropout. Each layer has its own set of fixed parameters for its sublayers. Head is composed of flattening and linear layers.
    Network accepts input with multiple channels.

    Attributes
    ----------
    _logger : logging.Logger
        Used for logging purposes.
    _channels : int
        The number of channels for input data.
    _conv1 : torch.nn.Conv1d
        First 1D convolutional sublayer.
    _norm1 : torch.nn.BatchNorm1d
        First 1D batch normalization sublayer.
    _relu1 : torch.nn.ReLU
        First activation sublayer.
    _pool1 : torch.nn.MaxPool1d
        First 1D max pooling sublayer.
    _drop1 : torch.nn.Dropout
        First dropout sublayer. Drops neurons with probability 0.03.
    _conv2 : torch.nn.Conv1d
        Second 1D convolutional sublayer.
    _norm2 : torch.nn.BatchNorm1d
        Second 1D batch normalization sublayer.
    _relu2 : torch.nn.ReLU
        Second activation sublayer.
    _pool2 : torch.nn.MaxPool1d
        Second 1D max pooling sublayer.
    _drop2 : torch.nn.Dropout
        Second dropout sublayer. Drops neurons with probability 0.03.
    _conv3 : torch.nn.Conv1d
        Third 1D convolutional sublayer.
    _norm3 : torch.nn.BatchNorm1d
        Third 1D batch normalization sublayer.
    _relu3 : torch.nn.ReLU
        Third activation sublayer.
    _pool3 : torch.nn.MaxPool1d
        Third 1D max pooling sublayer.
    _drop3 : torch.nn.Dropout
        Third dropout sublayer. Drops neurons with probability 0.03.
    _conv4 : torch.nn.Conv1d
        Fourth 1D convolutional sublayer.
    _norm4 : torch.nn.BatchNorm1d
        Fourth 1D batch normalization sublayer.
    _relu4 : torch.nn.ReLU
        Fourth activation sublayer.
    _pool4 : torch.nn.MaxPool1d
        Fourth 1D max pooling sublayer.
    _drop4 : torch.nn.Dropout
        Fourth dropout sublayer. Drops neurons with probability 0.03.
    _conv5 : torch.nn.Conv1d
        Fifth 1D convolutional sublayer.
    _norm5 : torch.nn.BatchNorm1d
        Fifth 1D batch normalization sublayer.
    _relu5 : torch.nn.ReLU
        Fifth activation sublayer.
    _pool5 : torch.nn.MaxPool1d
        Fifth 1D max pooling sublayer.
    _drop5 : torch.nn.Dropout
        Fifth dropout sublayer. Drops neurons with probability 0.03.
    _conv6 : torch.nn.Conv1d
        Sixth 1D convolutional sublayer.
    _norm6 : torch.nn.BatchNorm1d
        Sixth 1D batch normalization sublayer.
    _relu6 : torch.nn.ReLU
        Sixth activation sublayer.
    _pool6 : torch.nn.MaxPool1d
        Sixth 1D max pooling sublayer.
    _drop6 : torch.nn.Dropout
        Sixth dropout sublayer. Drops neurons with probability 0.03.
    _conv7 : torch.nn.Conv1d
        Seventh 1D convolutional sublayer.
    _norm7 : torch.nn.BatchNorm1d
        Seventh 1D batch normalization sublayer.
    _relu7 : torch.nn.ReLU
        Seventh activation sublayer.
    _pool7 : torch.nn.MaxPool1d
        Seventh 1D max pooling sublayer.
    _drop7 : torch.nn.Dropout
        Seventh dropout sublayer. Drops neurons with probability 0.03.
    _flat : torch.nn.Flatten
        Flatten layer to fit output from previous layers to the head.
    _linear : torch.nn.Linear
        The output of the model with variable output dimension.

    Examples
    --------
    X = <load features e.g. from EcgSignalLoader>
    model = Cnn(channels=12, input_dim=128, output_dim=32)
    predictions = model(X)

    """
    def __init__(self, channels: int, input_dim: int, output_dim: int) -> None:
        """
        Initiate CNN with the default values for layers and head with variable size.

        Parameters
        ----------
        channels : int
            The number of channels for the input data used in every 1D convolution layer.
        input_dim : int
            The input dimension used for calculations of the input size for head linear layer.
        output_dim : int
            The output dimension of the whole network.

        """
        super(Cnn, self).__init__()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._channels: int = channels

        # Layer 1
        self._conv1: torch.nn.Conv1d = torch.nn.Conv1d(in_channels=self._channels,
                                                       out_channels=self._channels,
                                                       kernel_size=Hyperparameters.Cnn.Layer1.Conv.KERNEL_SIZE,
                                                       stride=Hyperparameters.Cnn.Layer1.Conv.STRIDE, 
                                                       padding=Hyperparameters.Cnn.Layer1.Conv.PADDING)
        self._norm1: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self._channels)
        self._relu1: torch.nn.ReLU = torch.nn.ReLU()
        self._pool1: torch.nn.MaxPool1d = torch.nn.MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer1.MaxPool.KERNEL_SIZE,
                                                             stride=Hyperparameters.Cnn.Layer1.MaxPool.STRIDE,
                                                             padding=Hyperparameters.Cnn.Layer1.MaxPool.PADDING)
        self._drop1: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.Cnn.Layer1.DROPOUT)

        # Layer 2
        self._conv2: torch.nn.Conv1d = torch.nn.Conv1d(in_channels=self._channels,
                                                       out_channels=self._channels,
                                                       kernel_size=Hyperparameters.Cnn.Layer2.Conv.KERNEL_SIZE,
                                                       stride=Hyperparameters.Cnn.Layer2.Conv.STRIDE, 
                                                       padding=Hyperparameters.Cnn.Layer2.Conv.PADDING)
        self._norm2: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self._channels)
        self._relu2: torch.nn.ReLU = torch.nn.ReLU()
        self._pool2: torch.nn.MaxPool1d = torch.nn.MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer2.MaxPool.KERNEL_SIZE,
                                                             stride=Hyperparameters.Cnn.Layer2.MaxPool.STRIDE,
                                                             padding=Hyperparameters.Cnn.Layer2.MaxPool.PADDING)
        self._drop2: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.Cnn.Layer2.DROPOUT)

        # Layer 3
        self._conv3: torch.nn.Conv1d = torch.nn.Conv1d(in_channels=self._channels,
                                                       out_channels=self._channels,
                                                       kernel_size=Hyperparameters.Cnn.Layer3.Conv.KERNEL_SIZE,
                                                       stride=Hyperparameters.Cnn.Layer3.Conv.STRIDE, 
                                                       padding=Hyperparameters.Cnn.Layer3.Conv.PADDING)
        self._norm3: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self._channels)
        self._relu3: torch.nn.ReLU = torch.nn.ReLU()
        self._pool3: torch.nn.MaxPool1d = torch.nn.MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer3.MaxPool.KERNEL_SIZE,
                                                             stride=Hyperparameters.Cnn.Layer3.MaxPool.STRIDE,
                                                             padding=Hyperparameters.Cnn.Layer3.MaxPool.PADDING)
        self._drop3: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.Cnn.Layer3.DROPOUT)

        # Layer 4
        self._conv4: torch.nn.Conv1d = torch.nn.Conv1d(in_channels=self._channels,
                                                       out_channels=self._channels,
                                                       kernel_size=Hyperparameters.Cnn.Layer4.Conv.KERNEL_SIZE,
                                                       stride=Hyperparameters.Cnn.Layer4.Conv.STRIDE, 
                                                       padding=Hyperparameters.Cnn.Layer4.Conv.PADDING)
        self._norm4: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self._channels)
        self._relu4: torch.nn.ReLU = torch.nn.ReLU()
        self._pool4: torch.nn.MaxPool1d = torch.nn.MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer4.MaxPool.KERNEL_SIZE,
                                                             stride=Hyperparameters.Cnn.Layer4.MaxPool.STRIDE,
                                                             padding=Hyperparameters.Cnn.Layer4.MaxPool.PADDING)
        self._drop4: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.Cnn.Layer4.DROPOUT)

        # Layer 5
        self._conv5: torch.nn.Conv1d = torch.nn.Conv1d(in_channels=self._channels,
                                                       out_channels=self._channels,
                                                       kernel_size=Hyperparameters.Cnn.Layer5.Conv.KERNEL_SIZE,
                                                       stride=Hyperparameters.Cnn.Layer5.Conv.STRIDE, 
                                                       padding=Hyperparameters.Cnn.Layer5.Conv.PADDING)
        self._norm5: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self._channels)
        self._relu5: torch.nn.ReLU = torch.nn.ReLU()
        self._pool5: torch.nn.MaxPool1d = torch.nn.MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer5.MaxPool.KERNEL_SIZE,
                                                             stride=Hyperparameters.Cnn.Layer5.MaxPool.STRIDE,
                                                             padding=Hyperparameters.Cnn.Layer5.MaxPool.PADDING)
        self._drop5: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.Cnn.Layer5.DROPOUT)

        # Layer 6
        self._conv6: torch.nn.Conv1d = torch.nn.Conv1d(in_channels=self._channels,
                                                       out_channels=self._channels,
                                                       kernel_size=Hyperparameters.Cnn.Layer6.Conv.KERNEL_SIZE,
                                                       stride=Hyperparameters.Cnn.Layer6.Conv.STRIDE, 
                                                       padding=Hyperparameters.Cnn.Layer6.Conv.PADDING)
        self._norm6: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self._channels)
        self._relu6: torch.nn.ReLU = torch.nn.ReLU()
        self._pool6: torch.nn.MaxPool1d = torch.nn.MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer6.MaxPool.KERNEL_SIZE,
                                                             stride=Hyperparameters.Cnn.Layer6.MaxPool.STRIDE,
                                                             padding=Hyperparameters.Cnn.Layer6.MaxPool.PADDING)
        self._drop6: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.Cnn.Layer6.DROPOUT)

        # Layer 7
        self._conv7: torch.nn.Conv1d = torch.nn.Conv1d(in_channels=self._channels,
                                                       out_channels=self._channels,
                                                       kernel_size=Hyperparameters.Cnn.Layer7.Conv.KERNEL_SIZE,
                                                       stride=Hyperparameters.Cnn.Layer7.Conv.STRIDE, 
                                                       padding=Hyperparameters.Cnn.Layer7.Conv.PADDING)
        self._norm7: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self._channels)
        self._relu7: torch.nn.ReLU = torch.nn.ReLU()
        self._pool7: torch.nn.MaxPool1d = torch.nn.MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer7.MaxPool.KERNEL_SIZE,
                                                             stride=Hyperparameters.Cnn.Layer7.MaxPool.STRIDE,
                                                             padding=Hyperparameters.Cnn.Layer7.MaxPool.PADDING)
        self._drop7: torch.nn.Dropout = torch.nn.Dropout(Hyperparameters.Cnn.Layer7.DROPOUT)

        # Head
        self._flat: torch.nn.Flatten = torch.nn.Flatten()
        self._linear: torch.nn.Linear = torch.nn.Linear(self._calculate_last_layer_dim(input_dim), output_dim)
        self._init_weights()

    def _calculate_output_dim(self, input_dim: int, kernel_size: int, padding: int, stride: int) -> int:
        """
        Calculate output dimension from 1D convolution/max pooling layer based on input data size, kernel size, padding, and stride.

        Parameters
        ----------
        input_dim : int
            The size of input data.
        kernel_size : int
            The kernel size.
        padding : int
            The padding of the filter.
        stride : int
            The stride of the filter.

        Returns
        -------
        output_dim : int
            The calculated output dimension for the layer.

        See also
        --------
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        """
        output_dim: int = (input_dim - kernel_size + 2 * padding) // stride + 1
        return output_dim

    def _calculate_next_layer_size(self, input_dim: int, conv_kernel_size: int, conv_padding: int, conv_stride: int, 
                                   maxpool_kernel_size: int, maxpool_padding: int, maxpool_stride: int) -> int:
        """
        Calculate the size of the next Cnn layer.

        Parameters
        ----------
        input_dim : int
            The input dimension of the previous Cnn layer.
        conv_kernel_size : int
            The kernel size of the 1D convolution sublayer from the next layer.
        conv_padding : int
            The padding of the 1D convolution sublayer from the next layer.
        conv_stride : int
            The stride of the 1D convolution sublayer from the next layer.
        maxpool_kernel_size : int
            The kernel size of the 1D max pooling sublayer from the next layer.
        maxpool_padding : int
            The padding of the 1D max pooling sublayer from the next layer.
        maxpool_stride : int
            The stride of the 1D max pooling sublayer from the next layer.

        Returns
        -------
        output_dim : int
            The output dimension of the next Cnn layer.

        """
        conv_output_dim: int = self._calculate_output_dim(input_dim, conv_kernel_size, conv_padding, conv_stride)
        output_dim: int = self._calculate_output_dim(conv_output_dim, maxpool_kernel_size, maxpool_padding, maxpool_stride)
        return output_dim

    def _calculate_last_layer_dim(self, input_dim: int) -> int:
        """
        Calculate the input dimension for the Cnn linear layer.

        Parameters
        ----------
        input_dim : int
            The size of the input data.

        Returns
        -------
        dim : int
            The data dimension after applying all Cnn layers but head.

        """
        mutable_size_layers: List[torch.nn.Module] = [module for module in self.modules() if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.MaxPool1d)]
        conv_pool_layers_pairs: List[Tuple[torch.nn.Module, torch.nn.Module]] = list(zip(mutable_size_layers[0::2], mutable_size_layers[1::2]))
        dim: int = input_dim
        self._logger.debug(f"Input dim = {dim}")

        for conv, maxpool in conv_pool_layers_pairs:
            dim = self._calculate_next_layer_size(dim, conv.kernel_size[0], conv.padding[0], conv.stride[0],
                                                  maxpool.kernel_size, maxpool.padding, maxpool.stride)
        
        dim *= self._channels
        self._logger.debug(f"Last layer dim = {dim}")
        return dim

    def _init_weights(self) -> None:
        """Initialize linear layer weights and biases with Xavier method."""
        torch.nn.init.xavier_uniform_(self._linear.weight)
        self._linear.bias.data.fill_(Hyperparameters.INITIAL_BIAS)

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
        x = self._drop1(self._pool1(self._relu1(self._norm1(self._conv1(x)))))
        self._logger.debug("drop(pool(relu(norm(conv(x))))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._drop2(self._pool2(self._relu2(self._norm2(self._conv2(x)))))
        self._logger.debug("drop(pool(relu(norm(conv(x))))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._drop3(self._pool3(self._relu3(self._norm3(self._conv3(x)))))
        self._logger.debug("drop(pool(relu(norm(conv(x))))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._drop4(self._pool4(self._relu4(self._norm4(self._conv4(x)))))
        self._logger.debug("drop(pool(relu(norm(conv(x))))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._drop5(self._pool5(self._relu5(self._norm5(self._conv5(x)))))
        self._logger.debug("drop(pool(relu(norm(conv(x))))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._drop6(self._pool6(self._relu6(self._norm6(self._conv6(x)))))
        self._logger.debug("drop(pool(relu(norm(conv(x))))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._drop7(self._pool7(self._relu7(self._norm7(self._conv7(x)))))
        self._logger.debug("drop(pool(relu(norm(conv(x))))) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._flat(x)
        self._logger.debug("flat(x) done")
        self._logger.debug(f"Data size: {x.size()}")
        x = self._linear(x)
        self._logger.debug("linear(x) done")
        self._logger.debug(f"Data size: {x.size()}")
        return x

