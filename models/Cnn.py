from torch.nn import Module, Conv1d, MaxPool1d, ReLU, BatchNorm1d, Linear, Flatten, Dropout
from torch.nn.init import xavier_uniform_
from constants import Hyperparameters
from typing import List, Tuple

import logging
import torch


class Cnn(Module):
    def __init__(self, ecg_channels, output_dim, input_dim) -> None:
        super(Cnn, self).__init__()
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.ecg_channels: int = ecg_channels

        # Layer 1
        self.conv1: Conv1d = Conv1d(in_channels=self.ecg_channels,
                                    out_channels=self.ecg_channels,
                                    kernel_size=Hyperparameters.Cnn.Layer1.Conv.KERNEL_SIZE,
                                    stride=Hyperparameters.Cnn.Layer1.Conv.STRIDE, 
                                    padding=Hyperparameters.Cnn.Layer1.Conv.PADDING)
        self.norm1: BatchNorm1d = BatchNorm1d(self.ecg_channels)
        self.relu1: ReLU = ReLU()
        self.pool1: MaxPool1d = MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer1.MaxPool.KERNEL_SIZE,
                                          stride=Hyperparameters.Cnn.Layer1.MaxPool.STRIDE,
                                          padding=Hyperparameters.Cnn.Layer1.MaxPool.PADDING)
        self.drop1: Dropout = Dropout(Hyperparameters.Cnn.Layer1.DROPOUT)

        # Layer 2
        self.conv2: Conv1d = Conv1d(in_channels=self.ecg_channels,
                                    out_channels=self.ecg_channels,
                                    kernel_size=Hyperparameters.Cnn.Layer2.Conv.KERNEL_SIZE,
                                    stride=Hyperparameters.Cnn.Layer2.Conv.STRIDE, 
                                    padding=Hyperparameters.Cnn.Layer2.Conv.PADDING)
        self.norm2: BatchNorm1d = BatchNorm1d(self.ecg_channels)
        self.relu2: ReLU = ReLU()
        self.pool2: MaxPool1d = MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer2.MaxPool.KERNEL_SIZE,
                                          stride=Hyperparameters.Cnn.Layer2.MaxPool.STRIDE,
                                          padding=Hyperparameters.Cnn.Layer2.MaxPool.PADDING)
        self.drop2: Dropout = Dropout(Hyperparameters.Cnn.Layer2.DROPOUT)

        # Layer 3
        self.conv3: Conv1d = Conv1d(in_channels=self.ecg_channels,
                                    out_channels=self.ecg_channels,
                                    kernel_size=Hyperparameters.Cnn.Layer3.Conv.KERNEL_SIZE,
                                    stride=Hyperparameters.Cnn.Layer3.Conv.STRIDE, 
                                    padding=Hyperparameters.Cnn.Layer3.Conv.PADDING)
        self.norm3: BatchNorm1d = BatchNorm1d(self.ecg_channels)
        self.relu3: ReLU = ReLU()
        self.pool3: MaxPool1d = MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer3.MaxPool.KERNEL_SIZE,
                                          stride=Hyperparameters.Cnn.Layer3.MaxPool.STRIDE,
                                          padding=Hyperparameters.Cnn.Layer3.MaxPool.PADDING)
        self.drop3: Dropout = Dropout(Hyperparameters.Cnn.Layer3.DROPOUT)

        # Layer 4
        self.conv4: Conv1d = Conv1d(in_channels=self.ecg_channels,
                                    out_channels=self.ecg_channels,
                                    kernel_size=Hyperparameters.Cnn.Layer4.Conv.KERNEL_SIZE,
                                    stride=Hyperparameters.Cnn.Layer4.Conv.STRIDE, 
                                    padding=Hyperparameters.Cnn.Layer4.Conv.PADDING)
        self.norm4: BatchNorm1d = BatchNorm1d(self.ecg_channels)
        self.relu4: ReLU = ReLU()
        self.pool4: MaxPool1d = MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer4.MaxPool.KERNEL_SIZE,
                                          stride=Hyperparameters.Cnn.Layer4.MaxPool.STRIDE,
                                          padding=Hyperparameters.Cnn.Layer4.MaxPool.PADDING)
        self.drop4: Dropout = Dropout(Hyperparameters.Cnn.Layer4.DROPOUT)

        # Layer 5
        self.conv5: Conv1d = Conv1d(in_channels=self.ecg_channels,
                                    out_channels=self.ecg_channels,
                                    kernel_size=Hyperparameters.Cnn.Layer5.Conv.KERNEL_SIZE,
                                    stride=Hyperparameters.Cnn.Layer5.Conv.STRIDE, 
                                    padding=Hyperparameters.Cnn.Layer5.Conv.PADDING)
        self.norm5: BatchNorm1d = BatchNorm1d(self.ecg_channels)
        self.relu5: ReLU = ReLU()
        self.pool5: MaxPool1d = MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer5.MaxPool.KERNEL_SIZE,
                                          stride=Hyperparameters.Cnn.Layer5.MaxPool.STRIDE,
                                          padding=Hyperparameters.Cnn.Layer5.MaxPool.PADDING)
        self.drop5: Dropout = Dropout(Hyperparameters.Cnn.Layer5.DROPOUT)

        # Layer 6
        self.conv6: Conv1d = Conv1d(in_channels=self.ecg_channels,
                                    out_channels=self.ecg_channels,
                                    kernel_size=Hyperparameters.Cnn.Layer6.Conv.KERNEL_SIZE,
                                    stride=Hyperparameters.Cnn.Layer6.Conv.STRIDE, 
                                    padding=Hyperparameters.Cnn.Layer6.Conv.PADDING)
        self.norm6: BatchNorm1d = BatchNorm1d(self.ecg_channels)
        self.relu6: ReLU = ReLU()
        self.pool6: MaxPool1d = MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer6.MaxPool.KERNEL_SIZE,
                                          stride=Hyperparameters.Cnn.Layer6.MaxPool.STRIDE,
                                          padding=Hyperparameters.Cnn.Layer6.MaxPool.PADDING)
        self.drop6: Dropout = Dropout(Hyperparameters.Cnn.Layer6.DROPOUT)

        # Layer 7
        self.conv7: Conv1d = Conv1d(in_channels=self.ecg_channels,
                                    out_channels=self.ecg_channels,
                                    kernel_size=Hyperparameters.Cnn.Layer7.Conv.KERNEL_SIZE,
                                    stride=Hyperparameters.Cnn.Layer7.Conv.STRIDE, 
                                    padding=Hyperparameters.Cnn.Layer7.Conv.PADDING)
        self.norm7: BatchNorm1d = BatchNorm1d(self.ecg_channels)
        self.relu7: ReLU = ReLU()
        self.pool7: MaxPool1d = MaxPool1d(kernel_size=Hyperparameters.Cnn.Layer7.MaxPool.KERNEL_SIZE,
                                          stride=Hyperparameters.Cnn.Layer7.MaxPool.STRIDE,
                                          padding=Hyperparameters.Cnn.Layer7.MaxPool.PADDING)
        self.drop7: Dropout = Dropout(Hyperparameters.Cnn.Layer7.DROPOUT)

        # Head
        self.flat: Flatten = Flatten()
        self.linear: Linear = Linear(self._calculate_last_layer_dim(input_dim), output_dim)
        self._init_weights()

    def _calculate_output_dim(self, input_dim: int, kernel_size: int, padding: int, stride: int) -> int:
        output_dim: int = (input_dim - kernel_size + 2 * padding) // stride + 1
        return output_dim

    def _calculate_next_layer_size(self, input_dim: int, conv_kernel_size: int, conv_padding: int, conv_stride: int, 
                                   maxpool_kernel_size: int, maxpool_padding: int, maxpool_stride: int) -> int:
        conv_output_dim: int = self._calculate_output_dim(input_dim, conv_kernel_size, conv_padding, conv_stride)
        output_dim: int = self._calculate_output_dim(conv_output_dim, maxpool_kernel_size, maxpool_padding, maxpool_stride)
        return output_dim

    def _calculate_last_layer_dim(self, input_dim: int) -> int:
        mutable_size_layers: List[Module] = [module for module in self.modules() if isinstance(module, Conv1d) or isinstance(module, MaxPool1d)]
        conv_pool_layers_pairs: List[Tuple[Module, Module]] = list(zip(mutable_size_layers[0::2], mutable_size_layers[1::2]))
        dim: int = input_dim
        self.logger.debug(f"Input dim = {dim}")

        for conv, maxpool in conv_pool_layers_pairs:
            dim = self._calculate_next_layer_size(dim, conv.kernel_size[0], conv.padding[0], conv.stride[0],
                                                  maxpool.kernel_size, maxpool.padding, maxpool.stride)
        
        dim *= self.ecg_channels
        self.logger.debug(f"Last layer dim = {dim}")
        return dim

    def _init_weights(self) -> None:
        xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(Hyperparameters.INITIAL_BIAS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.debug("Starting forward pass!")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.drop1(self.pool1(self.relu1(self.norm1(self.conv1(x)))))
        self.logger.debug("drop(pool(relu(norm(conv2(x))))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.drop2(self.pool2(self.relu2(self.norm2(self.conv2(x)))))
        self.logger.debug("drop(pool(relu(norm(conv2(x))))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.drop3(self.pool3(self.relu3(self.norm3(self.conv3(x)))))
        self.logger.debug("drop(pool(relu(norm(conv2(x))))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.drop4(self.pool4(self.relu4(self.norm4(self.conv4(x)))))
        self.logger.debug("drop(pool(relu(norm(conv2(x))))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.drop5(self.pool5(self.relu5(self.norm5(self.conv5(x)))))
        self.logger.debug("drop(pool(relu(norm(conv2(x))))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.drop6(self.pool6(self.relu6(self.norm6(self.conv6(x)))))
        self.logger.debug("drop(pool(relu(norm(conv2(x))))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.drop7(self.pool7(self.relu7(self.norm7(self.conv7(x)))))
        self.logger.debug("drop(pool(relu(norm(conv2(x))))) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.flat(x)
        self.logger.debug("flat(x) done")
        self.logger.debug(f"Data size: {x.size()}")
        x = self.linear(x)
        self.logger.debug("linear(x) done")
        self.logger.debug(f"Data size: {x.size()}")
        return x

