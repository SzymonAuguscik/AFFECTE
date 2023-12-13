from torch.nn import Module, Conv1d, MaxPool1d, ReLU, BatchNorm1d, Linear, Flatten, Dropout
from torch.nn.init import xavier_uniform_
from constants import CnnLayers

import logging


class Cnn(Module):
    def __init__(self, ecg_channels, output_dim, input_dim):
        super(Cnn, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.ecg_channels = ecg_channels

        # Layer 1
        self.conv1 = Conv1d(in_channels=self.ecg_channels,
                            out_channels=self.ecg_channels,
                            kernel_size=CnnLayers.Layer1.Conv.KERNEL_SIZE,
                            stride=CnnLayers.Layer1.Conv.STRIDE, 
                            padding=CnnLayers.Layer1.Conv.PADDING)
        self.norm1 = BatchNorm1d(self.ecg_channels)
        self.relu1 = ReLU()
        self.pool1 = MaxPool1d(kernel_size=CnnLayers.Layer1.MaxPool.KERNEL_SIZE,
                               stride=CnnLayers.Layer1.MaxPool.STRIDE,
                               padding=CnnLayers.Layer1.MaxPool.PADDING)
        self.drop1 = Dropout(0.03)

        # Layer 2
        self.conv2 = Conv1d(in_channels=self.ecg_channels,
                            out_channels=self.ecg_channels,
                            kernel_size=CnnLayers.Layer2.Conv.KERNEL_SIZE,
                            stride=CnnLayers.Layer2.Conv.STRIDE, 
                            padding=CnnLayers.Layer2.Conv.PADDING)
        self.norm2 = BatchNorm1d(self.ecg_channels)
        self.relu2 = ReLU()
        self.pool2 = MaxPool1d(kernel_size=CnnLayers.Layer2.MaxPool.KERNEL_SIZE,
                               stride=CnnLayers.Layer2.MaxPool.STRIDE,
                               padding=CnnLayers.Layer2.MaxPool.PADDING)
        self.drop2 = Dropout(0.03)

        # Layer 3
        self.conv3 = Conv1d(in_channels=self.ecg_channels,
                            out_channels=self.ecg_channels,
                            kernel_size=CnnLayers.Layer3.Conv.KERNEL_SIZE,
                            stride=CnnLayers.Layer3.Conv.STRIDE, 
                            padding=CnnLayers.Layer3.Conv.PADDING)
        self.norm3 = BatchNorm1d(self.ecg_channels)
        self.relu3 = ReLU()
        self.pool3 = MaxPool1d(kernel_size=CnnLayers.Layer3.MaxPool.KERNEL_SIZE,
                               stride=CnnLayers.Layer3.MaxPool.STRIDE,
                               padding=CnnLayers.Layer3.MaxPool.PADDING)
        self.drop3 = Dropout(0.03)

        # Layer 4
        self.conv4 = Conv1d(in_channels=self.ecg_channels,
                            out_channels=self.ecg_channels,
                            kernel_size=CnnLayers.Layer4.Conv.KERNEL_SIZE,
                            stride=CnnLayers.Layer4.Conv.STRIDE, 
                            padding=CnnLayers.Layer4.Conv.PADDING)
        self.norm4 = BatchNorm1d(self.ecg_channels)
        self.relu4 = ReLU()
        self.pool4 = MaxPool1d(kernel_size=CnnLayers.Layer4.MaxPool.KERNEL_SIZE,
                               stride=CnnLayers.Layer4.MaxPool.STRIDE,
                               padding=CnnLayers.Layer4.MaxPool.PADDING)
        self.drop4 = Dropout(0.03)

        # Layer 5
        self.conv5 = Conv1d(in_channels=self.ecg_channels,
                            out_channels=self.ecg_channels,
                            kernel_size=CnnLayers.Layer5.Conv.KERNEL_SIZE,
                            stride=CnnLayers.Layer5.Conv.STRIDE, 
                            padding=CnnLayers.Layer5.Conv.PADDING)
        self.norm5 = BatchNorm1d(self.ecg_channels)
        self.relu5 = ReLU()
        self.pool5 = MaxPool1d(kernel_size=CnnLayers.Layer5.MaxPool.KERNEL_SIZE,
                               stride=CnnLayers.Layer5.MaxPool.STRIDE,
                               padding=CnnLayers.Layer5.MaxPool.PADDING)
        self.drop5 = Dropout(0.03)

        # Layer 6
        self.conv6 = Conv1d(in_channels=self.ecg_channels,
                            out_channels=self.ecg_channels,
                            kernel_size=CnnLayers.Layer6.Conv.KERNEL_SIZE,
                            stride=CnnLayers.Layer6.Conv.STRIDE, 
                            padding=CnnLayers.Layer6.Conv.PADDING)
        self.norm6 = BatchNorm1d(self.ecg_channels)
        self.relu6 = ReLU()
        self.pool6 = MaxPool1d(kernel_size=CnnLayers.Layer6.MaxPool.KERNEL_SIZE,
                               stride=CnnLayers.Layer6.MaxPool.STRIDE,
                               padding=CnnLayers.Layer6.MaxPool.PADDING)
        self.drop6 = Dropout(0.03)

        # Layer 7
        self.conv7 = Conv1d(in_channels=self.ecg_channels,
                            out_channels=self.ecg_channels,
                            kernel_size=CnnLayers.Layer7.Conv.KERNEL_SIZE,
                            stride=CnnLayers.Layer7.Conv.STRIDE, 
                            padding=CnnLayers.Layer7.Conv.PADDING)
        self.norm7 = BatchNorm1d(self.ecg_channels)
        self.relu7 = ReLU()
        self.pool7 = MaxPool1d(kernel_size=CnnLayers.Layer7.MaxPool.KERNEL_SIZE,
                               stride=CnnLayers.Layer7.MaxPool.STRIDE,
                               padding=CnnLayers.Layer7.MaxPool.PADDING)
        self.drop7 = Dropout(0.03)

        # Head
        self.flat = Flatten()
        self.linear = Linear(self._calculate_last_layer_dim(input_dim), output_dim)
        self._init_weights()

    def _calculate_output_dim(self, input_dim, kernel_size, padding, stride):
        output_dim = (input_dim - kernel_size + 2 * padding) // stride + 1
        return output_dim

    def _calculate_next_layer_size(self, input_dim, conv_kernel_size, conv_padding, conv_stride, maxpool_kernel_size, maxpool_padding, maxpool_stride):
        conv_output_dim = self._calculate_output_dim(input_dim, conv_kernel_size, conv_padding, conv_stride)
        output_dim = self._calculate_output_dim(conv_output_dim, maxpool_kernel_size, maxpool_padding, maxpool_stride)
        return output_dim

    def _calculate_last_layer_dim(self, input_dim):
        layers = [module for module in self.modules() if isinstance(module, Conv1d) or isinstance(module, MaxPool1d)]
        layers = list(zip(layers[0::2], layers[1::2]))
        dim = input_dim
        self.logger.debug(f"Input dim = {dim}")

        for conv, maxpool in layers:
            dim = self._calculate_next_layer_size(dim, conv.kernel_size[0], conv.padding[0], conv.stride[0],
                                                  maxpool.kernel_size, maxpool.padding, maxpool.stride)
        
        dim *= self.ecg_channels
        self.logger.debug(f"Last layer dim = {dim}")
        return dim

    def _init_weights(self):
        xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x):
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

