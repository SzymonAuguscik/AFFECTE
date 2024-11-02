from src.models.Cnn import Cnn
from src.constants import Hyperparameters
from tests import UnitTest


class CnnTest(UnitTest):
    CHANNELS: int = 12
    INPUT_DIMENSION: int = 128
    OUTPUT_DIMENSION: int = 16

    def setUp(self) -> None:
        self.cnn: Cnn = Cnn(self.CHANNELS, self.INPUT_DIMENSION, self.OUTPUT_DIMENSION)

    def test_calculate_output_dim(self) -> None:
        kernel_size: int = 24
        padding: int = 3
        stride: int = 2
        output_dim: int = self.cnn._calculate_output_dim(self.INPUT_DIMENSION, kernel_size, padding, stride)
        expected_output_dim: int = 56
        self.assertEqual(output_dim, expected_output_dim)

    def test_calculate_next_layer_size(self) -> None:
        conv_kernel_size: int = 32
        conv_padding: int = 5
        conv_stride: int = 3
        maxpool_kernel_size: int = 16
        maxpool_padding: int = 2
        maxpool_stride: int = 5
        output_dim: int = self.cnn._calculate_next_layer_size(self.INPUT_DIMENSION, conv_kernel_size, conv_padding, conv_stride,
                                                             maxpool_kernel_size, maxpool_padding, maxpool_stride)
        expected_output_dim: int = 5
        self.assertEqual(output_dim, expected_output_dim)

    def test_calculate_last_layer_dim(self) -> None:
        last_layer_dim: int = self.cnn._calculate_last_layer_dim(self.INPUT_DIMENSION)
        expected_last_layer_dim: int = 120
        self.assertEqual(last_layer_dim, expected_last_layer_dim)

    def test_set_input_data_to_minimal_value_if_smaller_is_provided(self) -> None:
        too_small_input_data: int = Hyperparameters.Cnn.MINIMAL_INPUT_DIMENSION - 1
        last_layer_dim: int = self.cnn._calculate_last_layer_dim(too_small_input_data)
        expected_last_layer_dim: int = 12
        self.assertEqual(last_layer_dim, expected_last_layer_dim)

