from unittest.mock import MagicMock, patch

from src.utils.TensorManager import TensorManager


class TestTensorManager:
    TENSOR_PATH: str = "/path/to/tensor"

    def setup_method(self) -> None:
        self.tensor_manager: TensorManager = TensorManager()

    @patch("torch.load")
    def test_load(self, torch_load) -> None:
        tensor_mock: MagicMock = MagicMock()
        torch_load.return_value = tensor_mock
        assert self.tensor_manager.load(self.TENSOR_PATH) == tensor_mock

    @patch("torch.save")
    def test_save(self, torch_save) -> None:
        tensor_mock: MagicMock = MagicMock()
        self.tensor_manager.save(tensor_mock, self.TENSOR_PATH)
        torch_save.assert_called_once_with(tensor_mock, self.TENSOR_PATH)
