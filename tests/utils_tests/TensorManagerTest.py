from src.utils.TensorManager import TensorManager
from unittest.mock import patch, MagicMock
from tests import UnitTest


class TensorManagerTest(UnitTest):
    TENSOR_PATH: str = "/path/to/tensor"

    def setUp(self) -> None:
        self.tensor_manager: TensorManager = TensorManager()

    @patch("torch.load")
    def test_load(self, torch_load) -> None:
        tensor_mock: MagicMock = MagicMock()
        torch_load.return_value = tensor_mock
        self.assertEqual(self.tensor_manager.load(self.TENSOR_PATH), tensor_mock)

    @patch("torch.save")
    def test_save(self, torch_save) -> None:
        tensor_mock: MagicMock = MagicMock()
        self.tensor_manager.save(tensor_mock, self.TENSOR_PATH)
        torch_save.assert_called_once_with(tensor_mock, self.TENSOR_PATH)

