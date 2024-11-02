from src.utils.EcgSignalLoader import EcgSignalLoader
from unittest.mock import patch, mock_open
from tests import UnitTest
from typing import List

import numpy as np

import torch


class EcgSignalLoaderTest(UnitTest):
    DATA_PATH: str = "/path/to/data"
    DATA_DIRNAME: str = "X_seconds"
    SUBJECTS_FILE_CONTENT: str = "1\n2"
    CHANNELS: List[int] = [0, 1]
    SECONDS: int = 5

    EXISTING_FEATURES: List[torch.Tensor] = [torch.Tensor([[[36, 20, 35], [61, 86, 16]],
                                                           [[60, 10, 97], [29, 76, 71]],
                                                           [[85, 41, 57], [46, 78, 87]],
                                                           [[11, 71, 97], [37, 57, 19]]]),
                                             torch.Tensor([[[98, 17, 56], [25, 57, 16]],
                                                           [[45, 87, 10], [28, 17, 59]],
                                                           [[35, 67, 26], [76, 18, 12]],
                                                           [[17, 87, 15], [75, 16, 81]]])]
    
    EXISTING_LABELS: List[torch.Tensor] = [torch.Tensor([1, 0, 0, 1]),
                                           torch.Tensor([0, 1, 0, 1])]

    FIRST_SUBJECT_FEATURES: List[np.ndarray] = [np.array([[19, 28, 17, 86],
                                                          [58, 33, 97, 27]]),
                                                np.array([[72, 61, 86, 17],
                                                          [58, 12, 96, 61]]),
                                                np.array([[91, 85, 75, 46],
                                                          [35, 17, 20, 18]])]
    FIRST_SUBJECT_LABELS: List[int] = [0, 1, 1]
    SECOND_SUBJECT_FEATURES: List[np.ndarray] = [np.array([[47, 71, 86, 19],
                                                           [26, 81, 36, 69]]),
                                                 np.array([[16, 87, 56, 28],
                                                           [39, 26, 47, 71]])]
    SECOND_SUBJECT_LABELS: List[int] = [1, 0]

    @patch("builtins.open", new_callable=mock_open, read_data=SUBJECTS_FILE_CONTENT)
    @patch("src.utils.TensorManager")
    def setUp(self, tensor_manager_mock, open_mock) -> None:
        self.ecg_signal_loader: EcgSignalLoader = EcgSignalLoader(self.DATA_PATH)
        self.ecg_signal_loader._tensor_manager = tensor_manager_mock
        self.X: List[torch.Tensor]
        self.y: List[torch.Tensor]

    @patch("os.path.exists", side_effect=[True])
    def test_dataset_already_prepared(self, exists_mock) -> None:
        self.ecg_signal_loader._tensor_manager.load.side_effect = [self.EXISTING_FEATURES, self.EXISTING_LABELS]
        self.X, self.y = self.ecg_signal_loader.prepare_dataset(self.CHANNELS, self.SECONDS)
        self.assertEqual(self.X, self.EXISTING_FEATURES)
        self.assertEqual(self.y, self.EXISTING_LABELS)

    @patch("src.utils.EcgSignalLoader.EcgSignalLoader._create_data_from_subject", side_effect=[(FIRST_SUBJECT_FEATURES, FIRST_SUBJECT_LABELS),
                                                                               (SECOND_SUBJECT_FEATURES, SECOND_SUBJECT_LABELS)])
    @patch("builtins.open", new_callable=mock_open, read_data=SUBJECTS_FILE_CONTENT)
    @patch("os.path.exists", side_effect=[False])
    @patch("os.mkdir")
    def test_create_new_dataset(self, mkdir_mock, exists_mock, open_mock, create_mock) -> None:
        self.X, self.y = self.ecg_signal_loader.prepare_dataset(self.CHANNELS, self.SECONDS)
        mkdir_mock.assert_called_once()
        self.ecg_signal_loader._tensor_manager.save.assert_called()

        for actual_data, expected_data in zip(self.X, [self.FIRST_SUBJECT_FEATURES, self.SECOND_SUBJECT_FEATURES]):
            self.assertTrue(torch.equal(actual_data, torch.Tensor(expected_data)[:, :, self.CHANNELS].permute(0, 2, 1)))

        for actual_data, expected_data in zip(self.y, [self.FIRST_SUBJECT_LABELS, self.SECOND_SUBJECT_LABELS]):
            self.assertTrue(torch.equal(actual_data, torch.Tensor(expected_data).reshape(-1, 1)))

