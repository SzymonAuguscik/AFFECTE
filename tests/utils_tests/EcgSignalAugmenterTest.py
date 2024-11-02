from src.exceptions.InvalidAugmentationStrategyAndModeCombinationException import InvalidAugmentationStrategyAndModeCombinationException
from src.exceptions.UnknownAugmentationStrategyException import UnknownAugmentationStrategyException
from src.exceptions.UnknownAugmentationModeException import UnknownAugmentationModeException
from src.utils.EcgSignalAugmenter import EcgSignalAugmenter
from unittest.mock import patch, mock_open
from tests import UnitTest
from typing import List

import torch


class EcgSignalAugmenterTest(UnitTest):
    FILEPATH: str = "/path/to/augmentation/config/file.txt"
    X: List[torch.Tensor] = [torch.Tensor([[[9, 3], [4, 10]],  [[4, 11], [19, 7]], [[9, 23], [3, 12]]]),
                             torch.Tensor([[[2, 8], [15, 19]], [[9, 2],  [20, 9]], [[8, 12], [8, 17]]]),
                             torch.Tensor([[[9, 5], [6, 13]],  [[3, 10], [3, 8]],  [[14, 8], [10, 12]]]),
                             torch.Tensor([[[1, 4], [5, 13]],  [[14, 6], [7, 4]],  [[17, 6], [4, 16]]])]

    # incorrect config
    INCORRECT_AUGMENTATION_MODE_IN_CONFIG_FILE: str = "NOISE 0 1 UNKNOWN_MODE"
    INCORRECT_AUGMENTATION_STRATEGY_IN_CONFIG_FILE: str = "UNKNOWN_STRATEGY 0 MODIFY"

    # valid mode/strategy combinations
    APPEND_ADD_AUGMENTATION_CONFIG_FILE: str = "ADD 0 1 APPEND"
    X_APPEND_ADD: List[torch.Tensor] = [torch.Tensor([[[9, 3], [4, 10],  [13, 13]],
                                                      [[4, 11], [19, 7], [23, 18]],
                                                      [[9, 23], [3, 12],  [12, 35]]]),
                                        torch.Tensor([[[2, 8], [15, 19], [17, 27]],
                                                      [[9, 2],  [20, 9], [29, 11]],
                                                      [[8, 12], [8, 17],  [16, 29]]]),
                                        torch.Tensor([[[9, 5], [6, 13],  [15, 18]],
                                                      [[3, 10], [3, 8],  [6, 18]],
                                                      [[14, 8], [10, 12], [24, 20]]]),
                                        torch.Tensor([[[1, 4], [5, 13],  [6, 17]],
                                                      [[14, 6], [7, 4],  [21, 10]],
                                                      [[17, 6], [4, 16],  [21, 22]]])]

    APPEND_NOISE_AUGMENTATION_CONFIG_FILE: str = "NOISE 0 1 APPEND"
    X_APPEND_NOISE: List[torch.Tensor] = [torch.Tensor([[[9.11, 4.31],  [3.25, 10.05]],
                                                        [[4.08, 19.12], [11.11, 7.26]],
                                                        [[9.2, 3.19],   [23.1, 12.15]]]),
                                          torch.Tensor([[[2.07, 15.19], [8.12, 19.29]],
                                                        [[9.03, 2.25],  [20.14, 9.09]],
                                                        [[8.12, 12.34], [8.25, 17.06]]]),
                                          torch.Tensor([[[9.24, 6.3],   [5.24, 13.13]],
                                                        [[3.01, 10.25], [3.18, 8.16]],
                                                        [[14.05, 8.09], [10.13, 12.16]]]),
                                          torch.Tensor([[[1.19, 5.24],  [4.05, 13.06]],
                                                        [[14.02, 6.17], [7.08, 4.17]],
                                                        [[17.06, 6.21], [4.16, 16.26]]])]

    MODIFY_NOISE_AUGMENTATION_CONFIG_FILE: str = "NOISE 1 MODIFY"
    X_MODIFY_NOISE: List[torch.Tensor] = [torch.Tensor([[[3.12, 10.21]],
                                                        [[11.16, 7.09]],
                                                        [[23.05, 12.1]]]),
                                          torch.Tensor([[[8.12, 19.2]],
                                                        [[20.05, 9.23]],
                                                        [[8.18, 17.31]]]),
                                          torch.Tensor([[[5.06, 13.2]],
                                                        [[3.09, 8.23]],
                                                        [[10.13, 12.05]]]),
                                          torch.Tensor([[[4.19, 13.12]],
                                                        [[7.01, 4.16]],
                                                        [[4.04, 16.19]]])]
    
    # invalid mode/strategy combinations
    MODIFY_ADD_AUGMENTATION_CONFIG_FILE: str = "ADD 0 1 MODIFY"

    def setUp(self) -> None:
        self.ecg_signal_augmenter: EcgSignalAugmenter = EcgSignalAugmenter(self.X.copy(), self.FILEPATH)

    @patch("builtins.open", new_callable=mock_open, read_data=INCORRECT_AUGMENTATION_MODE_IN_CONFIG_FILE)
    def test_unknown_augmentation_mode(self, open_mock) -> None:
        self.assertRaises(UnknownAugmentationModeException, self.ecg_signal_augmenter.augment)

    @patch("builtins.open", new_callable=mock_open, read_data=INCORRECT_AUGMENTATION_STRATEGY_IN_CONFIG_FILE)
    def test_unknown_augmentation_strategy(self, open_mock) -> None:
        self.assertRaises(UnknownAugmentationStrategyException, self.ecg_signal_augmenter.augment)

    @patch("builtins.open", new_callable=mock_open, read_data=APPEND_ADD_AUGMENTATION_CONFIG_FILE)
    def test_append_add_augmentation(self, open_mock) -> None:
        augmented_data: List[torch.Tensor] = self.ecg_signal_augmenter.augment()
        
        for actual_data, expected_data in zip(self.X_APPEND_ADD, augmented_data):
            self.assertTrue(torch.equal(actual_data, expected_data))

    @patch("builtins.open", new_callable=mock_open, read_data=MODIFY_ADD_AUGMENTATION_CONFIG_FILE)
    def test_modify_add_augmentation(self, open_mock) -> None:
        self.assertRaises(InvalidAugmentationStrategyAndModeCombinationException, self.ecg_signal_augmenter.augment)

    @patch("builtins.open", new_callable=mock_open, read_data=APPEND_NOISE_AUGMENTATION_CONFIG_FILE)
    @patch("src.utils.EcgSignalAugmenter.add_noise_to_signal", side_effect=[*X_APPEND_NOISE])
    def test_append_noise_augmentation(self, noise_mock, open_mock) -> None:
        augmented_data: List[torch.Tensor] = self.ecg_signal_augmenter.augment()

        for actual_data, expected_data in zip([torch.cat((signal, augmented_signal), dim=1) for signal, augmented_signal in zip(self.X, self.X_NOISE)], augmented_data):
            self.assertTrue(torch.equal(actual_data, expected_data))

    @patch("builtins.open", new_callable=mock_open, read_data=MODIFY_NOISE_AUGMENTATION_CONFIG_FILE)
    @patch("src.utils.EcgSignalAugmenter.add_noise_to_signal", side_effect=[*X_MODIFY_NOISE])
    def test_append_noise_augmentation(self, noise_mock, open_mock) -> None:
        augmented_data: List[torch.Tensor] = self.ecg_signal_augmenter.augment()

        for actual_data, expected_data in zip(self.X_MODIFY_NOISE, augmented_data):
            self.assertTrue(torch.equal(actual_data, expected_data))

