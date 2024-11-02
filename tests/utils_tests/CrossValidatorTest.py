from src.utils.CrossValidator import CrossValidator
from unittest.mock import patch
from operator import itemgetter
from typing import List, Any
from tests import UnitTest

import torch


def get_multiple_indices(array: List[Any], indices: List[int]) -> List[Any]:
    return itemgetter(*indices)(array)


class CrossValidatorTest(UnitTest):
    X: List[torch.Tensor] = [torch.rand(2, 3) for _ in range(10)]
    Y: List[torch.Tensor] = [torch.randint(0, 2, (1,)) for _ in range(10)]
    SHUFFLED_INDICES: List[int] = [7, 4, 5, 2, 9, 0, 1, 6, 3, 8]
    TRAIN_INDICES: List[int] = [6, 1, 7, 0, 4, 2, 9, 5]
    TEST_INDICES: List[int] = [8, 3]
    VALIDATION_STEP: int = 0

    @patch("sklearn.utils.shuffle", side_effect=[(get_multiple_indices(X, SHUFFLED_INDICES), get_multiple_indices(Y, SHUFFLED_INDICES))])
    def setUp(self, shuffle_mock) -> None:
        self.cross_validator: CrossValidator = CrossValidator(self.X, self.Y)

    @patch("sklearn.utils.shuffle", side_effect=[(get_multiple_indices(X, TRAIN_INDICES), get_multiple_indices(Y, TRAIN_INDICES)),
                                                 (get_multiple_indices(X, TEST_INDICES), get_multiple_indices(Y, TEST_INDICES))])
    def test_folds_preparing(self, shuffle_mock) -> None:
        X_train: torch.Tensor
        y_train: torch.Tensor
        X_test: torch.Tensor
        y_test: torch.Tensor
        X_train, y_train, X_test, y_test = self.cross_validator.prepare_fold(self.cross_validator[self.VALIDATION_STEP])
        self.assertEqual(X_train, get_multiple_indices(self.X, self.TRAIN_INDICES))
        self.assertEqual(y_train, get_multiple_indices(self.Y, self.TRAIN_INDICES))
        self.assertEqual(X_test, get_multiple_indices(self.X, self.TEST_INDICES))
        self.assertEqual(y_test, get_multiple_indices(self.Y, self.TEST_INDICES))

