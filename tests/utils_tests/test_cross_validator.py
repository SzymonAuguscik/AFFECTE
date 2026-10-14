from operator import itemgetter
from typing import Any
from unittest.mock import patch

import torch

from src.utils.CrossValidator import CrossValidator


def get_multiple_indices(array: list[Any], indices: list[int]) -> list[Any]:
    return itemgetter(*indices)(array)


class TestCrossValidator:
    X: list[torch.Tensor] = [torch.rand(2, 3) for _ in range(10)]
    Y: list[torch.Tensor] = [torch.randint(0, 2, (1,)) for _ in range(10)]
    SHUFFLED_INDICES: list[int] = [7, 4, 5, 2, 9, 0, 1, 6, 3, 8]
    TRAIN_INDICES: list[int] = [6, 1, 7, 0, 4, 2, 9, 5]
    TEST_INDICES: list[int] = [8, 3]
    VALIDATION_STEP: int = 0

    def setup_method(self) -> None:
        with patch(
            "sklearn.utils.shuffle",
            side_effect=[
                (
                    get_multiple_indices(self.X, self.SHUFFLED_INDICES),
                    get_multiple_indices(self.Y, self.SHUFFLED_INDICES),
                )
            ],
        ):
            self.cross_validator: CrossValidator = CrossValidator(self.X, self.Y)

    @patch(
        "sklearn.utils.shuffle",
        side_effect=[
            (
                get_multiple_indices(X, TRAIN_INDICES),
                get_multiple_indices(Y, TRAIN_INDICES),
            ),
            (
                get_multiple_indices(X, TEST_INDICES),
                get_multiple_indices(Y, TEST_INDICES),
            ),
        ],
    )
    def test_folds_preparing(self, shuffle_mock) -> None:
        X_train: torch.Tensor
        y_train: torch.Tensor
        X_test: torch.Tensor
        y_test: torch.Tensor
        X_train, y_train, X_test, y_test = self.cross_validator.prepare_fold(
            self.cross_validator[self.VALIDATION_STEP]
        )
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                X_train, get_multiple_indices(self.X, self.TRAIN_INDICES), strict=True
            )
        )
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                y_train, get_multiple_indices(self.Y, self.TRAIN_INDICES), strict=True
            )
        )
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                X_test, get_multiple_indices(self.X, self.TEST_INDICES), strict=True
            )
        )
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                y_test, get_multiple_indices(self.Y, self.TEST_INDICES), strict=True
            )
        )
