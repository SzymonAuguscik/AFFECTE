from typing import Optional, Iterator, Dict, List, Tuple
from sklearn.utils import shuffle
from constants import CV

import logging
import torch


class CrossValidator:
    def __init__(self, X: List[torch.Tensor], y: List[torch.Tensor], dataset_custom_size: Optional[int] = None, k: int = 5):
        self.X: List[torch.Tensor] = X
        self.y: List[torch.Tensor] = y
        self.dataset_custom_size: Optional[int] = dataset_custom_size
        self.k: int = k
        self.test_size_coef: float = 1.0 / self.k
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.folds = self._cross_validation_split()

    def __iter__(self) -> Iterator[List[Dict[str, List[torch.Tensor]]]]:
        return iter(self.folds)

    def __getitem__(self, number: int) -> Dict[str, List[torch.Tensor]]:
        return self.folds[number]

    def _cross_validation_split(self) -> List[Dict[str, List[torch.Tensor]]]:
        X: List[torch.Tensor]
        y: List[torch.Tensor]
        X, y = shuffle(self.X, self.y)

        step: int = len(X) // self.k
        remainder: int = len(y) % self.k
        self.logger.debug(f"Remainder: {remainder}")

        left_pivot: int = 0
        right_pivot: int
        folds: List[Dict[str, List[torch.Tensor]]] = []

        for i in range(self.k):
            right_pivot = left_pivot + step
            self.logger.debug(f"Interval: ({left_pivot}, {right_pivot})")
            
            if i < remainder:
                right_pivot += 1

            fold: Dict[str, List[torch.Tensor]] = {
                CV.X_TRAIN : X[:left_pivot] + X[right_pivot:],
                CV.Y_TRAIN : y[:left_pivot] + y[right_pivot:],
                CV.X_TEST  : X[left_pivot : right_pivot],
                CV.Y_TEST  : y[left_pivot : right_pivot]
            }
            folds.append(fold)
            
            left_pivot = right_pivot
        
        return folds

    def prepare_fold(self, fold) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO potential bug
        X_train: torch.Tensor = torch.cat(fold[CV.X_TRAIN])
        y_train: torch.Tensor = torch.cat(fold[CV.Y_TRAIN])
        X_test: torch.Tensor = torch.cat(fold[CV.X_TEST])
        y_test: torch.Tensor = torch.cat(fold[CV.Y_TEST])

        dataset_size: int = y_train.size(0) + y_test.size(0)
        size: int = self.dataset_custom_size if self.dataset_custom_size else dataset_size
        self.logger.debug(f"Size = {size}")

        train_size: int = min(round((1 - self.test_size_coef) * size), 2 * (y_train == 0).size(0), 2 * (y_train == 1).size(0))
        test_size: int = min(round(self.test_size_coef * size), 2 * (y_test == 0).size(0), 2 * (y_test == 1).size(0))
        
        self.logger.debug(f"Test size: {test_size}")
        self.logger.debug(f"Train size: {train_size}")
        
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        X_train, y_train = X_train[:train_size], y_train[:train_size]
        X_test, y_test = X_test[:test_size], y_test[:test_size]
        
        arr_train: float
        arr_test: float
        arr_train, arr_test = sum(y_train) / len(y_train), sum(y_test) / len(y_test)
        self.logger.debug(f"Arrhytmia fraction for train set = {arr_train}")
        self.logger.debug(f"Arrhytmia fraction for test set = {arr_test}")

        return X_train, y_train, X_test, y_test

