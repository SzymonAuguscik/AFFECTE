from typing import Optional, Iterator, Dict, List, Tuple
from sklearn.utils import shuffle
from src.constants import CV

import logging
import torch


class CrossValidator:
    """
    CrossValidator is used for preparing dataset split into k folds. The folds are stored and each of them can be accessed separately.
    It supports inter-patient split (all segments from the same source can be either in train or test set).

    Attributes
    ----------
    _X : List[torch.Tensor]
        The list of features from different sources.
    _y : List[torch.Tensor]
        The list of labels from different sources.
    _dataset_custom_size : Optional[int]
        If set, CrossValidator will use only that count of features/labels when extracting folds.
    _k : int
        The number of folds.
    _test_size_coef : float
        It indicates the whole dataset fraction that will serve as a test set. The inverse of _k.
    _logger : logging.Logger
        Used for logging purposes.
    _folds : List[Dict[str, List[torch.Tensor]]]
        The folds prepared by CrossValidator. Each of them stores the type of data (train/test features/labels)
        and their corresponding values.

    Examples
    --------
    <load list of features and labels e.g. from EcgSignalLoader>
    k = 10
    cv = CrossValidator(features, labels, k=k)

    for i in range(k):
        X_train, y_train, X_test, y_test = cv.prepare_fold(fold=i)

    """

    def __init__(self, X: List[torch.Tensor], y: List[torch.Tensor], dataset_custom_size: Optional[int] = None, k: int = 5):
        """
        Initiate CrossValidator with the lists of features and labels and prepare folds by splitting those lists.

        Parameters
        ----------
        X : List[torch.Tensor]
            The list of features from different sources.
        y : List[torch.Tensor]
            The list of labels from different sources.
        dataset_custom_size : Optional[int], optional
            If set, CrossValidator will use only that count of features/labels when extracting folds.
        k : int, optional
            The number of folds.

        """
        self._X: List[torch.Tensor] = X
        self._y: List[torch.Tensor] = y
        self._dataset_custom_size: Optional[int] = dataset_custom_size
        self._k: int = k
        self._test_size_coef: float = 1.0 / self._k
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._folds: List[Dict[str, List[torch.Tensor]]] = self._cross_validation_split()

    def __iter__(self) -> Iterator[List[Dict[str, List[torch.Tensor]]]]:
        """
        Returns the iterator to the next fold.

        Returns
        -------
        Iterator[List[Dict[str, List[torch.Tensor]]]]
            The iterator to the next fold.

        """
        return iter(self._folds)

    def __getitem__(self, number: int) -> Dict[str, List[torch.Tensor]]:
        """
        Returns the fold of given number.

        Parameters
        ----------
        number : int
            The number of the fold to be returned.

        Returns
        -------
        Dict[str, List[torch.Tensor]]
            The specified fold.

        """
        return self._folds[number]

    def _cross_validation_split(self) -> List[Dict[str, List[torch.Tensor]]]:
        """
        Split features and labels with inter-patient method and store such folds.
        It divides the data set into n parts and, based on the fold number,
        extracts the nth part of the dataset and makes it test set. The rest is saved as a train set.

        Returns
        -------
        folds : List[Dict[str, List[torch.Tensor]]]
            The folds created by split.

        """
        X: List[torch.Tensor]
        y: List[torch.Tensor]
        X, y = shuffle(self._X, self._y)

        step: int = len(X) // self._k
        remainder: int = len(y) % self._k
        self._logger.debug(f"Remainder: {remainder}")

        left_pivot: int = 0
        right_pivot: int
        folds: List[Dict[str, List[torch.Tensor]]] = []

        for i in range(self._k):
            right_pivot = left_pivot + step
            self._logger.debug(f"Interval: ({left_pivot}, {right_pivot})")
            
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

    def prepare_fold(self, fold: Dict[str, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare train/test sets of given fold. The preparation relies on determining if custom dataset size should be used
        and setting the train/test sizes with respect to the fewer class examples (therefore, it is more probable
        that the similar number of samples will be chosen). The data is shuffled and then the train and test sets are created.

        Parameters
        ----------
        fold : Dict[str, List[torch.Tensor]]
            The fold that will be prepared.

        Returns
        -------
        X_train : torch.Tensor
            The train features.
        y_train : torch.Tensor
            The train labels.
        X_test : torch.Tensor
            The test features.
        y_test : torch.Tensor
            The test labels.

        """
        # TODO potential bug
        X_train: torch.Tensor = torch.cat(fold[CV.X_TRAIN])
        y_train: torch.Tensor = torch.cat(fold[CV.Y_TRAIN])
        X_test: torch.Tensor = torch.cat(fold[CV.X_TEST])
        y_test: torch.Tensor = torch.cat(fold[CV.Y_TEST])

        dataset_size: int = y_train.size(0) + y_test.size(0)
        size: int = self._dataset_custom_size if self._dataset_custom_size else dataset_size
        self._logger.debug(f"Size = {size}")

        # TODO check and remove min (use only round(...))
        train_size: int = min(round((1 - self._test_size_coef) * size), 2 * (y_train == 0).size(0), 2 * (y_train == 1).size(0))
        test_size: int = min(round(self._test_size_coef * size), 2 * (y_test == 0).size(0), 2 * (y_test == 1).size(0))
        
        self._logger.debug(f"Test size: {test_size}")
        self._logger.debug(f"Train size: {train_size}")
        
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        X_train, y_train = X_train[:train_size], y_train[:train_size]
        X_test, y_test = X_test[:test_size], y_test[:test_size]
        
        arr_train: float
        arr_test: float
        arr_train, arr_test = sum(y_train) / len(y_train), sum(y_test) / len(y_test)
        self._logger.debug(f"Arrhytmia fraction for train set = {arr_train}")
        self._logger.debug(f"Arrhytmia fraction for test set = {arr_test}")

        return X_train, y_train, X_test, y_test

