from utils.TensorManager import TensorManager
from sklearn.utils import shuffle
from constants import CV, Paths

import logging
import torch
import os


class CrossValidator:
    def __init__(self, X, y, dataset_custom_size=None, k=5):
        self.X = X
        self.y = y
        self.dataset_custom_size = dataset_custom_size
        self.k = k
        self.test_size_coef = 1.0 / self.k
        self.logger = logging.getLogger(__name__)
        self.folds = self._cross_validation_split()

    def __iter__(self):
        return iter(self.folds)

    def _cross_validation_split(self):
        X, y = shuffle(self.X, self.y)
        step = len(X) // self.k
        remainder = len(y) % self.k
        self.logger.debug(f"Remainder: {remainder}")

        left_pivot = 0
        folds = []

        for i in range(self.k):
            right_pivot = left_pivot + step
            self.logger.debug(f"Interval: ({left_pivot}, {right_pivot})")
            
            if i < remainder:
                right_pivot += 1

            fold = {
                CV.X_TRAIN : X[:left_pivot] + X[right_pivot:],
                CV.Y_TRAIN : y[:left_pivot] + y[right_pivot:],
                CV.X_TEST  : X[left_pivot : right_pivot],
                CV.Y_TEST  : y[left_pivot : right_pivot]
            }
            folds.append(fold)
            
            left_pivot = right_pivot
        
        return folds

    def prepare_fold(self, fold):
        # TODO potential bug
        X_train, y_train = torch.cat(fold[CV.X_TRAIN]), torch.cat(fold[CV.Y_TRAIN])
        X_test, y_test = torch.cat(fold[CV.X_TEST]), torch.cat(fold[CV.Y_TEST])

        dataset_size = y_train.size(0) + y_test.size(0)
        size = self.dataset_custom_size if self.dataset_custom_size else dataset_size
        self.logger.debug(f"Size = {size}")

        train_size = min(round((1 - self.test_size_coef) * size), 2 * (y_train == 0).size(0), 2 * (y_train == 1).size(0))
        test_size = min(round(self.test_size_coef * size), 2 * (y_test == 0).size(0), 2 * (y_test == 1).size(0))
        
        self.logger.debug(f"Test size: {test_size}")
        self.logger.debug(f"Train size: {train_size}")
        
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        X_train, y_train = X_train[:train_size], y_train[:train_size]
        X_test, y_test = X_test[:test_size], y_test[:test_size]
        # X_train = (torch.cat((X_train[y_train==0][:train_size // 2], X_train[y_train==1][:train_size // 2]))).reshape(train_size, 1, -1)
        # X_test = (torch.cat((X_test[y_test==0][:test_size // 2], X_test[y_test==1][:test_size // 2]))).reshape(test_size, 1, -1)
        # y_train = (torch.cat((y_train[y_train==0][:train_size // 2], y_train[y_train==1][:train_size // 2]))).reshape(-1, 1)
        # y_test = (torch.cat((y_test[y_test==0][:test_size // 2], y_test[y_test==1][:test_size // 2]))).reshape(-1, 1)
        # X_train, y_train = shuffle(X_train[:train_size], y_train[:train_size])
        # X_test, y_test = shuffle(X_test[:test_size], y_test[:test_size])
        
        # self.logger.debug(f"Train features size: {X_train.size()}")
        # self.logger.debug(f"Train labels size: {y_train.size()}")
        # self.logger.debug(f"Test features size: {X_test.size()}")
        # self.logger.debug(f"Test labels size: {y_test.size()}")
        
        arr_train, arr_test = sum(y_train) / len(y_train), sum(y_test) / len(y_test)
        self.logger.debug(f"Arrhytmia fraction for train set = {arr_train}")
        self.logger.debug(f"Arrhytmia fraction for test set = {arr_test}")

        return X_train, y_train, X_test, y_test

