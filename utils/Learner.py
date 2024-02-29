from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
from constants import Results, Paths, Hyperparameters
from typing import Dict, List, TextIO, Optional
from utils.Visualizer import Visualizer
from utils.Timer import Timer
from functools import reduce

import numpy as np

import warnings
import logging
import torch
import os
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class LinkConstraintsLoss(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super(LinkConstraintsLoss, self).__init__()
        self._device: torch.device = device

    def forward(self, beta: torch.Tensor, e: np.ndarray) -> float:
        i: torch.Tensor = torch.arange(beta.size(0)).reshape(-1, 1).to(self._device)
        j: torch.Tensor = torch.arange(beta.size(0)).to(self._device)
        loss: float = 0.5 * torch.norm(beta[i, 0] - e[i, j] * beta[j, 0], p=2)
        return loss

class Learner:
    def __init__(self, model: torch.nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor,
                 seconds: int, lr: float, batch_size: int, epochs: int) -> None:
        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module = model.to(self._device)
        self._X_train: torch.Tensor = X_train.to(self._device)
        self._y_train: torch.Tensor = y_train.to(self._device)
        self._X_test: torch.Tensor = X_test.to(self._device)
        self._y_test: torch.Tensor = y_test.to(self._device)
        self._seconds: int = seconds
        self._lr: float = lr
        self._batch_size: int = batch_size
        self._entropy_loss: torch.nn.Module = torch.nn.BCELoss()
        self._lc_loss: torch.nn.Module = LinkConstraintsLoss(self._device)
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._epochs: int = epochs
        self._visualizer: Visualizer = Visualizer()
        self._timer: Timer = Timer()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._results: List[Dict[str, float]] = []
        self._hyperparameters: Dict[str, float] = {
            Hyperparameters.Names.SECONDS       : self._seconds,
            Hyperparameters.Names.LEARNING_RATE : self._lr,
            Hyperparameters.Names.BATCH_SIZE    : self._batch_size,
            Hyperparameters.Names.EPOCHS        : self._epochs,
            **self._model.hyperparameters,
        }
        self._logger.debug(f"Layers: {len([param for param in model.parameters()])}")
        self._logger.debug(f"Params: {sum(param.numel() for param in model.parameters())}")

    def _make_predictions(self, X: torch.Tensor) -> np.ndarray:
        test_data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(X, batch_size=self._batch_size)
        y_pred: np.ndarray = np.array([])

        for x in test_data_loader:
            y_pred = np.append(y_pred, self._model(x).round().cpu().detach().numpy())

        y_pred = y_pred.reshape(-1, 1)
        return y_pred

    def _get_link_coefficients(self, y_batch: torch.Tensor) -> torch.Tensor:
        size: int = y_batch.size(0)
        coefficients: torch.Tensor = torch.zeros(size, size).to(self._device)

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                coefficients[i, j] = 1 if y_batch[i, 0] == y_batch[j, 0] else 0

        return coefficients

    def _build_dirname(self) -> str:
        dirname: str = reduce(lambda directory_name, parameter: f"{directory_name}_{parameter[0]}{parameter[1]}",
                              self._hyperparameters.items(),
                              self._model.name)
        return os.path.join(Paths.Directories.RESULTS, dirname)

    def _save_metrics(self, file: TextIO) -> None:
        self._logger.debug(f"Saving metrics to {file}")
        file.write(f"{Results.Metrics.ACCURACY},{Results.Metrics.F1_SCORE},{Results.Metrics.PRECISION},{Results.Metrics.RECALL}\n")

        for result in self._results:
            file.write(f"{result[Results.Metrics.ACCURACY]},"
                       f"{result[Results.Metrics.F1_SCORE]},"
                       f"{result[Results.Metrics.PRECISION]},"
                       f"{result[Results.Metrics.RECALL]}\n")

    def _save_training_time(self, file: TextIO) -> None:
        self._logger.debug(f"Saving training time to {file}")
        file.write(f"{round(self._timer.get_time())}s\n")

    def _save_device_name(self, file: TextIO) -> None:
        self._logger.debug(f"Saving device name to {file}")
        file.write(f"{self._device}\n")

    def _plot_metrics(self, dirname: str) -> None:
        self._logger.debug("Creating plots...")
        self._visualizer.plot_train_loss(os.path.join(dirname, Results.Visualization.Files.LOSS))
        self._visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.ACCURACY),\
                                             Results.Metrics.TRAIN_ACCURACY,\
                                             Results.Metrics.TEST_ACCURACY)
        self._visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.F1_SCORE),\
                                             Results.Metrics.TRAIN_F1_SCORE,\
                                             Results.Metrics.TEST_F1_SCORE)
        self._visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.PRECISION),\
                                             Results.Metrics.TRAIN_PRECISION,\
                                             Results.Metrics.TEST_PRECISION)
        self._visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.RECALL),\
                                             Results.Metrics.TRAIN_RECALL,\
                                             Results.Metrics.TEST_RECALL)

    def train(self) -> None:
        self._logger.info("Training model...")
        self._logger.debug(f"Using {self._device} device")
        self._logger.debug("Starting timer...")
        self._timer.start()
        train_data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(list(zip(self._X_train, self._y_train)), shuffle=True, batch_size=self._batch_size)
        
        for epoch in range(self._epochs):
            self._model.train()
            self._logger.info(f"Epoch no. {epoch + 1}")

            loss: float = 0
            for (x, y) in train_data_loader:
                pred: torch.Tensor = self._model(x)
                self._logger.debug("Calculating loss...")
                loss = self._entropy_loss(pred, y)

                embedded: Optional[torch.Tensor] = self._model.embedded
                if embedded is not None:
                    link_coefficients: np.ndarray = self._get_link_coefficients(y)
                    loss += 1e-8 * self._lc_loss(embedded, link_coefficients)
                
                self._logger.debug("Updating weights...")
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            self._model.eval()
            y_train_true: np.ndarray = self._y_train.cpu().detach().numpy()
            y_train_pred: np.ndarray = self._make_predictions(self._X_train)
            y_test_true: np.ndarray = self._y_test.cpu().detach().numpy()
            y_test_pred: np.ndarray = self._make_predictions(self._X_test)

            self._logger.info(f"Loss = {loss.item()}")

            train_accuracy: float = accuracy_score(y_train_true, y_train_pred)
            train_f1_score: float = f1_score(y_train_true, y_train_pred)
            train_precision: float = precision_score(y_train_true, y_train_pred)
            train_recall: float = recall_score(y_train_true, y_train_pred)

            self._logger.debug(f"Accuracy score = {train_accuracy}")
            self._logger.debug(f"F1 score = {train_f1_score}")
            self._logger.debug(f"Precision score = {train_precision}")
            self._logger.info(f"Recall score = {train_recall}")

            test_accuracy: float = accuracy_score(y_test_true, y_test_pred)
            test_f1_score: float = f1_score(y_test_true, y_test_pred)
            test_precision: float = precision_score(y_test_true, y_test_pred)
            test_recall: float = recall_score(y_test_true, y_test_pred)

            self._logger.debug(f"Accuracy score = {test_accuracy}")
            self._logger.debug(f"F1 score = {test_f1_score}")
            self._logger.debug(f"Precision score = {test_precision}")
            self._logger.info(f"Recall score = {test_recall}")
            
            self._visualizer.update_metric(Results.Metrics.LOSS, loss.item())

            self._visualizer.update_metric(Results.Metrics.TRAIN_ACCURACY, train_accuracy)
            self._visualizer.update_metric(Results.Metrics.TRAIN_F1_SCORE, train_f1_score)
            self._visualizer.update_metric(Results.Metrics.TRAIN_PRECISION, train_precision)
            self._visualizer.update_metric(Results.Metrics.TRAIN_RECALL, train_recall)

            self._visualizer.update_metric(Results.Metrics.TEST_ACCURACY, test_accuracy)
            self._visualizer.update_metric(Results.Metrics.TEST_F1_SCORE, test_f1_score)
            self._visualizer.update_metric(Results.Metrics.TEST_PRECISION, test_precision)
            self._visualizer.update_metric(Results.Metrics.TEST_RECALL, test_recall)
        
        self._logger.debug("Stopping timer...")
        self._timer.stop()
        training_time: float = round(self._timer.get_time())
        self._logger.debug(f"Training time was {training_time}s")

    def test(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self._device)
        y = y.to(self._device)
        self._logger.info("Testing model...")
        self._model.eval()
        y_pred: np.ndarray = self._make_predictions(X)
        y_true: np.ndarray = y.cpu().detach().numpy()

        self._logger.info(f"{int(sum(y)[0])}/{y.shape[0]} samples from test set are AF")
        self._logger.info(f"{int(sum(y_pred)[0])}/{y_pred.shape[0]} samples marked as AF")

        accuracy: float = accuracy_score(y_true, y_pred)
        f1_metric: float = f1_score(y_true, y_pred)
        precision: float = precision_score(y_true, y_pred)
        recall: float = recall_score(y_true, y_pred)

        self._logger.debug(f"Accuracy: {accuracy}")
        self._logger.debug(f"F1 score: {f1_metric}")
        self._logger.debug(f"Precision: {precision}")
        self._logger.debug(f"Recall: {recall}")

        results: Dict[str, float] = {
            Results.Metrics.ACCURACY  : accuracy,
            Results.Metrics.F1_SCORE  : f1_metric,
            Results.Metrics.PRECISION : precision,
            Results.Metrics.RECALL    : recall,
        }
        self._results.append(results)

    def save_results(self, fold: str) -> None:
        self._logger.debug("Saving results...")
        if not os.path.exists(Paths.Directories.RESULTS):
            os.mkdir(Paths.Directories.RESULTS)

        dirname: str = self._build_dirname()
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self._logger.debug(f"Saving to {dirname}")

        fold_dirname = os.path.join(dirname, f"fold_{fold}")
        if not os.path.exists(fold_dirname):
            os.mkdir(fold_dirname)

        results_filename: str = os.path.join(fold_dirname, Paths.Files.RESULTS)

        with open(results_filename, 'w') as file:
            self._save_metrics(file)
            self._save_training_time(file)
            self._save_device_name(file)

        self._plot_metrics(fold_dirname)

