from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.LinkConstraintsLoss import LinkConstraintsLoss
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


class Learner:
    """
    Learner is a wrapper for training and testing model. It provides the trainable model with the typical learning hyperparameters (like batch size
    or learning rate). Besides training and testing purposes, it enables to measure the time of the training or store, save and visualize training/test results.

    Parameters
    ----------
    _device : torch.device
        The device to perform calculation on.
    _model : torch.nn.Module
        The model to be trained and tested.
    _X_train : torch.Tensor
        The features for the training.
    _y_train : torch.Tensor
        The labels for the training.
    _X_test : torch.Tensor
        The features for the testing.
    _y_test : torch.Tensor
        The labels for the testing.
    _seconds : int
        Used for the saving results purposes.
    _lr : float
        The learning rate used during the training.
    _batch_size : int
        The batch size of the data during the training.
    _entropy_loss : torch.nn.Module
        The entropy loss for the classification purposes.
    _lc_loss : torch.nn.Module
        The link constraints loss for the regularization purposes.
    _optimizer : torch.optim.Optimizer
        The optimizer determining the model learning method.
    _epochs : int
        The number of the epochs. In other words, the training duration.
    _visualizer : Visualizer
        The visualizer tool for plotting learning curves and train loss.
    _timer : Timer
        The timer used for the training time measurements.
    _logger : logging.Logger
        Used for logging purposes.
    _results : List[Dict[str, float]]
        The results of the model, gathered after testing it.
    _hyperparameters : Dict[str, float]
        All hyperparameters of the Learner: model's hyperparameters and seconds, batch size, learning rate, and epochs.

    Examples
    --------
    loader = EcgSignalLoader("/path/to/data")
    seconds = 10
    Xs, ys = loader.prepare_dataset(channels=[0, 2, 3], seconds=seconds)
    k = 10
    cv = CrossValidator(X=Xs, y=ys, k=k)
    for step, fold in enumerate(cv):
        model = MyCustomModel() # exemplary torch.nn.Module
        X_train, y_train, X_test, y_test = cv.prepare_fold(fold)
        learner = Learner(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          seconds=seconds, lr=0.001, batch_size=128, epochs=150)
        learner.train()
        learner.test(X_test, y_test)
        learner.save_results(step)

    """
    def __init__(self, model: torch.nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor,
                 seconds: int, lr: float, batch_size: int, epochs: int) -> None:
        """
        Initiate Learner with given model and data (features and labels for training/test) and common learning hyperparameters.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained and tested.
        X_train : torch.Tensor
            The features for the training.
        y_train : torch.Tensor
            The labels for the training.
        X_test : torch.Tensor
            The features for the testing.
        y_test : torch.Tensor
            The labels for the testing.
        seconds : int
            Used for the saving results purposes.
        lr : float
            The learning rate used during the training.
        batch_size : int
            The batch size of the data during the training.
        epochs : int
            The number of the epochs. In other words, the training duration.

        """
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
        """
        Make predictions for given features. For efficiency, predictions are made in batches.

        Parameters
        ----------
        X : torch.Tensor
            Features to be used by _model for predictions.

        Returns
        -------
        y_pred : np.ndarray
            Predictions returned by _model.

        """
        test_data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(X, batch_size=self._batch_size)
        y_pred: np.ndarray = np.array([])

        for x in test_data_loader:
            y_pred = np.append(y_pred, self._model(x).round().cpu().detach().numpy())

        y_pred = y_pred.reshape(-1, 1)
        return y_pred

    def _get_link_coefficients(self, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Create link coefficients based on label vectors. If vectors are from the same class, the link between them should be made
        (link coefficient is 1). Otherwise, no link is created (link coefficient is 0).

        Parameters
        ----------
        y_batch : torch.Tensor
            The labels which the link coefficients should be retrieved from.

        Returns
        -------
        coefficients : torch.Tensor
            The coefficients indicating whether there are links between vectors.
        """
        size: int = y_batch.size(0)
        coefficients: torch.Tensor = torch.zeros(size, size).to(self._device)

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                coefficients[i, j] = 1 if y_batch[i, 0] == y_batch[j, 0] else 0

        return coefficients

    def _build_dirname(self) -> str:
        """
        Create a result directory name based on the model name, hyperparameters names and their values.
        It is used for saving results and plots.

        Returns
        -------
        str
            The path to the result directory.
        """
        dirname: str = reduce(lambda directory_name, parameter: f"{directory_name}_{parameter[0]}{parameter[1]}",
                              self._hyperparameters.items(),
                              self._model.name)
        return os.path.join(Paths.Directories.RESULTS, dirname)

    def _save_selected_metrics(self, file: TextIO, *metrics: str) -> None:
        """
        Save selected metrics to the specified file.

        Parameters
        ----------
        file : TextIO
            The file that the results will be saved to.
        *metrics : str
            The metrics to be saved.

        """
        self._logger.debug(f"Metrics to be saved: {metrics}")
        header: str = ",".join(metrics) + '\n'
        file.write(header)

        for result in self._results:
            metrics_row: str = reduce(lambda metric1, metric2: f"{metric1},{metric2}",
                               map(lambda metric: result[metric], metrics)) \
                               + '\n'
            file.write(metrics_row)

    def _save_metrics(self, file: TextIO) -> None:
        """
        Save metrics to the specified file. There are 8 metrics saved: TN, FP, FN, TP, accuracy, F1 score, precision, recall, and specificity.

        Parameters
        ----------
        file : TextIO
            The file that the results will be saved to.

        """
        self._logger.debug(f"Saving metrics to {file}")
        self._save_selected_metrics(file, Results.Metrics.TN, Results.Metrics.FP, Results.Metrics.FN, Results.Metrics.TP)
        self._save_selected_metrics(file, Results.Metrics.ACCURACY, Results.Metrics.F1_SCORE, Results.Metrics.PRECISION, Results.Metrics.RECALL, Results.Metrics.SPECIFICITY)

    def _save_training_time(self, file: TextIO) -> None:
        """
        Save training time to the specified file. The time is rounded to the seconds.

        Parameters
        ----------
        file : TextIO
            The file that the time will be saved to.

        """
        self._logger.debug(f"Saving training time to {file}")
        file.write(f"{round(self._timer.get_time())}s\n")

    def _save_device_name(self, file: TextIO) -> None:
        """
        Save device name to the specified file.

        Parameters
        ----------
        file : TextIO
            The file that the device name will be saved to.

        """
        self._logger.debug(f"Saving device name to {file}")
        file.write(f"{self._device}\n")

    def _plot_metrics(self, dirname: str) -> None:
        """
        Plot metrics gathered throughout the training and save the plots into specified directory.
        The plotted metrics are: training loss and training/test accuracy, F1 score, precision, recall, and specificity.

        Parameters
        ----------
        dirname : str
            The path to the directory where plots should be saved to.

        """
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
        self._visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.SPECIFICITY),\
                                              Results.Metrics.TRAIN_SPECIFICITY,\
                                              Results.Metrics.TEST_SPECIFICITY)

    def train(self) -> None:
        """
        Train _model. The data is split into batches, then the model is repeatedly fed with those batches.
        During the training part, for each epoch, the loss is calculated. Its value depends on the embedded attribute of a model
        (which is not None when model contains Transformer). When it is available, both entropy loss and link constraint loss are taken into account.
        Otherwise, link constraint one is skipped. For every batch the gradients are calculated based on the loss which are then used by optimizer.
        During the evaluation part, the model makes predictions and is evaluated with 4 metrics: F1 score, accuracy, precision, recall, and specificity.
        referring to the training and test features/labels separately. The visualizer saves those metrics for further plotting purposes.

        """
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
            train_specificity: float = recall_score(y_train_true, y_train_pred, pos_label=0)

            self._logger.info(f"Accuracy score = {train_accuracy}")
            self._logger.debug(f"F1 score = {train_f1_score}")
            self._logger.debug(f"Precision score = {train_precision}")
            self._logger.debug(f"Recall score = {train_recall}")
            self._logger.debug(f"Specificity score = {train_specificity}")

            test_accuracy: float = accuracy_score(y_test_true, y_test_pred)
            test_f1_score: float = f1_score(y_test_true, y_test_pred)
            test_precision: float = precision_score(y_test_true, y_test_pred)
            test_recall: float = recall_score(y_test_true, y_test_pred)
            test_specificity: float = recall_score(y_test_true, y_test_pred, pos_label=0)

            self._logger.info(f"Accuracy score = {test_accuracy}")
            self._logger.debug(f"F1 score = {test_f1_score}")
            self._logger.debug(f"Precision score = {test_precision}")
            self._logger.debug(f"Recall score = {test_recall}")
            self._logger.debug(f"Specificity score = {test_specificity}")
            
            self._visualizer.update_metric(Results.Metrics.LOSS, loss.item())

            self._visualizer.update_metric(Results.Metrics.TRAIN_ACCURACY, train_accuracy)
            self._visualizer.update_metric(Results.Metrics.TRAIN_F1_SCORE, train_f1_score)
            self._visualizer.update_metric(Results.Metrics.TRAIN_PRECISION, train_precision)
            self._visualizer.update_metric(Results.Metrics.TRAIN_RECALL, train_recall)
            self._visualizer.update_metric(Results.Metrics.TRAIN_SPECIFICITY, train_specificity)

            self._visualizer.update_metric(Results.Metrics.TEST_ACCURACY, test_accuracy)
            self._visualizer.update_metric(Results.Metrics.TEST_F1_SCORE, test_f1_score)
            self._visualizer.update_metric(Results.Metrics.TEST_PRECISION, test_precision)
            self._visualizer.update_metric(Results.Metrics.TEST_RECALL, test_recall)
            self._visualizer.update_metric(Results.Metrics.TEST_SPECIFICITY, test_specificity)
        
        self._logger.debug("Stopping timer...")
        self._timer.stop()
        training_time: float = round(self._timer.get_time())
        self._logger.debug(f"Training time was {training_time}s")

    def test(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Test model with given features and expected labels. After setting model into evaluation mode, the model makes predictions
        which are then evaluated with 8 metrics: TN, FP, FN, TP, F1 score, accuracy, precision, recall, and specificity.
        Those results are then stored.

        Parameters
        ----------
        X : torch.Tensor
            The features which model will make predictions from.
        y : torch.Tensor
            The expected labels.

        """
        X = X.to(self._device)
        y = y.to(self._device)
        self._logger.info("Testing model...")
        self._model.eval()
        y_pred: np.ndarray = self._make_predictions(X)
        y_true: np.ndarray = y.cpu().detach().numpy()

        self._logger.info(f"{int(sum(y)[0])}/{y.shape[0]} samples from test set are AF")
        self._logger.info(f"{int(sum(y_pred)[0])}/{y_pred.shape[0]} samples marked as AF")

        tn: int
        fp: int
        fn: int
        tp: int
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy: float = accuracy_score(y_true, y_pred)
        f1_metric: float = f1_score(y_true, y_pred)
        precision: float = precision_score(y_true, y_pred)
        recall: float = recall_score(y_true, y_pred)
        specificity: float = recall_score(y_true, y_pred, pos_label=0)

        self._logger.debug(f"TN: {tn}")
        self._logger.debug(f"FP: {fp}")
        self._logger.debug(f"FN: {fn}")
        self._logger.debug(f"TP: {tp}")
        self._logger.debug(f"Accuracy: {accuracy}")
        self._logger.debug(f"F1 score: {f1_metric}")
        self._logger.debug(f"Precision: {precision}")
        self._logger.debug(f"Recall: {recall}")
        self._logger.debug(f"Specificity: {specificity}")

        results: Dict[str, float] = {
            Results.Metrics.TN          : tn,
            Results.Metrics.FP          : fp,
            Results.Metrics.FN          : fn,
            Results.Metrics.TP          : tp,
            Results.Metrics.ACCURACY    : accuracy,
            Results.Metrics.F1_SCORE    : f1_metric,
            Results.Metrics.PRECISION   : precision,
            Results.Metrics.RECALL      : recall,
            Results.Metrics.SPECIFICITY : specificity,
        }
        self._results.append(results)

    def save_results(self, fold: int) -> None:
        """
        Save results for the specified cross validation fold. Saved results are: metrics, training time and device name.
        It also creates necessary directories if needed.

        Parameters
        ----------
        fold : int
            The fold number which results are referring to. It also indicates the directory to save results.

        """
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

