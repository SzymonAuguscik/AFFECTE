from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
from constants import Results, Paths, Hyperparameters
from torch.utils.data import DataLoader
from utils.Visualizer import Visualizer
from utils.Timer import Timer
from torch.nn import BCELoss, Module
from torch.optim import Adam
from functools import reduce

import numpy as np

import warnings
import logging
import torch
import os
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.autograd.set_detect_anomaly(True)


class LinkConstraintsLoss(Module):
    def __init__(self):
        super(LinkConstraintsLoss, self).__init__()

    def forward(self, beta, e):
        i = torch.arange(beta.size(0)).reshape(-1, 1)
        j = torch.arange(beta.size(0))
        loss = 0.5 * torch.sum(torch.sqrt((beta[i, 0] - e[i, j] * beta[j, 0])**2))
        return loss

class Learner:
    def __init__(self, model, X_train, y_train, X_test, y_test, seconds, lr, batch_size, epochs):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.seconds = seconds
        self.lr = lr
        self.batch_size = batch_size
        self.entropy_loss = BCELoss()
        self.lc_loss = LinkConstraintsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.epochs = epochs
        self.visualizer = Visualizer()
        self.timer = Timer()
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.hyperparameters = {
            Hyperparameters.SECONDS       : self.seconds,
            Hyperparameters.LEARNING_RATE : self.lr,
            Hyperparameters.BATCH_SIZE    : self.batch_size,
            Hyperparameters.EPOCHS        : self.epochs,
            **self.model.get_hyperparameters(),
        }

    def _make_predictions(self, X):
        test_data_loader = DataLoader(X, batch_size=self.batch_size)
        y_pred = np.array([])

        for x in test_data_loader:
            y_pred = np.append(y_pred, self.model(x).round().detach().numpy())

        y_pred = y_pred.reshape(-1, 1)
        return y_pred

    @classmethod
    def _get_link_coefficients(cls, y_batch):
        size = y_batch.size(0)
        coefficients = torch.zeros(size, size)

        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                coefficients[i, j] = 1 if y_batch[i, 0] == y_batch[j, 0] else 0

        return coefficients

    def _build_dirname(self):
        dirname = reduce(lambda directory_name, parameter: f"{directory_name}_{parameter[0]}{parameter[1]}",
                         self.hyperparameters.items(),
                         self.model.name)
        return os.path.join(Paths.Directories.RESULTS, dirname)

    def _save_metrics(self, file):
        self.logger.debug(f"Saving metrics to {file}")
        file.write(f"{Results.Metrics.ACCURACY},{Results.Metrics.F1_SCORE},{Results.Metrics.PRECISION},{Results.Metrics.RECALL}\n")

        for result in self.results:
            file.write(f"{result[Results.Metrics.ACCURACY]},"
                       f"{result[Results.Metrics.F1_SCORE]},"
                       f"{result[Results.Metrics.PRECISION]},"
                       f"{result[Results.Metrics.RECALL]}\n")

    def _save_training_time(self, file):
        self.logger.debug(f"Saving training time to {file}")
        file.write(f"{round(self.timer.get_time())}s\n")

    def _plot_metrics(self, dirname):
        self.logger.debug("Creating plots...")
        self.visualizer.plot_train_loss(os.path.join(dirname, Results.Visualization.Files.LOSS))
        self.visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.ACCURACY),\
                                             Results.Metrics.TRAIN_ACCURACY,\
                                             Results.Metrics.TEST_ACCURACY)
        self.visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.F1_SCORE),\
                                             Results.Metrics.TRAIN_F1_SCORE,\
                                             Results.Metrics.TEST_F1_SCORE)
        self.visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.PRECISION),\
                                             Results.Metrics.TRAIN_PRECISION,\
                                             Results.Metrics.TEST_PRECISION)
        self.visualizer.plot_learning_curves(os.path.join(dirname, Results.Visualization.Files.RECALL),\
                                             Results.Metrics.TRAIN_RECALL,\
                                             Results.Metrics.TEST_RECALL)

    def train(self):
        self.logger.info("Training model...")
        self.logger.debug("Starting timer...")
        self.timer.start()
        train_data_loader = DataLoader(list(zip(self.X_train, self.y_train)), shuffle=True, batch_size=self.batch_size)
        
        for epoch in range(self.epochs):
            self.model.train()
            self.logger.info(f"Epoch no. {epoch + 1}")

            loss = 0
            for (x, y) in train_data_loader:
                pred = self.model(x)
                self.logger.debug("Calculating loss...")
                loss = self.entropy_loss(pred, y)

                embedded = self.model.get_embedded()
                if embedded is not None:
                    link_coefficients = self._get_link_coefficients(y)
                    loss += 1e-8 * self.lc_loss(embedded, link_coefficients)
                
                self.logger.debug("Updating weights...")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            y_train_true = self.y_train.detach().numpy()
            y_train_pred = self._make_predictions(self.X_train)
            y_test_true = self.y_test.detach().numpy()
            y_test_pred = self._make_predictions(self.X_test)

            self.logger.info(f"Loss = {loss.item()}")

            train_accuracy = accuracy_score(y_train_true, y_train_pred)
            train_f1_score = f1_score(y_train_true, y_train_pred)
            train_precision = precision_score(y_train_true, y_train_pred)
            train_recall = recall_score(y_train_true, y_train_pred)

            self.logger.debug(f"Accuracy score = {train_accuracy}")
            self.logger.debug(f"F1 score = {train_f1_score}")
            self.logger.debug(f"Precision score = {train_precision}")
            self.logger.info(f"Recall score = {train_recall}")

            test_accuracy = accuracy_score(y_test_true, y_test_pred)
            test_f1_score = f1_score(y_test_true, y_test_pred)
            test_precision = precision_score(y_test_true, y_test_pred)
            test_recall = recall_score(y_test_true, y_test_pred)

            self.logger.debug(f"Accuracy score = {test_accuracy}")
            self.logger.debug(f"F1 score = {test_f1_score}")
            self.logger.debug(f"Precision score = {test_precision}")
            self.logger.info(f"Recall score = {test_recall}")
            
            self.visualizer.update_metric(Results.Metrics.LOSS, loss.item())

            self.visualizer.update_metric(Results.Metrics.TRAIN_ACCURACY, train_accuracy)
            self.visualizer.update_metric(Results.Metrics.TRAIN_F1_SCORE, train_f1_score)
            self.visualizer.update_metric(Results.Metrics.TRAIN_PRECISION, train_precision)
            self.visualizer.update_metric(Results.Metrics.TRAIN_RECALL, train_recall)

            self.visualizer.update_metric(Results.Metrics.TEST_ACCURACY, test_accuracy)
            self.visualizer.update_metric(Results.Metrics.TEST_F1_SCORE, test_f1_score)
            self.visualizer.update_metric(Results.Metrics.TEST_PRECISION, test_precision)
            self.visualizer.update_metric(Results.Metrics.TEST_RECALL, test_recall)
        
        self.logger.debug("Stopping timer...")
        self.timer.stop()
        training_time = round(self.timer.get_time())
        self.logger.debug(f"Training time was {training_time}s")

    def test(self, X, y):
        self.logger.info("Testing model...")
        self.model.eval()
        y_pred = self._make_predictions(X)
        y_true = y.detach().numpy()

        self.logger.info(f"{int(sum(y)[0])}/{y.shape[0]} samples from test set are AF")
        self.logger.info(f"{int(sum(y_pred)[0])}/{y_pred.shape[0]} samples marked as AF")

        accuracy = accuracy_score(y_true, y_pred)
        f1_metric = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        self.logger.debug(f"Accuracy: {accuracy}")
        self.logger.debug(f"F1 score: {f1_metric}")
        self.logger.debug(f"Precision: {precision}")
        self.logger.debug(f"Recall: {recall}")

        results = {
            Results.Metrics.ACCURACY  : accuracy,
            Results.Metrics.F1_SCORE  : f1_metric,
            Results.Metrics.PRECISION : precision,
            Results.Metrics.RECALL    : recall,
        }
        self.results.append(results)

    def save_results(self, fold):
        self.logger.debug("Saving results...")
        if not os.path.exists(Paths.Directories.RESULTS):
            os.mkdir(Paths.Directories.RESULTS)

        dirname = self._build_dirname()
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.logger.debug(f"Saving to {dirname}")

        fold_dirname = os.path.join(dirname, f"fold_{fold}")
        if not os.path.exists(fold_dirname):
            os.mkdir(fold_dirname)

        results_filename = os.path.join(fold_dirname, Paths.Files.RESULTS)

        with open(results_filename, 'w') as file:
            self._save_metrics(file)
            self._save_training_time(file)

        self._plot_metrics(fold_dirname)

