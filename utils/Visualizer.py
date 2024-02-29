from matplotlib import pyplot as plt
from constants import Results
from typing import Dict, List

import logging


class Visualizer:
    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._metrics: Dict[str, List[float]] = {
            Results.Metrics.LOSS            : [],
            Results.Metrics.TRAIN_ACCURACY  : [],
            Results.Metrics.TRAIN_F1_SCORE  : [],
            Results.Metrics.TRAIN_PRECISION : [],
            Results.Metrics.TRAIN_RECALL    : [],
            Results.Metrics.TEST_ACCURACY   : [],
            Results.Metrics.TEST_F1_SCORE   : [],
            Results.Metrics.TEST_PRECISION  : [],
            Results.Metrics.TEST_RECALL     : [],
        }

    def update_metric(self, metric: str, value: float) -> None:
        self._metrics[metric].append(value)

    def plot_train_loss(self, filename: str) -> None:
        plt.clf()
        plt.title(Results.Visualization.Names.LOSS_TITLE)
        plt.xlabel(Results.Visualization.Names.X_LABEL)
        plt.ylabel(Results.Metrics.LOSS)
        plt.plot(range(0, len(self._metrics[Results.Metrics.LOSS])), self._metrics[Results.Metrics.LOSS])
        plt.savefig(filename, format=Results.Visualization.Files.EXTENSION)

    def plot_learning_curves(self, filename: str, *metrics: str) -> None:
        plt.clf()
        plt.title(Results.Visualization.Names.LR_TITLE)
        plt.xlabel(Results.Visualization.Names.X_LABEL)
        plt.ylabel(Results.Visualization.Names.LR_Y_LABEL)
        for metric in metrics:
            plt.plot(range(0, len(self._metrics[metric])), self._metrics[metric])
        plt.legend(metrics)
        plt.savefig(filename, format=Results.Visualization.Files.EXTENSION)

