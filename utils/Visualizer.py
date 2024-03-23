from matplotlib import pyplot as plt
from constants import Results
from typing import Dict, List

import logging


class Visualizer:
    """
    Visualizer is a tool to plot the training/test results.

    Attributes
    ----------
    _logger : logging.Logger
        Used for logging purposes.
    _metrics : Dict[str, List[float]]
        Supported metrics and their values gathered throughout the training/test.

    Examples
    --------
    visualizer = Visualizer()
    losses = [204.5818, 197.583, 200.41251, 198.8182, 201.25124]
    test_f1_scores = [0.89, 0.91, 0.92, 0.88, 0.95]
    train_accuracies = [0.96, 0.95, 0.93, 0.99, 0.97]

    for loss, f1_score, accuracy in zip(losses, test_f1_scores, train_accuracies):
        visualizer.update_metric(Results.Metrics.LOSS, loss)
        visualizer.update_metric(Results.Metrics.TEST_F1_SCORE, f1_score)
        visualizer.update_metric(Results.Metrics.TRAIN_ACCURACY, accuracy)

    visualizer.plot_train_loss("/plots/train_loss")
    visualizer.plot_learning_curves("/plots/learning_curves", Results.Metrics.TEST_F1_SCORE, Results.Metrics.TRAIN_ACCURACY)

    """
    def __init__(self) -> None:
        """Initiate Visualizer with default logger and metrics: loss, accuracy, precision, recall, and F1 score"""
        self._logger: logging.Logger = logging.getLogger(__name__)
        # TODO consider removing default metrics and adding custom ones by user
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
        """
        Add a new value for specific metric.

        Parameters
        ----------
        metric : str
            Metric to be updated.
        value : float
            Another value of metric.

        """
        self._metrics[metric].append(value)

    def plot_train_loss(self, filename: str) -> None:
        """
        Plot training loss.

        Parameters
        ----------
        filename : str
            A path (without file extension) to save the plot.

        """
        plt.clf()
        plt.title(Results.Visualization.Names.LOSS_TITLE)
        plt.xlabel(Results.Visualization.Names.X_LABEL)
        plt.ylabel(Results.Metrics.LOSS)
        plt.plot(range(0, len(self._metrics[Results.Metrics.LOSS])), self._metrics[Results.Metrics.LOSS])
        plt.savefig(filename, format=Results.Visualization.Files.EXTENSION)

    def plot_learning_curves(self, filename: str, *metrics: str) -> None:
        """
        Plot learning curves.

        Parameters
        ----------
        filename : str
            A path (without file extension) to save the plot.
        *metrics : str
            Metrics to be visualized. They will be plotted in the same plot.

        """
        plt.clf()
        plt.title(Results.Visualization.Names.LR_TITLE)
        plt.xlabel(Results.Visualization.Names.X_LABEL)
        plt.ylabel(Results.Visualization.Names.LR_Y_LABEL)
        for metric in metrics:
            plt.plot(range(0, len(self._metrics[metric])), self._metrics[metric])
        plt.legend(metrics)
        plt.savefig(filename, format=Results.Visualization.Files.EXTENSION)

