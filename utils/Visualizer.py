from matplotlib import pyplot as plt
from constants import Results

import logging


class Visualizer:
    def __init__(self):
        self.metrics = {
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

        self.logger = logging.getLogger(__name__)

    def update_metric(self, metric, value):
        self.metrics[metric].append(value)

    def plot_train_loss(self, filename):
        plt.clf()
        plt.title(Results.Visualization.Names.LOSS_TITLE)
        plt.xlabel(Results.Visualization.Names.X_LABEL)
        plt.ylabel(Results.Metrics.LOSS)
        plt.plot(range(0, len(self.metrics[Results.Metrics.LOSS])), self.metrics[Results.Metrics.LOSS])
        plt.savefig(filename, format=Results.Visualization.Files.EXTENSION)

    def plot_learning_curves(self, filename, *metrics):
        plt.clf()
        plt.title(Results.Visualization.Names.LR_TITLE)
        plt.xlabel(Results.Visualization.Names.X_LABEL)
        plt.ylabel(Results.Visualization.Names.LR_Y_LABEL)
        for metric in metrics:
            plt.plot(range(0, len(self.metrics[metric])), self.metrics[metric])
        plt.legend(metrics)
        plt.savefig(filename, format=Results.Visualization.Files.EXTENSION)

