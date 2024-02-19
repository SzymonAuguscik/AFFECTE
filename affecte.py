"""
AFFECTE - Atrial Fibrillation Finder from Electrocardiogram with Convolution and Transformer Encoder
"""

from utils.funs import init_logger
from models.AtrialFibrillationDetector import AtrialFibrillationDetector
from utils.CrossValidator import CrossValidator
from utils.Learner import Learner
# from pyhrv.tools import plot_ecg
from utils.EcgSignalLoader import EcgSignalLoader
from constants import Paths

import numpy as np

import argparse
import random
import torch

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="affecte",
                                     description="Script to train and test model to classify atrial fibrillation",
                                     epilog="Atrial Fibrillation Finder from Electrocardiogram with Convolution and Transformer Encoder",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--channels", default=[0], nargs="+", choices=[0, 1], metavar="CHANNEL", type=int, help="ECG channels to be used in the training")
    parser.add_argument("-s", "--seconds", default=10, choices=range(1, 60), metavar="[1-60]", type=int, help="length of the windows that the ECG signal will be split")
    parser.add_argument("-e", "--epochs", default=100, choices=range(1, 500), metavar="[1-500]", type=int, help="training iterations")
    parser.add_argument("-l", "--learning_rate", default=1e-3, type=float, help="learning rate to be used in the optimizer")
    parser.add_argument("-b", "--batch_size", default=128, choices=range(1, 2048), metavar="[1-2048]", type=int, help="the data samples used per epoch")
    parser.add_argument("-t", "--transformer_dimension", default=128, choices=range(1, 2048), metavar="[1-2048]", type=int, help="d_model for Transformer")
    parser.add_argument("-f", "--transformer_hidden_dimension", default=256, choices=range(1, 2048), metavar="[1-2048]", type=int, help="dimension of feed forward layers in Transformer")
    parser.add_argument("-a", "--transformer_heads", default=16, choices=range(1, 128), metavar="[1-128]", type=int, help="number of attention heads in Transformer")
    parser.add_argument("-n", "--transformer_encoder_layers", default=8, choices=range(1, 64), metavar="[1-64]", type=int, help="number of encoding layers in Transformer")
    parser.add_argument("-d", "--dataset_custom_size", choices=range(1, 1_000_000), metavar="[1-1000000]", type=int, help="set how many samples should be used from dataset; use all samples if not set")
    parser.add_argument("--use_cnn", action="store_true", help="if set, use CNN layer in the main model")
    parser.add_argument("--use_transformer", action="store_true", help="if set, use Transformer layer in the main model")
    args = parser.parse_args()

    logger = init_logger()
    logger.info("Loading subjects...")

    channels = args.channels
    seconds = args.seconds
    logger.debug(f"Data will be split into {seconds} seconds intervals")

    ecg_signal_loader = EcgSignalLoader(Paths.Directories.LONG_TERM_AF)
    X, y = ecg_signal_loader.prepare_dataset(channels=channels, seconds=seconds)
    # print(X[0].size())
    # num = 0
    # i = 3
    # fig = plot_ecg(X[num][i, 0, :], rpeaks=False, sampling_rate=128, interval=[0,10])
    # print(y[num][i])
    # fig["ecg_plot"].savefig(f"subject_{num}_{'AF' if y[num][i] else 'SR'}.svg")
    cross_validator = CrossValidator(X=X, y=y, dataset_custom_size=args.dataset_custom_size)
    # cross_validator.prepare()
    # cross_validator.do_cleanup()
    
    for i, subjects_split in enumerate(cross_validator):
        logger.info(f"Cross validation fold #{i + 1}")

        X_train, y_train, X_test, y_test = cross_validator.prepare_fold(subjects_split)

        logger.info(f"Number of training examples: {len(X_train)}")
        logger.info(f"Number of test examples: {len(X_test)}")
        logger.info(f"Arrhythmia fraction = {(sum(y_train) + sum(y_test)) / (len(y_train) + len(y_test))}")
        
        learner = Learner(model=AtrialFibrillationDetector(ecg_channels=len(channels),
                                                           window_length=X_train[0].size(1),
                                                           transformer_dimension=args.transformer_dimension,
                                                           transformer_hidden_dimension=args.transformer_hidden_dimension,
                                                           transformer_heads=args.transformer_heads,
                                                           transformer_encoder_layers=args.transformer_encoder_layers,
                                                           use_cnn=args.use_cnn,
                                                           use_transformer=args.use_transformer),
                          X_train=X_train,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test,
                          seconds=seconds,
                          lr=args.learning_rate,
                          batch_size=args.batch_size,
                          epochs=args.epochs)
        learner.train()
        learner.test(X_train, y_train)
        learner.test(X_test, y_test)
        learner.save_results(fold=str(i))
        break
