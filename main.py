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

import random
import numpy as np
import torch

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if __name__ == "__main__":
    logger = init_logger()
    logger.info("Loading subjects...")

    channels = [0]
    seconds = 10
    logger.debug(f"Data will be split into {seconds} seconds intervals")

    ecg_signal_loader = EcgSignalLoader(Paths.Directories.LONG_TERM_AF)
    X, y = ecg_signal_loader.prepare_dataset(channels=channels, seconds=seconds)
    # print(X[0].size())
    # num = 0
    # i = 3
    # fig = plot_ecg(X[num][i, 0, :], rpeaks=False, sampling_rate=128, interval=[0,10])
    # print(y[num][i])
    # fig["ecg_plot"].savefig(f"subject_{num}_{'AF' if y[num][i] else 'SR'}.svg")
    cross_validator = CrossValidator(X=X, y=y)
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
                                                           transformer_dimension=128,
                                                           use_cnn=False,
                                                           use_transformer=False),
                          X_train=X_train,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test,
                          seconds=seconds,
                          lr=1e-3,
                          batch_size=100,
                          epochs=100)
        learner.train()
        learner.test(X_train, y_train)
        learner.test(X_test, y_test)
        learner.save_results(fold=str(i))
        break

