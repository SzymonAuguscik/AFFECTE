from constants import Time
from math import floor

import numpy as np

import scipy.signal
import logging
import wfdb
import sys


def format_time(time_fraction: float) -> str:
    hours: int = floor(time_fraction)
    precise_minutes: float = (time_fraction - hours) * Time.MINUTES_IN_HOUR
    minutes: int = floor(minutes)
    seconds: int = int((minutes - precise_minutes) * Time.SECONDS_IN_MINUTE)
    return f"{hours}:{minutes}:{seconds}"

def bandpass_filter(signal) -> np.ndarray:
    b, a = scipy.signal.butter(N=5, Wn=(0.5, 35), btype="bandpass", fs=128)
    return scipy.signal.lfilter(b, a, signal)

def preprocess_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    for i in range(signal.shape[1]):
        signal[:, i], _ = wfdb.processing.resample_sig(signal[:, i], fs, 128)
        signal[:, i] -= scipy.signal.medfilt(signal[:, i], int(0.2 * 128))
        signal[:, i] -= scipy.signal.medfilt(signal[:, i], int(0.6 * 128) + 1)
        signal[:, i] -= bandpass_filter(signal[:, i])
    return signal

def align_format(record: logging.LogRecord) -> bool:
    record.levelname = f"[{record.levelname}]"
    return True

def init_logger(name:str = None) -> logging.Logger:
    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter = logging.Formatter('%(asctime)s:%(levelname)-10s:%(name)-40s:  %(message)s')
    handler.setFormatter(formatter)
    handler.addFilter(align_format)
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

