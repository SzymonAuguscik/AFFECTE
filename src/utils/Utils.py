from src.constants import Time
from typing import List
from math import floor

import numpy as np

import scipy.signal
import logging
import torch
import wfdb
import sys


def format_time(time_fraction: float) -> str:
    """"
    Format time from hours to hours, minutes, and seconds.

    Parameters
    ----------
    time_fraction : float
        Time in hours.

    Returns
    -------
    str
        Time in "HH:MM:SS" format.

    """
    hours: int = floor(time_fraction)
    precise_minutes: float = (time_fraction - hours) * Time.MINUTES_IN_HOUR
    minutes: int = floor(precise_minutes)
    seconds: int = int((precise_minutes - minutes) * Time.SECONDS_IN_MINUTE)
    return f"{hours}:{minutes}:{seconds}"

def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """
    Apply bandpass filter on the signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal to be filtered.

    Returns
    -------
    np.ndarray
        Filtered signal.

    """
    b, a = scipy.signal.butter(N=5, Wn=(0.5, 35), btype="bandpass", fs=128)
    return scipy.signal.lfilter(b, a, signal)

def preprocess_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply transformations to the signal:
    - resampling to 128 Hz
    - median filter - 200 ms
    - median filter - 600 ms
    - bandpass filter to 0.5-35 Hz.
    Each transformation is applied separately to the signal channels.

    Parameters
    ----------
    signal : np.ndarray
        Signal to be processed.
    fs : float
        Sampling frequency of the signal.

    Returns
    -------
    signal : np.ndarray
        Preprocessed signal.

    """
    for i in range(signal.shape[1]):
        signal[:, i], _ = wfdb.processing.resample_sig(signal[:, i], fs, 128)
        signal[:, i] -= scipy.signal.medfilt(signal[:, i], int(0.2 * 128))
        signal[:, i] -= scipy.signal.medfilt(signal[:, i], int(0.6 * 128) + 1)
        signal[:, i] -= bandpass_filter(signal[:, i])
    return signal

def align_format(record: logging.LogRecord) -> bool:
    """
    Align format for logging purposes (e.g. in handlers).

    Parameters
    ----------
    record : logging.LogRecord
        A single text line regirested by logger.

    Returns
    -------
    bool
        Always True - each record should be processed.

    See also
    --------
    https://docs.python.org/3/library/logging.html

    """
    record.levelname = f"[{record.levelname}]"
    return True

def init_logger(name: str = None) -> logging.Logger:
    """
    Create default logger used in classes across project.

    Parameters
    ----------
    name : str, optional
        Logger name.

    Returns
    -------
    logger : logging.Logger
        Logger with custom formatting and stream handler.

    """
    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter = logging.Formatter('%(asctime)s:%(levelname)-10s:%(name)-40s:  %(message)s')
    handler.setFormatter(formatter)
    handler.addFilter(align_format)
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def add(ecg_signal: torch.Tensor, channel_1: int, channel_2: int) -> torch.Tensor:
    """
    Add two ECG channels.

    Parameters
    ----------
    ecg_signal : torch.Tensor
        Source ECG signal of N x C x V size (N - samples number, C - channels, V - values).
    channel_1 : int
        First summand channel number (corresponding to N x 1 x V size).
    channel_2 : int
        Second summand channel number (corresponding to N x 1 x V size).

    Returns
    -------
        New ECG signal channel (N x 1 x V size).

    """
    return (ecg_signal[:,channel_1,:] + ecg_signal[:,channel_2,:]).unsqueeze(1)

def add_noise_to_signal(ecg_signal: torch.Tensor, *channels: List[int]) -> torch.Tensor:
    """
    Add Gaussian noise to the signal.

    Parameters
    ----------
    ecg_signal : torch.Tensor
        Source ECG signal of N x C x V size (N - samples number, C - channels, V - values).
    channels : List[int]
        Channel numbers to add noise (corresponding to N x len(channels) x V size).

    Returns
    -------
        New ECG signal channel (N x C x V size).
    """
    mean = torch.mean(ecg_signal[:, channels, :])
    std = torch.std(ecg_signal[:, channels, :])
    return ecg_signal[:, channels, :] + np.random.normal(mean, std, ecg_signal[:, channels, :].size()).astype(np.float32)

