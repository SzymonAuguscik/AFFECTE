
from constants import Time
from math import floor

import scipy.signal
import logging
import wfdb
import sys


def format_time(time_fraction):
    hours = floor(time_fraction)
    minutes = (time_fraction - hours) * Time.MINUTES_IN_HOUR
    minutes = floor(minutes)
    seconds = int((minutes - minutes) * Time.SECONDS_IN_MINUTE)
    return f"{hours}:{minutes}:{seconds}"

def bandpass_filter(signal):
    b, a = scipy.signal.butter(N=5, Wn=(0.5, 35), btype="bandpass", fs=128)
    return scipy.signal.lfilter(b, a, signal)

def preprocess_signal(signal, fs):
    for i in range(signal.shape[1]):
        signal[:, i], _ = wfdb.processing.resample_sig(signal[:, i], fs, 128)
        signal[:, i] -= scipy.signal.medfilt(signal[:, i], int(0.2 * 128))
        signal[:, i] -= scipy.signal.medfilt(signal[:, i], int(0.6 * 128) + 1)
        signal[:, i] -= bandpass_filter(signal[:, i])
    return signal

def align_format(record):
    record.levelname = "[%s]" % record.levelname
    return True

def init_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)-10s:%(name)-40s:  %(message)s'))
    handler.addFilter(align_format)
    logger.addHandler(handler)
    return logger

