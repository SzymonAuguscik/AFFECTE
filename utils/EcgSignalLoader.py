from utils.funs import preprocess_signal, format_time
from utils.TensorManager import TensorManager
from constants import Tags, Time, Paths

import numpy as np

import wfdb.processing
import logging
import torch
import wfdb
import os


class EcgSignalLoader:
    def __init__(self, dataset_path):
        self.logger = logging.getLogger(__name__)
        self.tensor_manager = TensorManager()
        self.dataset_path = dataset_path
        self.subjects = self._get_subjects(self.dataset_path)
        self.X = []
        self.y = []

    def _get_subjects(self, records_dir):
        with open(os.path.join(Paths.Directories.DATA, records_dir, Paths.Files.RECORDS)) as f:
            return f.read().strip().split('\n')

    def _load_signal(self, records_dir, subject):
        self.logger.debug("Loading signal")
        record = wfdb.rdrecord(os.path.join(Paths.Directories.DATA, records_dir, subject))
        annotation = wfdb.rdann(os.path.join(Paths.Directories.DATA, records_dir, subject), 'atr')
        symbols = annotation.aux_note              
        samples = annotation.sample
        return (record, symbols, samples)

    def _split_signal(self, signal, rhythm_intervals, chunk_size):
        data, labels = [], []

        for rhythm_type, intervals in rhythm_intervals.items():
            is_af = int(rhythm_type == Tags.AF_SYMBOL)

            for interval in intervals:
                for idx in range(interval[0], interval[1], chunk_size):
                    chunk = signal[idx : idx + chunk_size]
                    
                    if len(chunk) == chunk_size:
                        data.append(chunk)
                        labels.append(is_af)

        return data, labels

    def _create_data_from_subject(self, records_dir, subject, seconds):
        self.logger.info(f"Subject: {subject}")
        record, symbols, samples = self._load_signal(records_dir, subject)
        start_sample, start_symbol = 0, None
        rhythm_intervals = {rhythm_type : [] for rhythm_type in Tags.ARRHYTHMIA_SYMBOLS}

        for sample, symbol in zip(samples, symbols):
            if symbol in Tags.SYMBOLS_TO_IGNORE:
                continue

            if start_symbol is not None:
                rhythm_interval = (start_sample, sample)
                rhythm_intervals[start_symbol].append(rhythm_interval)

            start_symbol = symbol
            start_sample = sample
        
        self.logger.debug("Preprocessing signal: resampling, 2 median filters and bandpass filter")
        signal = preprocess_signal(record.p_signal, record.fs)
        rhythm_intervals[start_symbol].append((start_sample, len(signal)))
        rhythm_intervals = { symbol : rhythm_intervals[symbol] for symbol in Tags.CLASSIFICATION_SYMBOLS }
        rhythms = { rhythm_type : sum(list(map(lambda interval: interval[1] - interval[0], intervals))) for rhythm_type, intervals in rhythm_intervals.items() }

        self.logger.info(f"{[(rhythm_type, format_time(rhythm / Time.MINUTES_IN_HOUR / Time.SECONDS_IN_MINUTE / record.fs)) for rhythm_type, rhythm in rhythms.items()]}")
        
        chunk_size = seconds * record.fs
        return self._split_signal(signal, rhythm_intervals, chunk_size)

    def prepare_dataset(self, channels, seconds):
        dirname = os.path.join(Paths.Directories.DATA, Paths.Directories.DATASETS, f"{seconds}_seconds")
        self.logger.debug(f"Checking {dirname}")

        if os.path.exists(dirname):
            self.logger.debug(f"Found directory {dirname} with already created dataset! Now loading it...")
            self.X = self.tensor_manager.load(os.path.join(dirname, Paths.Files.FEATURES))
            self.y = self.tensor_manager.load(os.path.join(dirname, Paths.Files.LABELS))

            if len(channels) >= 1:
                for i in range(len(self.X)):
                    self.X[i] = self.X[i][:, channels, :]
        else:
            self.logger.debug(f"Creating dataset...")
            self.logger.debug(f"{len(self.subjects)} subjects to be loaded")
            
            for subject in self.subjects:
                self.logger.debug(f"Reading subject no. {subject}")
                data, labels = self._create_data_from_subject(self.dataset_path, subject, seconds=seconds)
                X_subject = torch.tensor(np.array(data), dtype=torch.float32).permute(0, 2, 1)
                y_subject = torch.tensor(np.array(labels), dtype=torch.float32).reshape(-1, 1)
                self.X.append(X_subject)
                self.y.append(y_subject)

            os.mkdir(dirname)
            self.logger.debug(f"Saving dataset to {dirname}")
            self.tensor_manager.save(self.X, os.path.join(dirname, Paths.Files.FEATURES))
            self.tensor_manager.save(self.y, os.path.join(dirname, Paths.Files.LABELS))

            self.logger.info("Dataset ready!")
            self.logger.info(f"No. of samples: {len(torch.cat(self.y))}")

        return self.X, self.y

