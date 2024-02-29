from utils.funs import preprocess_signal, format_time
from typing import List, Tuple, Optional, Dict
from utils.TensorManager import TensorManager
from constants import Tags, Time, Paths

import numpy as np

import wfdb.processing
import logging
import torch
import wfdb
import os


class EcgSignalLoader:
    def __init__(self, dataset_path: str) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._tensor_manager: TensorManager = TensorManager()
        self._dataset_path: str = dataset_path
        self._subjects: List[str] = self._get_subjects(self._dataset_path)
        self._X: List[torch.Tensor] = []
        self._y: List[torch.Tensor] = []

    def _get_subjects(self, records_dir: str) -> List[str]:
        records_path: str = os.path.join(Paths.Directories.DATA, records_dir, Paths.Files.RECORDS)
        with open(records_path) as file:
            return file.read().strip().split('\n')

    def _load_signal(self, records_dir: str, subject: str) -> Tuple[wfdb.Record, List[str], np.ndarray]:
        self._logger.debug("Loading signal")
        record_path: str = os.path.join(Paths.Directories.DATA, records_dir, subject)
        record: wfdb.Record = wfdb.rdrecord(record_path)
        subject_path: str = os.path.join(Paths.Directories.DATA, records_dir, subject)
        annotation: wfdb.Annotation = wfdb.rdann(subject_path, 'atr')
        symbols: List[str] = annotation.aux_note              
        samples: np.ndarray = annotation.sample
        return record, symbols, samples

    def _split_signal(self, signal: np.ndarray, rhythm_intervals: Dict[str, List[Tuple[int, int]]], chunk_size: int) -> Tuple[List[np.ndarray], List[int]]:
        data: List[np.ndarray] = []
        labels: List[int] = []

        for rhythm_type, intervals in rhythm_intervals.items():
            is_af: bool = rhythm_type == Tags.AF_SYMBOL

            for interval in intervals:
                for idx in range(interval[0], interval[1], chunk_size):
                    chunk: np.ndarray = signal[idx : idx + chunk_size]
                    
                    if len(chunk) == chunk_size:
                        data.append(chunk)
                        labels.append(int(is_af))

        return data, labels

    def _create_data_from_subject(self, records_dir: str, subject: str, seconds: int) -> Tuple[List[np.ndarray], List[int]]:
        self._logger.info(f"Subject: {subject}")
        record: wfdb.Record
        symbols: List[str]
        samples: np.ndarray
        record, symbols, samples = self._load_signal(records_dir, subject)

        start_sample: int = 0
        start_symbol: Optional[str] = None
        rhythm_intervals: Dict[str, List[Tuple[int, int]]] = {rhythm_type : [] for rhythm_type in Tags.ARRHYTHMIA_SYMBOLS}

        for sample, symbol in zip(samples, symbols):
            if symbol in Tags.SYMBOLS_TO_IGNORE:
                continue

            if start_symbol is not None:
                rhythm_interval: Tuple[int, int] = (start_sample, sample)
                rhythm_intervals[start_symbol].append(rhythm_interval)

            start_symbol = symbol
            start_sample = sample
        
        self._logger.debug("Preprocessing signal: resampling, 2 median filters and bandpass filter")
        signal: np.ndarray = preprocess_signal(record.p_signal, record.fs)
        rhythm_intervals[start_symbol].append((start_sample, len(signal)))
        rhythm_intervals = { symbol : rhythm_intervals[symbol] for symbol in Tags.CLASSIFICATION_SYMBOLS }
        rhythms: Dict[str, int] = { rhythm_type : sum(list(map(lambda interval: interval[1] - interval[0], intervals))) for rhythm_type, intervals in rhythm_intervals.items() }

        self._logger.info(f"{[(rhythm_type, format_time(rhythm / Time.MINUTES_IN_HOUR / Time.SECONDS_IN_MINUTE / record.fs)) for rhythm_type, rhythm in rhythms.items()]}")
        
        chunk_size: int = int(seconds * record.fs)
        return self._split_signal(signal, rhythm_intervals, chunk_size)

    def prepare_dataset(self, channels, seconds) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        dirname: str = os.path.join(Paths.Directories.DATA, Paths.Directories.DATASETS, f"{seconds}_seconds")
        self._logger.debug(f"Checking {dirname}")

        if os.path.exists(dirname):
            self._logger.debug(f"Found directory {dirname} with already created dataset! Now loading it...")
            self._X = self._tensor_manager.load(os.path.join(dirname, Paths.Files.FEATURES))
            self._y = self._tensor_manager.load(os.path.join(dirname, Paths.Files.LABELS))

            if len(channels) >= 1:
                for i in range(len(self._X)):
                    self._X[i] = self._X[i][:, channels, :]
        else:
            self._logger.debug(f"Creating dataset...")
            self._logger.debug(f"{len(self._subjects)} subjects to be loaded")
            
            for subject in self._subjects:
                self._logger.debug(f"Reading subject no. {subject}")

                data: List[np.ndarray]
                labels: List[int]
                data, labels = self._create_data_from_subject(self._dataset_path, subject, seconds=seconds)

                X_subject: torch.Tensor = torch.tensor(np.array(data), dtype=torch.float32).permute(0, 2, 1)
                y_subject: torch.Tensor = torch.tensor(np.array(labels), dtype=torch.float32).reshape(-1, 1)
                self._X.append(X_subject)
                self._y.append(y_subject)

            os.mkdir(dirname)
            self._logger.debug(f"Saving dataset to {dirname}")
            self._tensor_manager.save(self._X, os.path.join(dirname, Paths.Files.FEATURES))
            self._tensor_manager.save(self._y, os.path.join(dirname, Paths.Files.LABELS))

            self._logger.info("Dataset ready!")
            self._logger.info(f"No. of samples: {len(torch.cat(self._y))}")

        return self._X, self._y

