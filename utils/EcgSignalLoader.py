from utils.Utils import preprocess_signal, format_time
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
    """
    EcgSignalLoader is used for preprocessing ECG gathered from subjects, including filtering signal
    and creating dataset for classification purposes.

    Attributes
    ----------
    _logger : logging.Logger
        Used for logging purposes.
    _tensor_manager : TensorManager
        Used for loading/saving features and labels extracted from ECG.
    _data_path : str
        Path to read raw data from.
    _subjects : List[str]
        List of subject annotations.
    _X : List[torch.Tensor]
        List of features per subject (ECG signal values).
    _y : List[torch.Tensor]
        List of labels per subject (arrhythmia annotations).
    #TODO change to enum "Mode"
    _af_sr_split : bool
        Determine if dataset should be split by AF and SR
        or AF and non-AF annotations.

    Examples
    --------
    loader = EcgSignalLoader("/path/to/data")
    Xs, ys = loader.prepare_dataset(channels=[0, 2, 3], seconds=5)

    for y in ys:
        # y is torch.Tensor
        y = y.int()

    """
    def __init__(self, data_path: str, af_sr_split: bool = False) -> None:
        """
        Initiate EcgSignalLoader with all subjects for _subjects and default values for other attributes.

        Parameters
        ----------
        data_path : str
            Path to read raw data from.
        af_sr_split : bool, optional
            Determine if dataset should be split by AF and SR
            or AF and non-AF annotations.

        """
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._tensor_manager: TensorManager = TensorManager()
        self._data_path: str = data_path
        self._subjects: List[str] = self._get_subjects(self._data_path)
        self._X: List[torch.Tensor] = []
        self._y: List[torch.Tensor] = []
        self._af_sr_split: bool = af_sr_split

    def _get_subjects(self, records_dir: str) -> List[str]:
        """
        Read all subject annotations from specified file.

        Parameters
        ----------
        records_dir : str
            A directory where subject annotations file is stored.

        Returns
        -------
        List[str]
            The annotations of all subjects.

        """
        records_path: str = os.path.join(Paths.Directories.DATA, records_dir, Paths.Files.RECORDS)
        with open(records_path) as file:
            return file.read().strip().split('\n')

    def _load_signal(self, records_dir: str, subject: str) -> Tuple[wfdb.Record, List[str], np.ndarray]:
        """
        Load ECG signal for specific subject.

        Parameters
        ----------
        records_dir : str
            A directory where subject signals are stored.
        subject : str
            Subject annotations to be loaded.

        Returns
        -------
        record : wfdb.Record
            An ECG record that includes e.g. sampling frequency or signal values.
        symbols : List[str]
            Arrhythmia annotations (for instance various arrhythmia types, normal beats or ambigous rhythm).
        samples : np.ndarray
            Indexes of arrhythmia annotations.

        """
        self._logger.debug("Loading signal")
        record_path: str = os.path.join(Paths.Directories.DATA, records_dir, subject)
        record: wfdb.Record = wfdb.rdrecord(record_path)
        subject_path: str = os.path.join(Paths.Directories.DATA, records_dir, subject)
        annotation: wfdb.Annotation = wfdb.rdann(subject_path, 'atr')
        symbols: List[str] = annotation.aux_note              
        samples: np.ndarray = annotation.sample
        return record, symbols, samples

    def _split_signal_by_af_and_sr(self, signal: np.ndarray, rhythm_intervals: Dict[str, List[Tuple[int, int]]], chunk_size: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Split singal into atrial fibrillation and sinus rhythm chunks.

        Parameters
        ----------
        signal : np.ndarray
            An ECG signal that will be split.
        rhythm_intervals : Dict[str, List[Tuple[int, int]]]
            A dictionary that each key is an annotation and each value is an interval indicating
            the start and the end point of a rhythm type.
        chunk_size : int
            The size of a single chunk.

        Returns
        -------
        chunks : List[np.ndarray]
            List of prepared chunks.
        labels : List[int]
            Labels for chunks (1 if atrial fibrillation, 0 if sinus rhythm).

        """
        chunks: List[np.ndarray] = []
        labels: List[int] = []

        for rhythm_type, intervals in rhythm_intervals.items():
            is_af: bool = rhythm_type == Tags.AF_SYMBOL
            is_sr: bool = rhythm_type == Tags.SR_SYMBOL

            for interval in intervals:
                for idx in range(interval[0], interval[1], chunk_size):
                    chunk: np.ndarray = signal[idx : idx + chunk_size]
                    
                    if len(chunk) == chunk_size:
                        if is_af or is_sr:
                            chunks.append(chunk)
                            labels.append(int(is_af))

        return chunks, labels        

    def _split_signal_by_af(self, signal: np.ndarray, rhythm_intervals: Dict[str, List[Tuple[int, int]]], chunk_size: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Split singal into atrial fibrillation and non attrial fibrillation chunks.

        Parameters
        ----------
        signal : np.ndarray
            An ECG signal that will be split.
        rhythm_intervals : Dict[str, List[Tuple[int, int]]]
            A dictionary that each key is an annotation and each value is an interval indicating
            the start and the end point of a rhythm type.
        chunk_size : int
            The size of a single chunk.

        Returns
        -------
        chunks : List[np.ndarray]
            List of prepared chunks.
        labels : List[int]
            Labels for chunks (1 if atrial fibrillation, 0 otherwise).

        """
        chunks: List[np.ndarray] = []
        labels: List[int] = []

        for rhythm_type, intervals in rhythm_intervals.items():
            is_af: bool = rhythm_type == Tags.AF_SYMBOL

            for interval in intervals:
                for idx in range(interval[0], interval[1], chunk_size):
                    chunk: np.ndarray = signal[idx : idx + chunk_size]
                    
                    if len(chunk) == chunk_size:
                        chunks.append(chunk)
                        labels.append(int(is_af))

        return chunks, labels

    def _create_data_from_subject(self, records_dir: str, subject: str, seconds: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Read data for given subject, perform filtration (signal preprocessing and removing unneeded signal annotations)
        and split signal based on atrial fibrillation.

        Parameters
        ----------
        records_dir : str
            A directory where subject signals are stored.
        subject : str
            Subject annotations to be loaded.
        seconds : int
            The length of a single chunk after signal is split (the size of a moving window).

        Returns
        -------
        Tuple[List[np.ndarray], List[int]]
            See _split_signal_by_af() or _split_signal_by_af_and_sr().

        """
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
        return self._split_signal_by_af_and_sr(signal, rhythm_intervals, chunk_size) if self._af_sr_split else \
               self._split_signal_by_af(signal, rhythm_intervals, chunk_size)

    def prepare_dataset(self, channels: List[int], seconds: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Create dataset from all available subjects ECG signals. Signal is split to seconds length windows.
        Only specified channels are taken into account. If dataset has not been created yet, it prepares the features and labels
        for all available channels. Otherwise, loads features (for specified channels only) and labels for given seconds split.

        Parameters
        ----------
        channels : List[int]
            Indicates which signal channels should be used for creating features.
        seconds : int
            The length of a single chunk after signal split.

        Returns
        -------
        self._X : List[torch.Tensor]
            List of features per subject (ECG signal values).
        self._y : List[torch.Tensor]
            List of labels per subject (arrhythmia annotations).

        """
        split_kind: str = "AF_SR" if self._af_sr_split else "AF_nonAF"
        dirname: str = os.path.join(Paths.Directories.DATA, Paths.Directories.DATASETS, f"{seconds}_seconds_{split_kind}")
        self._logger.debug(f"Checking {dirname}")

        if os.path.exists(dirname):
            self._logger.debug(f"Found directory {dirname} with already created dataset! Now loading it...")
            self._X = self._tensor_manager.load(os.path.join(dirname, Paths.Files.FEATURES))
            self._y = self._tensor_manager.load(os.path.join(dirname, Paths.Files.LABELS))
        else:
            self._logger.debug(f"Creating dataset...")
            self._logger.debug(f"{len(self._subjects)} subjects to be loaded")
            
            for subject in self._subjects:
                self._logger.debug(f"Reading subject no. {subject}")

                data: List[np.ndarray]
                labels: List[int]
                data, labels = self._create_data_from_subject(self._data_path, subject, seconds)

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

        if len(channels) >= 1:
            for i in range(len(self._X)):
                self._X[i] = self._X[i][:, channels, :]

        return self._X, self._y

