from utils.Utils import add
from typing import List, Callable
from enum import Enum

import logging
import torch


class EcgSignalAugmenter:
    """
    EcgSignalAugmenter can create new ECG channels based on already existing ones.
    It uses several augmentation strategies which specify the operation to prepare
    a new channel and channels to be used.

    Attributes
    ----------
    AugmentationStrategy : enum.Enum
        The set of supported augmentation strategies.
    _logger : logging.Logger
        Used for logging purposes.
    _steps_file_path : str
        The path to the augmentation steps file.
        Each row is a name of augmentation strategy followed by numbers of channels
        which will be used, e.g. "ADD 0 1" means that the augmented channel
        will be a summation of channels: 0 and 1.
    _ecg_signal : List[torch.Tensor]
        Original ECG signal which will be appended during augmentation.

    Examples
    --------
    X = <load features e.g. from EcgSignalLoader>
    ecg_signal_augmenter = EcgSignalAugmenter(X, Paths.Files.AUGMENTATION_CONFIG)
    X = ecg_signal_augmenter.augment()

    """
    AugmentationStrategy = Enum("AugmentationStrategy", ["ADD"])

    def __init__(self, X: List[torch.Tensor], steps_file_path: str) -> None:
        """
        Initiate EcgSignalAugmenter with the ECG signal to be augmented and augmentation steps filepath.

        Parameters
        ----------
        X : List[torch.Tensor]
            Original ECG signal.
        steps_file_path : str
            The path to the augmentation steps file.

        """
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._steps_file_path: str = steps_file_path
        self._ecg_signal: List[torch.Tensor] = X

    # TODO decorators?
    def _append_new_channel(self, fun: Callable[[torch.Tensor, List[int]], torch.Tensor], channels: List[int]) -> None:
        """
        Add new ECG channel based on provided function and its arguments.

        Parameters
        ----------
        fun : Callable[[torch.Tensor, List[int]], torch.Tensor]
            A function which takes channel numbers and construct
            a new channel using specified ones.
        channels : List[int]
            A list of ECG channels to be used when augmenting a new one.

        """
        self._ecg_signal: List[torch.Tensor] = [torch.cat((signal, fun(signal, *channels)), dim=1) for signal in self._ecg_signal]

    def _augment_on_strategy(self, strategy: AugmentationStrategy, channels: List[int]) -> None:
        """
        Add new ECG channel based on provided strategy and ECG channels.

        Parameters
        ----------
        strategy : AugmentationStrategy
            A strategy which specifies the function to create a new ECG channel.
        channels : List[int]
            A list of ECG channels to be used when augmenting a new one.

        """
        # TODO new strategies
        if strategy == self.AugmentationStrategy.ADD:
            self._logger.debug(f"Adding channels {channels}")
            self._append_new_channel(add, channels)
        else:
            #TODO exception
            self._logger.error(f"Unknown strategy {strategy}!")

    def augment(self) -> List[torch.Tensor]:
        """
        Augment ECG signal using augmentation steps file.
        The number of augmented channels is equal to the number
        of entries in the file.

        Returns
        -------
        List[torch.Tensor]
            The augmented ECG signal with new channels.

        """
        self._logger.info("Augmenting ECG data")
        with open(self._steps_file_path, 'r') as steps:
            for step in steps.readlines():
                strategy: str
                channels: List[int]
                strategy, *channels = step.split()
                channels = [int(channel) for channel in channels]
                self._augment_on_strategy(self.AugmentationStrategy[strategy], channels)

        return self._ecg_signal