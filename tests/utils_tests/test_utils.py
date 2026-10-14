import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.utils import Utils


class TestUtils:
    # format_time
    TIME_IN_HOURS: float = 13.67
    TIME_FORMAT: str = "13:40:12"

    # bandpass_filter
    SIGNAL_TO_FILTER: np.ndarray = np.array(
        [[19, 76, 45, 40, 91, 17, 25], [60, 21, 95, 71, 59, 81, 39]]
    )

    FILTERED_SIGNAL: np.ndarray = np.array(
        [
            [
                1.34805629,
                11.43950728,
                36.63943713,
                57.47209883,
                49.30100348,
                33.9228314,
                32.59175484,
            ],
            [
                4.25701987,
                20.58663731,
                42.65846837,
                55.86822252,
                61.10478215,
                58.20728138,
                43.49388091,
            ],
        ]
    )

    # preprocess_signal
    SIGNAL_TO_PREPROCESS: np.ndarray = np.array(
        [
            [1.81, 0.91, 0.41, -0.51, 0.02, -0.25, -0.56],
            [0.51, 0.76, 0.49, 0.32, 0.21, 0.01, -0.23],
            [-0.23, -0.71, -1.25, -0.69, -0.28, 0.13, 0.43],
        ]
    )
    PREPROCESSED_SIGNAL: np.ndarray = np.array(
        [
            [
                1.6815799,
                0.8454352,
                0.38091036,
                -0.47381533,
                0.01858099,
                -0.23226242,
                -0.52026781,
            ],
            [
                -0.61926288,
                -1.48649474,
                -1.7051663,
                -0.05327545,
                -0.27635228,
                0.10096116,
                0.64764015,
            ],
            [
                -0.54511183,
                -1.21889212,
                -1.50501704,
                -0.24725778,
                -0.26554684,
                0.11541742,
                0.57422364,
            ],
        ]
    )
    FS: float = 128

    # align_format
    LOG_RECORD_NAME: str = "log record name"
    LOG_RECORD_LEVELNAME: str = "DEBUG"
    LOG_RECORD_PATH: str = "log record path"
    LOG_RECORD_LINE_NUMBER: int = 18
    LOG_RECORD_MESSAGE: str = "log record message"

    # init_logger
    LOGGER_NAME: str = "custom logger name"

    # add & add_noise_to_signal
    SIGNAL: torch.Tensor = torch.Tensor(
        [
            [[18, 92, 51, 61], [69, 81, 75, 72]],
            [[27, 96, 40, 56], [60, 51, 26, 71]],
            [[38, 81, 71, 48], [98, 12, 8, 54]],
        ]
    )
    SIGNAL_WITH_NEW_CHANNEL: torch.Tensor = torch.Tensor(
        [[[87, 173, 126, 133]], [[87, 147, 66, 127]], [[136, 93, 79, 102]]]
    )
    CHANNEL_1: int = 0
    CHANNEL_2: int = 1

    NOISE: np.array = np.array(
        [
            [[0.58, 0.61, 0.51, 0.41]],
            [[0.91, 0.35, 0.61, 0.56]],
            [[0.13, 0.29, 0.22, 0.81]],
        ]
    ).astype(np.float32)

    # get_enum_value
    ITEM: str = "some item"
    ENUM_VALUE: int = 10

    def test_format_time(self) -> None:
        assert Utils.format_time(self.TIME_IN_HOURS) == self.TIME_FORMAT

    def test_bandpass_filter(self) -> None:
        assert np.allclose(
            Utils.bandpass_filter(self.SIGNAL_TO_FILTER), self.FILTERED_SIGNAL
        )

    def test_preprocess_signal(self) -> None:
        signal = np.tile(self.SIGNAL_TO_PREPROCESS, (26, 1))
        preprocessed_signal = Utils.preprocess_signal(signal, self.FS)
        assert np.allclose(preprocessed_signal[[0, 38, 77]], self.PREPROCESSED_SIGNAL)

    def test_align_format(self) -> None:
        log_record: logging.LogRecord = logging.LogRecord(
            self.LOG_RECORD_NAME,
            logging.DEBUG,
            self.LOG_RECORD_PATH,
            self.LOG_RECORD_LINE_NUMBER,
            self.LOG_RECORD_MESSAGE,
            args=None,
            exc_info=None,
        )
        assert log_record.levelname == self.LOG_RECORD_LEVELNAME
        Utils.align_format(log_record)
        assert log_record.levelname == f"[{self.LOG_RECORD_LEVELNAME}]"

    def test_initialize_logger(self) -> None:
        logger: logging.Logger = Utils.init_logger(self.LOGGER_NAME)
        assert logger.level == logging.INFO
        assert logger.name == self.LOGGER_NAME
        assert len(logger.handlers) == 1
        assert len(logger.handlers[0].filters) == 1
        assert logger.handlers[0].filters[0] == Utils.align_format

    def test_add_two_signal_channels(self) -> None:
        signal_with_new_channel: torch.Tensor = Utils.add(
            self.SIGNAL, self.CHANNEL_1, self.CHANNEL_2
        )
        assert torch.equal(signal_with_new_channel, self.SIGNAL_WITH_NEW_CHANNEL)

    @patch("numpy.random.normal", side_effect=[NOISE])
    def test_add_noise_to_two_signal_channels(self, normal_mock) -> None:
        signal_with_noise: torch.Tensor = Utils.add_noise_to_signal(
            self.SIGNAL, self.CHANNEL_1, self.CHANNEL_2
        )
        assert torch.equal(signal_with_noise, self.SIGNAL + self.NOISE)

    @patch("enum.Enum")
    def test_invalid_enum_value(self, enum_mock) -> None:
        with pytest.raises(ValueError):
            Utils.get_enum_value(self.ITEM, enum_mock, ValueError)

    @patch("enum.Enum")
    def test_valid_enum_value(self, enum_mock) -> None:
        enum_mock.__contains__.return_value = lambda _, item: item == self.ITEM
        enum_mock.__getitem__.return_value = self.ENUM_VALUE
        item: type(enum_mock) = Utils.get_enum_value(self.ITEM, enum_mock, Exception)
        enum_mock.__contains__.assert_called_with(self.ITEM)
        enum_mock.__getitem__.assert_called_with(self.ITEM)
        assert item == self.ENUM_VALUE
