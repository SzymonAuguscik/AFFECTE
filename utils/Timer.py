from typing import Optional

import logging
import time


class Timer:
    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._measured_time: Optional[float] = None
        self._logger: logging.Logger = logging.getLogger(__name__)

    def start(self) -> None:
        self._start_time = time.time()
        self._logger.debug(f"Start time = {self._start_time}s")

    def stop(self) -> None:
        if self._start_time is None:
            self._logger.error("Timer was not started!")
            return
        self._measured_time = time.time() - self._start_time
        self._logger.debug(f"Measured time = {self._measured_time}s")

    def get_start_time(self) -> None:
        return self._start_time

    def get_time(self) -> Optional[float]:
        if self._measured_time is not None:
            return self._measured_time
        # TODO exception
        self._logger.error("Time was not measured! Has the timer been started?")

