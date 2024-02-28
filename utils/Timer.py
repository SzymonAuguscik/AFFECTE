from typing import Optional

import logging
import time


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.measured_time: Optional[float] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

    def start(self) -> None:
        self.start_time = time.time()
        self.logger.debug(f"Start time = {self.start_time}s")

    def stop(self) -> None:
        if self.start_time is None:
            self.logger.error("Timer was not started!")
            return
        self.measured_time = time.time() - self.start_time
        self.logger.debug(f"Measured time = {self.measured_time}s")

    def get_start_time(self) -> None:
        return self.start_time

    def get_time(self) -> Optional[float]:
        if self.measured_time is not None:
            return self.measured_time
        # TODO exception
        self.logger.error("Time was not measured! Has the timer been started?")

