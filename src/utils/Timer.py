from src.exceptions.TimerNotStartedException import TimerNotStartedException
from src.exceptions.TimerNotStoppedException import TimerNotStoppedException
from typing import Optional

import logging
import time


class Timer:
    """
    Timer is used for time measurements of chosen operations.

    Attributes
    ----------
    _start_time : Optional[float]
        Set by start() at the beginning of the measurement.
    _measured_time : Optional[float]
        Set by stop() at the end of the measurement.
    _logger : logging.Logger
        Used for logging purposes.

    Examples
    --------
    timer = Timer()
    timer.start()
    ...
    timer.stop()
    measured_time = timer.get_time()

    """
    def __init__(self) -> None:
        """Initiate the timer with default logger."""
        self._start_time: Optional[float] = None
        self._measured_time: Optional[float] = None
        self._logger: logging.Logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start measurement by getting current time. Should be used before stop()."""
        self._start_time = time.time()
        self._logger.debug(f"Start time = {self._start_time}s")

    def stop(self) -> None:
        """Stop current measurement. Each call overwrites _measured_time."""
        if self._start_time is None:
            raise TimerNotStartedException()
        self._measured_time = time.time() - self._start_time
        self._logger.debug(f"Measured time = {self._measured_time}s")

    def get_time(self) -> Optional[float]:
        """
        Return measured time.

        Raises
        ------
        TimerNotStartedException
            when timer.stop() method was used before timer.start().

        Returns
        -------
        _measured_time : Optional[float]
            time between start() and last stop() calls.

        """
        if self._measured_time is not None:
            return self._measured_time
        raise TimerNotStoppedException()

